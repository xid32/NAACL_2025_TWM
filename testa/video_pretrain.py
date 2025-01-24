'''
Adapted from https://github.com/salesforce/BLIP
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import deepspeed
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from models.blip_pretrain import blip_pretrain, blip_videoimage_pretrain
import utils
from pprint import pformat
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader


def get_deepspeed_config(args):
    config_params = {
        'train_batch_size': args.effective_batch_size,
    }

    use_fp16 = args.deepspeed_fp16
    use_amp = not args.deepspeed_fp16  # by default, if not use deepspeed fp16, will enable deepspeed amp

    if use_amp:
        config_params['amp'] = {
            'enabled': True,
            'opt_level': f'O{args.amp_opt_level}',
        }

    if use_fp16:
        config_params['fp16'] = {
            'enabled': True,
        }

    gradient_clip = args.max_grad_norm
    if gradient_clip:
        config_params['gradient_clipping'] = gradient_clip

    config_params['flops_profiler'] = {
        'enabled': False,
        'profile_step': 1,
        'module_depth': -1,
        'top_modules': 3,
        'detailed': True,
    }

    # config_params['logging'] = {
    #    'steps_per_print': 50,
    # }
    if hasattr(args, "zero_opt_stage") and args.zero_opt_stage > 0:
        config_params['zero_optimization'] = {
            'stage': args.zero_opt_stage,
        }
        if args.zero_opt_stage > 0:
            config_params['fp16'] = {
                'enabled': True
            }
        config_params['zero_allow_untested_optimizer'] = True

    print(pformat(config_params))
    return config_params


def fp32_to_fp16(inputs):
    # deepspeed does not auto cast inputs.
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            v = v.to(dtype=torch.half)
        inputs[k] = v
    return inputs


def mixed_precision_init(args, model, optimizer):
    if args.mixed_precision_method == "deepspeed":
        config = get_deepspeed_config(args)
        model, optimizer, _, _ = deepspeed.initialize(
            config_params=config,
            model=model,
            optimizer=optimizer
        )
    '''
    else:
        # opt_level is O0, Apex will run as fp32
        model, optimizer = amp.initialize(
            model, optimizer,
            enabled=True,
            opt_level=f'O{args.amp_opt_level}')
        if args.distributed:
            model = DDP(model)
    '''
    return args, model, optimizer


def train(model, data_loader, optimizer, epoch, device, config, args, scaler):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    data_loader.sampler.set_epoch(epoch)

    for i, (video, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if epoch == 0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

        B, N, C, H, W = video.size()
        if model.module.timesformer is False:
            video = video.view(-1, C, H, W)
        else:
            video = video.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video = video.to(device, non_blocking=True)

        # ramp up alpha in the first 2 epochs
        alpha = config['alpha'] * min(1, (epoch * len(data_loader) + i) / (2 * len(data_loader)))

        if args.mixed_precision_method == 'deepspeed':
            inputs = {"video": video, 'caption': caption}
            inputs = fp32_to_fp16(inputs)
            video = inputs['video']
            caption = inputs['caption']
            loss_ita, loss_itm, loss_lm = model(video, caption, alpha, video=True, B=B)
            loss = loss_ita + loss_itm + loss_lm
        elif args.mixed_precision_method == 'apex':
            with autocast():
                loss_ita, loss_itm, loss_lm = model(video, caption, alpha, video=True, B=B)
                loss = loss_ita + loss_itm + loss_lm
        else:
            loss_ita, loss_itm, loss_lm = model(video, caption, alpha, video=True, B=B)
            loss = loss_ita + loss_itm + loss_lm

        if args.mixed_precision_method == 'apex':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        elif args.mixed_precision_method == 'deepspeed':
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train_two_loaders(model, loader1, loader2, optimizer, epoch, device, config, args, scaler, max_steps):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    loader1.sampler.set_epoch(epoch)
    loader2.sampler.set_epoch(epoch)

    # loader1_iter = iter(loader1)
    # loader2_iter = iter(loader2)

    for i, ((video, caption), (video_2, caption_2)) in enumerate(
            metric_logger.log_every(zip(loader1, loader2), print_freq, header, tot=max_steps)):

        # video, caption = next(loader1_iter)
        # video_2, caption_2 = next(loader2_iter)

        if i > max_steps:
            break

        if epoch == 0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

        B1, N, C, H, W = video.size()
        if model.module.timesformer is False:
            video = video.view(-1, C, H, W)
        else:
            video = video.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video = video.to(device, non_blocking=True)

        B2, N, C, H, W = video_2.size()
        if model.module.timesformer is False:
            video_2 = video_2.view(-1, C, H, W)
        else:
            video_2 = video_2.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video_2 = video_2.to(device, non_blocking=True)

        # ramp up alpha in the first 2 epochs
        alpha = config['alpha'] * min(1, (epoch + i / max_steps) / (2 * max_steps))

        if args.mixed_precision_method == 'deepspeed':
            inputs = {"video": video, 'caption': caption}
            inputs = fp32_to_fp16(inputs)
            video = inputs['video']
            caption = inputs['caption']
            inputs_2 = {"video": video_2, 'caption': caption_2}
            inputs_2 = fp32_to_fp16(inputs_2)
            video_2 = inputs_2['video']
            caption_2 = inputs_2['caption']
            loss_ita, loss_itm, loss_lm = model(video, caption, B1, video_2, caption_2, B2, alpha)
            loss = loss_ita + loss_itm + loss_lm
        elif args.mixed_precision_method == 'apex':
            with autocast():
                loss_ita, loss_itm, loss_lm = model(video, caption, B1, video_2, caption_2, B2, alpha)
                loss = loss_ita + loss_itm + loss_lm
        else:
            loss_ita, loss_itm, loss_lm = model(video, caption, B1, video_2, caption_2, B2, alpha)
            loss = loss_ita + loss_itm + loss_lm

        if args.mixed_precision_method == 'apex':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        elif args.mixed_precision_method == 'deepspeed':
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config, img_config=None):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    scaler = None
    #### Mixed precision
    if args.mixed_precision_method == "apex":
        fp16_trainning = f"apex O{args.amp_opt_level}"
        scaler = GradScaler()
    elif args.mixed_precision_method == "deepspeed":
        amp_info = '' if args.deepspeed_fp16 else f'amp, {args.amp_opt_level}'
        fp16_info = '' if not args.deepspeed_fp16 else f'fp16, {args.zero_opt_stage}'
        fp16_trainning = f"deepspeed, {amp_info}{fp16_info}"
    else:
        fp16_trainning = None

    print("16-bits training: {}".format(fp16_trainning))

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('pretrain_video', config, min_scale=0.5)]
    print('number of training samples: %d' % len(datasets[0]))
    if img_config is not None:
        img_datasets = [create_dataset('pretrain_video', img_config, min_scale=0.5)]
        print('number of image training samples: %d' % len(img_datasets[0]))
        video_steps = len(datasets[0]) // (config['batch_size'] * args.world_size)
        img_steps = len(img_datasets[0]) // (img_config['batch_size'] * args.world_size)
        max_steps = min(video_steps, img_steps)
        print("video bs:{}, video max steps:{}".format(len(datasets[0]), video_steps))
        print("image bs:{}, image max steps:{}".format(len(img_datasets[0]), img_steps))
        print("So max steps={}".format(max_steps))
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    if img_config is not None:
        img_samplpers = create_sampler(img_datasets, [True], num_tasks, global_rank)

    data_loader = \
    create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[True],
                  collate_fns=[None])[0]
    if img_config is not None:
        img_data_loader = \
        create_loader(img_datasets, img_samplpers, batch_size=[img_config['batch_size']], num_workers=[4],
                      is_trains=[True], collate_fns=[None])[0]

        #### Model ####
    print("Creating model")
    if img_config is not None:
        model = blip_videoimage_pretrain(pretrained=config['pretrained'], image_size=config['image_size'],
                                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                         vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'],
                                         model_cfg=config)
    else:
        model = blip_pretrain(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                              vit_grad_ckpt=config['vit_grad_ckpt'],
                              vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'],
                              model_cfg=config)

    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    weight_decay = args.weight_decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_param_tp],
         'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_param_tp],
         'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['init_lr'], eps=args.adam_epsilon)

    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'optimizer' in checkpoint.keys() and 'epoch' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print('resume checkpoint from %s' % args.checkpoint)

    if args.mixed_precision_method:
        args.effective_batch_size = config['batch_size'] * args.num_gpus
        args, model, optimizer = mixed_precision_init(args, model, optimizer)

    model_without_ddp = model
    if args.distributed:
        if args.mixed_precision_method != 'deepspeed':
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['max_epoch']):

        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
        if img_config is not None:
            train_stats = train_two_loaders(model, data_loader, img_data_loader, optimizer, epoch, device, config, args,
                                            scaler, max_steps)
        else:
            train_stats = train(model, data_loader, optimizer, epoch, device, config, args, scaler)
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain_video.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--rank', default=0, type=int, help='rank of process')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--mixed_precision_method', type=str, default=None)
    parser.add_argument('--amp_opt_level', type=int, default=1)
    parser.add_argument('--deepspeed_fp16', action='store_true')
    parser.add_argument('--zero_opt_stage', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--sep_image', action='store_true')
    parser.add_argument('--img_config', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    img_config = None
    if args.sep_image and (args.img_config is not None):
        img_config = yaml.load(open(args.img_config, 'r'), Loader=yaml.Loader)
        yaml.dump(img_config, open(os.path.join(args.output_dir, 'img_config.yaml'), 'w'))

    main(args, config, img_config)
