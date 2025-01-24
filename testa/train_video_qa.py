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
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.blip import load_checkpoint
from models.testa_vqa import testa_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import deepspeed
from pprint import pformat
from data.video_dataset import ivqa


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


def train(model, data_loader, optimizer, epoch, device, scaler):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100

    for i, (video, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        B, N, C, H, W = video.size()
        video = video.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video, weights = video.to(device, non_blocking=True), weights.to(device, non_blocking=True)

        if args.mixed_precision_method == 'deepspeed':
            inputs = {"video": video, "question": question, "answer": answer}
            inputs = fp32_to_fp16(inputs)
            video, question, answer = inputs["video"], inputs["question"], inputs["answer"]
            loss = model(video, question, answer, train=True, n=n, weights=weights)
        elif args.mixed_precision_method == 'apex':
            with autocast():
                loss = model(video, question, answer, train=True, n=n, weights=weights)
        else:
            loss = model(video, question, answer, train=True, n=n, weights=weights)

        if args.mixed_precision_method == 'apex':
            scaler.scale(loss).backward()
            if ((i + 1) % args.accumulation_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        elif args.mixed_precision_method == 'deepspeed':
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            if ((i + 1) % args.accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 100

    result = []

    if config['inference'] == 'rank':
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
        answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id

    for n, (video, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        B, N, C, H, W = video.size()
        video = video.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video = video.to(device, non_blocking=True)

        if config['inference'] == 'generate':
            if args.mixed_precision_method == 'apex':
                with autocast():
                    answers = model(video, question, train=False, inference='generate')
            else:
                answers = model(video, question, train=False, inference='generate')

            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())
                result.append({"question_id": ques_id, "answer": answer})

        elif config['inference'] == 'rank':
            if args.mixed_precision_method == 'apex':
                with autocast():
                    answer_ids = model(video, question, answer_candidates, train=False, inference='rank',
                                       k_test=config['k_test'])
            else:
                answer_ids = model(video, question, answer_candidates, train=False, inference='rank',
                                   k_test=config['k_test'])

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id": int(ques_id.item()), "answer": answer_list[answer_id]})

    return result


def cal_acc(result_file, dataset):
    if dist.is_initialized():
        dist.barrier()
    with open(result_file, "r") as f:
        vqa_result = json.load(f)
    filtered_vqa_result = []
    qids = set()
    for res in vqa_result:
        if res["question_id"] not in qids:
            qids.add(res["question_id"])
            filtered_vqa_result.append(res)
    vqa_result = filtered_vqa_result
    assert len(vqa_result) == len(
        dataset.annotation), f"len(vqa_result): {len(vqa_result)}, len(dataset.annotation): {len(dataset.annotation)}"
    correct = 0
    for res in vqa_result:
        qid, ans = res["question_id"], res["answer"]
        if isinstance(dataset, ivqa):
            if dataset.annotation[qid]["answer_cnt"][ans] == 1:
                correct += 0.5
            elif dataset.annotation[qid]["answer_cnt"][ans] >= 2:
                correct += 1
        else:
            if ans == dataset.annotation[qid]["answer"]:
                correct += 1
    result = {'acc': f'{correct / len(vqa_result) * 100:.2f}%'}
    return result


def main(args, config):
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
    print("Creating vqa datasets")
    train_dataset, val_dataset, test_dataset = create_dataset(config['dataset'], config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train'],
                                                                      config['batch_size_test'],
                                                                      config['batch_size_test']],
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[vqa_collate_fn, None, None])
    #### Model #### 
    print("Creating model")
    model = testa_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                      vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                      token_merging=config['token_merging'], testa_r=config['testa_r'],
                      merging_type=config['merging_type'],
                      model_cfg=config)

    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.mixed_precision_method:
        args.effective_batch_size = config['batch_size_train'] * args.num_gpus
        args, model, optimizer = mixed_precision_init(args, model, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, scaler)

        else:
            break

        val_result = evaluation(model_without_ddp, val_loader, device, config)
        result_file = save_result(val_result, args.result_dir, 'val_result')
        val_result = cal_acc(result_file, val_loader.dataset)

        if utils.is_main_process():
            if float(val_result['acc'].strip('%')) > best:
                print('Saving current checkpoint')
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = float(val_result['acc'].strip('%'))
                best_epoch = epoch
            print('Val result: ', val_result)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_result.items()},
                         'epoch': epoch,
                         'best_epoch': best_epoch,
                         }
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()
        torch.cuda.empty_cache()

    if not args.evaluate:  # load best ckpt after fine-tuning
        model_without_ddp, _ = load_checkpoint(model_without_ddp, os.path.join(args.output_dir, 'checkpoint_best.pth'))
    vqa_result = evaluation(model_without_ddp, test_loader, device, config)
    result_file = save_result(vqa_result, args.result_dir, 'test_result')
    test_result = cal_acc(result_file, test_loader.dataset)
    print('Test result: ', test_result)
    if utils.is_main_process():
        file_name = "evaluate.txt" if args.evaluate else "log.txt"
        with open(os.path.join(args.output_dir, file_name), "a") as f:
            f.write(json.dumps(test_result) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml')
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--mixed_precision_method', type=str, default=None)
    parser.add_argument('--amp_opt_level', type=int, default=1)
    parser.add_argument('--deepspeed_fp16', action='store_true')
    parser.add_argument('--zero_opt_stage', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'), default_flow_style=False)

    main(args, config)
