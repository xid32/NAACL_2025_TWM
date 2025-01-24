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
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from models.blip import load_checkpoint
from models.testa_retrieval import testa_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from torch.cuda.amp import autocast
import pickle
from torch.cuda.amp import GradScaler
import deepspeed
import zipfile
from pprint import pformat
from alvs_inter import *
import wandb
from model_stats import params_count, gpu_mem_usage, get_model_stats


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




def sim(u: torch.Tensor, v: torch.Tensor, temperature: float):
    return F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=-1) / temperature


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: shape (B, D) where M is the batch size and D is the feature dimension
        # v: shape (B, D) where M is the batch size and D is the feature dimension

        # Compute cosine similarity between all pairs
        similarity_matrix = sim(u, v, self.temperature)

        # Create labels for cross entropy loss
        labels = torch.arange(u.shape[0], device=u.device)

        # Compute the loss
        loss1 = self.cross_entropy_loss(similarity_matrix, labels)

        # Compute cosine similarity between all pairs
        similarity_matrix = sim(v, u, self.temperature)

        # Create labels for cross entropy loss
        labels = torch.arange(v.shape[0], device=v.device)

        # Compute the loss
        loss2 = self.cross_entropy_loss(similarity_matrix, labels)

        return loss1 + loss2



def train(model, data_loader, optimizer, epoch, device, config, args, scaler, loss_func):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_vtm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_vtc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Video Retrieval Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (video, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        B, N, C, H, W = video.size()
        video = video.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video = video.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))


        inputs = {"video": video, 'caption': caption}
        inputs = fp32_to_fp16(inputs)
        video = inputs['video']
        caption = inputs['caption']
        video_feat, text_feat = model(video, caption, alpha, idx, bsz=B)
        loss = loss_func(video_feat, text_feat)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"./alvs.pt")

    return


@torch.no_grad()
def inference(model, data_loader, device, config, args):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=config['max_words'], return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    video_feats = []
    video_embeds = []
    for video, vid_id in data_loader:
        video = video.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video = video.to(device, non_blocking=True)
        video_feat = model.visual_encoder(video)  # [bsz*N, (image_size/patch_size)^2+1, 768]
        video_embed = model.vision_proj(torch.mean(video_feat, dim=1))
        video_embed = F.normalize(video_embed, dim=-1)
        if args.low_resource_eval:
            video_feat = video_feat.half()
        video_feats.append(video_feat.cpu())
        video_embeds.append([video_embed, vid_id])
    save_path = f"./datasets/caption/CMD/idxs"
    for i in range(len(video_embeds)):
        video_embed, vid_id = video_embeds[i]
        text_embed = text_embeds[i]
        s_inx = iterative_sampling(video_embed, text_embed, 5, 7, 0.6, 0.4)
        with open(os.path.join(save_path, f"{vid_id}.pkl"), "wb") as f:
            pickle.dump(s_inx, f)
    return




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
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config['dataset'], config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    #### Model ####
    print("Creating model")
    model = testa_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                            queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                            token_merging=config['token_merging'], testa_r=config['testa_r'],
                            merging_type=config['merging_type'],
                            model_cfg=config, max_words=config['max_words'])

    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.mixed_precision_method:
        args.effective_batch_size = config['batch_size_train'] * args.num_gpus
        args, model, optimizer = mixed_precision_init(args, model, optimizer)

    model_without_ddp = model
    if args.distributed:
        if args.mixed_precision_method != 'deepspeed':
            static_graph = True if config['vit_grad_ckpt'] is True else False
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], static_graph=static_graph)
            model_without_ddp = model.module

    if utils.is_main_process():
        wandb.init(
            # set the wandb project where this run will be logged
            project=config['dataset'],
            name=args.output_dir.split('/')[-1],
            # track hyperparameters and run metadata
            config=config
        )

        if config['vit_grad_ckpt'] is False:
            visual_encoder = model_without_ddp.visual_encoder
            model_stat = {'Params (M)': params_count(visual_encoder) / 1024 ** 2, 'Mem (G)': gpu_mem_usage(),
                          'Flops (G)': get_model_stats(visual_encoder, config, "flop", True),
                          'Activations (M)': get_model_stats(visual_encoder, config, "activation", True)}
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(model_stat) + "\n")

            wandb.log(model_stat)


    print("Start training")

    loss_info = InfoNCELoss(1)

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train(model, train_loader, optimizer, epoch, device, config, args, scaler, loss_info)
    model.load_state_dict(torch.load(f"./datasets/caption/MSRVTT/models/alvs.pt"))
    model.eval()
    inference(model, test_loader, device, config, args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_condensedmovies_f32.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_QuerYD_zeroshot')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
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
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--sep_image', action='store_true')
    parser.add_argument('--img_config', type=str)
    parser.add_argument('--k_test_batch_size', type=int, default=16)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--low_resource_eval', action='store_true',
                        help='reduce the memory cost during evaluation. use it when infer on didemo or anet without TESTA')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'), default_flow_style=False)

    main(args, config)
