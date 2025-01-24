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
from torch.cuda.amp import GradScaler
import deepspeed
import zipfile
from pprint import pformat
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


def train(model, data_loader, optimizer, epoch, device, config, args, scaler):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_vtm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_vtc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Video Retrieval Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (video, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        B, N, C, H, W = video.size()
        video = video.permute(0, 2, 1, 3, 4)  # (B,C,N,H,W)
        video = video.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        if args.mixed_precision_method == 'deepspeed':
            inputs = {"video": video, 'caption': caption}
            inputs = fp32_to_fp16(inputs)
            video = inputs['video']
            caption = inputs['caption']
            loss_vtc, loss_vtm = model(video, caption, alpha, idx, bsz=B)
            loss = loss_vtc + loss_vtm
        elif args.mixed_precision_method == 'apex':
            with autocast():
                loss_vtc, loss_vtm = model(video, caption, alpha, idx, bsz=B)
                loss = loss_vtc + loss_vtm
        else:
            loss_vtc, loss_vtm = model(video, caption, alpha, idx, bsz=B)
            loss = loss_vtc + loss_vtm

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

        metric_logger.update(loss_vtm=loss_vtm.item())
        metric_logger.update(loss_vtc=loss_vtc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if utils.is_main_process():
            wandb.log({'train/loss_vtm': loss_vtm.item(), 'train/loss_vtc': loss_vtc.item(),
                       'train/lr': optimizer.param_groups[0]["lr"]})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config, args):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
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
        video_embeds.append(video_embed)

    video_feats = torch.cat(video_feats, dim=0)
    video_embeds = torch.cat(video_embeds, dim=0)

    sims_matrix = video_embeds @ text_embeds.t()
    score_matrix_v2t = torch.full((len(data_loader.dataset.video), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, 'V2T ' + header)):
        if args.k_test_batch_size >= config['k_test']:
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
            encoder_output = video_feats[start + i].repeat(config['k_test'], 1, 1).to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            if args.mixed_precision_method == 'apex':
                with autocast():
                    output = model.text_encoder(text_ids[topk_idx],
                                                attention_mask=text_atts[topk_idx],
                                                encoder_hidden_states=encoder_output,
                                                encoder_attention_mask=encoder_att,
                                                return_dict=True,
                                                )
            else:
                output = model.text_encoder(text_ids[topk_idx],
                                            attention_mask=text_atts[topk_idx],
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True,
                                            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_v2t[start + i, topk_idx] = score + topk_sim
        else:
            for j in range(0, config['k_test'], args.k_test_batch_size):
                topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
                topk_sim = topk_sim[j:j + args.k_test_batch_size]
                topk_idx = topk_idx[j:j + args.k_test_batch_size]
                encoder_output = video_feats[start + i].repeat(len(topk_idx), 1, 1).to(device)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
                if args.mixed_precision_method == 'apex':
                    with autocast():
                        output = model.text_encoder(text_ids[topk_idx],
                                                    attention_mask=text_atts[topk_idx],
                                                    encoder_hidden_states=encoder_output,
                                                    encoder_attention_mask=encoder_att,
                                                    return_dict=True,
                                                    )
                else:
                    output = model.text_encoder(text_ids[topk_idx],
                                                attention_mask=text_atts[topk_idx],
                                                encoder_hidden_states=encoder_output,
                                                encoder_attention_mask=encoder_att,
                                                return_dict=True,
                                                )
                score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_v2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2v = torch.full((len(texts), len(data_loader.dataset.video)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, 'T2I ' + header)):
        if args.k_test_batch_size >= config['k_test']:
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
            topk_idx = topk_idx.cpu()
            encoder_output = video_feats[topk_idx].to(device, non_blocking=True)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)
            if args.mixed_precision_method == 'apex':
                with autocast():
                    output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                                attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                                encoder_hidden_states=encoder_output,
                                                encoder_attention_mask=encoder_att,
                                                return_dict=True,
                                                )
            else:
                output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                            attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True,
                                            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2v[start + i, topk_idx] = score + topk_sim
        else:
            for j in range(0, config['k_test'], args.k_test_batch_size):
                topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
                topk_sim = topk_sim[j:j + args.k_test_batch_size]
                topk_idx = topk_idx[j:j + args.k_test_batch_size]
                topk_idx = topk_idx.cpu()
                encoder_output = video_feats[topk_idx].to(device, non_blocking=True)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)
                if args.mixed_precision_method == 'apex':
                    with autocast():
                        output = model.text_encoder(text_ids[start + i].repeat(len(topk_idx), 1),
                                                    attention_mask=text_atts[start + i].repeat(len(topk_idx), 1),
                                                    encoder_hidden_states=encoder_output,
                                                    encoder_attention_mask=encoder_att,
                                                    return_dict=True,
                                                    )
                else:
                    output = model.text_encoder(text_ids[start + i].repeat(len(topk_idx), 1),
                                                attention_mask=text_atts[start + i].repeat(len(topk_idx), 1),
                                                encoder_hidden_states=encoder_output,
                                                encoder_attention_mask=encoder_att,
                                                return_dict=True,
                                                )
                score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_t2v[start + i, topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_v2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2v, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_v2t.cpu().numpy(), score_matrix_t2v.cpu().numpy()


@torch.no_grad()
def vtm_eval(scores_v2t, scores_t2v, txt2vid, vid2txt):
    # Videos->Text
    ranks = np.zeros(scores_v2t.shape[0])
    for index, score in enumerate(scores_v2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in vid2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Videos
    ranks = np.zeros(scores_t2v.shape[0])

    for index, score in enumerate(scores_t2v):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2vid[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'vid_r1': ir1,
                   'vid_r5': ir5,
                   'vid_r10': ir10,
                   'vid_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


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

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, config, args, scaler)
            torch.cuda.empty_cache()

            score_val_v2t, score_val_t2v, = evaluation(model_without_ddp, val_loader, device, config, args)

            if utils.is_main_process():
                print('Validation')
                val_result = vtm_eval(score_val_v2t, score_val_t2v, val_loader.dataset.txt2video, val_loader.dataset.video2txt)
                print('Val result: ', val_result)

                if val_result['vid_r_mean'] > best:
                    print('Saving current checkpoint')
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = val_result['vid_r_mean']
                    best_epoch = epoch
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                wandb.log({f'val/{k}': v for k, v in val_result.items()})

        if args.evaluate:
            break

        dist.barrier()
        torch.cuda.empty_cache()

    if not args.evaluate:  # load best ckpt after fine-tuning
        model_without_ddp, _ = load_checkpoint(model_without_ddp, os.path.join(args.output_dir, 'checkpoint_best.pth'))
    score_test_v2t, score_test_t2v = evaluation(model_without_ddp, test_loader, device, config, args)

    if utils.is_main_process():
        if config['dataset'] == 'retrieval_condensedmovies':  # for Condensed Movies dataset submission
            np.save(os.path.join(args.output_dir, 'score_test_v2t.npy'), score_test_v2t)
            np.save(os.path.join(args.output_dir, 'score_test_t2v.npy'), score_test_t2v)
            from pathlib import Path
            sim_save_fp = Path(os.path.join(args.output_dir, 'score_test_t2v.npy'))
            zipfile.ZipFile(os.path.join(args.output_dir, 'submission.zip'), mode='w').write(sim_save_fp, 'sim_matrix_test.npy')
        else:
            print('Test evaluation')
            test_result = vtm_eval(score_test_v2t, score_test_t2v, test_loader.dataset.txt2video, test_loader.dataset.video2txt)
            print('Test result: ', test_result)
            log_stats = {**{f'test_{k}': v for k, v in test_result.items()},
                         'best_epoch': best_epoch,
                         }
            file_name = "evaluate.txt" if args.evaluate else "log.txt"
            with open(os.path.join(args.output_dir, file_name), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
            wandb.log({f'test/{k}': v for k, v in test_result.items()})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_queryd.yaml')
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
