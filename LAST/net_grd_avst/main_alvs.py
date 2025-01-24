from __future__ import print_function
import sys
# sys.path.append("/home/guangyao_li/projects/avqa/music_avqa_camera_ready")
import argparse
from base_options import BaseOptions
from gpuinfo import GPUInfo
import os

args = BaseOptions().parse()

mygpu = GPUInfo.get_info()[0]
gpu_source = {}

if 'N/A' in mygpu.keys():
    for info in mygpu['N/A']:
        if info in gpu_source.keys():
            gpu_source[info] += 1
        else:
            gpu_source[info] = 1

for gpu_id in args.gpu:
    gpu_id = str(gpu_id)

    if gpu_id not in gpu_source.keys():
        print('go gpu:', gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        break
    elif gpu_source[gpu_id] < 1:
        print('go gpu:', gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        break
import torch
import torch.nn as nn
import torch.optim as optim
from ipdb import set_trace
from dataloader_alvs import *
from torch.nn import functional as F
# from dataloader_avst_bk import *
from net_alvs import AVQA_Net

import ast
import json
import numpy as np
import pdb
# from .net_avst import AVQA_Fusion_Net

import warnings
from datetime import datetime
import wandb
from alvs_inter import *

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/net_avst/'+TIMESTAMP)

import certifi

os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())

print("\n--------------- ALVS --------------- \n")





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




def inference(model, mode):
    save_path = f"./datasets/avqa/MUSIC_AVQA/{mode}_idxs"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    samples = inference_dataloader()
    model.frozen_model()
    for sample in samples:
        visual_posi, target, question, wave = sample['visual_posi'].to('cuda'), sample['label'].to('cuda'), sample[
            'question'].to('cuda'), sample['wave'].to('cuda')
        f_v, f_a, f_qst = model(wave, visual_posi, question)
        f_qst = f_qst.squeeze()
        if mode == "visual":
            s_inx = iterative_sampling(f_v, f_qst[0, :], 11, 8, 0.2, 0.8)
            with open(os.path.join(save_path, f"{sample['name']}.pkl"), "wb") as f:
                pickle.dump(s_inx, f)
        else:
            with open(os.path.join(save_path, f"{sample['name']}.pkl"), "rb") as f:
                s_inx = pickle.load(f)
            visual_posi = visual_posi[s_inx]
            f_v, f_a, f_qst = model(wave, visual_posi, question)
            f_qst = f_qst.squeeze()
            # Calculate att_f_a
            att_f_a = torch.stack([calculate_attention(f_qst[0, :], f_a[i].unsqueeze(0)) for i in range(f_a.shape[0])])

            # Compute cosine similarity between consecutive att_f_a vectors
            sim = cosine_similarity(att_f_a[:-1], att_f_a[1:])
            sim = torch.cat([sim, sim[-1].unsqueeze(0)])  # Ensure last and second-last are the same

            max_sim_index = [torch.argmax(sim)]
            with open(os.path.join(save_path, f"{sample['name']}.pkl"), "wb") as f:
                pickle.dump(max_sim_index, f)
    return




def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    # print("audio data: ", audio_data.shape)
    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0

    return out_match, batch_labels


def train(args, model, train_loader, optimizer, criterion, epoch, mode):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):
        visual_posi, target, question, wave = sample['visual_posi'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['wave'].to('cuda')

        optimizer.zero_grad()
        f_v, f_a, f_qst = model(wave, visual_posi, question)

        if mode == "visual":
            loss_v = criterion(f_v, f_qst)
            loss_q = criterion(f_qst, f_v)
            loss = loss_q + loss_v
        else:
            loss_v = criterion(f_v, f_a)
            loss_a = criterion(f_a, f_v)
            loss = loss_a + loss_v

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(visual_posi), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main():
    mode = "audio"
    # Training settings
    if args.wandb:
        wandb.init(config=args, project="AVQA", name=args.model_name)

    torch.manual_seed(args.seed)


    model = AVQA_Net(args)
    model = model.to('cuda')

    train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir,
                                 video_res14x14_dir=args.video_res14x14_dir,
                                 transform=transforms.Compose([ToTensor()]), mode_flag='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)


    param_group = []
    train_params = 0
    total_params = 0
    additional_params = 0
    for name, param in model.named_parameters():

        param.requires_grad = True
        ### ---> compute params
        tmp = 1
        for num in param.shape:
            tmp *= num

        if 'ViT' in name or 'swin' in name or 'Resnet' in name:
            if 'norm' in name:
                param.requires_grad = bool(args.is_vit_ln)
                total_params += tmp
                train_params += tmp
            else:
                param.requires_grad = False
                total_params += tmp

        # ### <----

        elif 'adapter_blocks' in name:
            param.requires_grad = True
            train_params += tmp
            additional_params += tmp
            total_params += tmp
            print('########### train layer:', name)
        else:
            param.requires_grad = True
            train_params += tmp
            total_params += tmp

        if 'adapter_blocks' in name:
            param_group.append({"params": param, "lr": args.lr_block})
        else:
            param_group.append({"params": param, "lr": args.lr})
    print('####### Trainable params: %0.4f  #######' % (train_params * 100 / total_params))
    print(
        '####### Additional params: %0.4f  ######' % (additional_params * 100 / (total_params - additional_params)))
    print('####### Total params in M: %0.1f M  #######' % (total_params / 1000000))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = InfoNCELoss()
    best_F = 0
    count = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, criterion, epoch=epoch, mode=mode)
        scheduler.step(epoch)
        count += 1
        torch.save(model.state_dict(), f"./{mode}_alvs.pt")
        if count == args.early_stop:
            exit()
    inference(model, mode)


if __name__ == '__main__':
    main()