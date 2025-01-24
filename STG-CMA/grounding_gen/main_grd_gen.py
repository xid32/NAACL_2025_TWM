from __future__ import print_function
import sys 
sys.path.append("/home/guangyao_li/projects/music_avqa/")   # path to your project root

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_grd_gen import *
from nets_grd_gen import AVQA_AVatt_Grounding
import ast
import json
import numpy as np
import torch.nn.functional as F

from utils import do_mixup, get_mix_lambda, do_mixup_label

import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/grounding_gen/'+TIMESTAMP)


print("\n ------------------- grounding_gen ---------------------\n")


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    label = np.array(1)
    for batch_idx, sample in enumerate(train_loader):
        video_id, audio, video, target = sample['video_id'], sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['label'].to('cuda')
        optimizer.zero_grad()

        if args.backbone_type == "audioset":
            mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(audio))).to('cuda')
        else:
            mixup_lambda = None
   
        feat = model(video_id, audio, video, mixup_lambda=mixup_lambda)
        B = target.shape[0]
        C = target.shape[1]
        target = target.view(-1, B*C).squeeze()
        target = target.type(torch.LongTensor).cuda()
        # output.clamp_(min=1e-7, max=1 - 1e-7)
        
        loss = criterion(feat, target)
        writer.add_scalar('train/grd_gen_loss',loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def eval(args, model, val_loader, epoch):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            video_id, audio, video, target = sample['video_id'], sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['label'].to('cuda')
            
            if args.backbone_type == "audioset":
                mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(audio))).to('cuda')
            else:
                mixup_lambda = None
            
            preds = model(video_id, audio, video, mixup_lambda=mixup_lambda)
            
            _, predicted = torch.max(preds, 1)
            total += preds.size(0)
            correct += (predicted == target.view(target.size(0) * target.size(1))).sum().item()
            
    print('Accuracy: %.2f %%' % (100 * correct / total))
    writer.add_scalar('eval/grd_gen_acc', float((100 * correct / total)), epoch)

    return 100 * correct / total


def test(args, model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('./data/json/avqa-test_real.json', 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            video_id, audio, video, target = sample['video_id'], sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['label'].to('cuda')

            if args.backbone_type == "audioset":
                mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(audio))).to('cuda')
            else:
                mixup_lambda = None
                
            preds = model(video_id, audio, video, mixup_lambda=mixup_lambda)
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                    # AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())

    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (100 * correct / total))
    return 100 * correct / total

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default='/root/autodl-tmp/data/AVQA/audio', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='/root/autodl-tmp/data/AVQA/frames', help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='./data/feats/r2plus1d_18', help="video dir")

    parser.add_argument(
        "--label_train", type=str, default="/root/autodl-tmp/data/AVQA/json/avqa-train_real.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="/root/autodl-tmp/data/AVQA/json/avqa-val_real.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="/root/autodl-tmp/data/AVQA/json/avqa-test_real.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_AVatt_Grounding', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='grounding_gen/models_grounding_gen/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='lavish_grounding_gen', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')
    parser.add_argument(
        '--wandb', type=int, default=0, help='weight and bias setup')
    parser.add_argument('--audio_length', type=float, default= 1, help='audio length')
    parser.add_argument('--num_workers', type=int, default= 16, help='worker for dataloader')
    parser.add_argument('--model_name', type=str, default=None, help="for log")

    parser.add_argument('--adapter_kind', type=str, default='bottleneck', help="for log")

    parser.add_argument('--Adapter_downsample', type=int, default=16, help="tune top k")

    parser.add_argument('--num_conv_group', type=int, default=4, help="group conv")

    parser.add_argument('--is_audio_adapter_p1', type=int, default=0, help="TF audio adapter")
    parser.add_argument('--is_audio_adapter_p2', type=int, default=0, help="TF audio adapter")
    parser.add_argument('--is_audio_adapter_p3', type=int, default=0, help="TF audio adapter")

    parser.add_argument('--is_bn', type=int, default=0, help="TF audio adapter")
    parser.add_argument('--is_gate', type=int, default=0, help="TF audio adapter")
    parser.add_argument('--is_multimodal', type=int, default=1, help="TF audio adapter")
    parser.add_argument('--is_before_layernorm', type=int, default=1, help="TF audio adapter")
    parser.add_argument('--is_post_layernorm', type=int, default=1, help="TF audio adapter")

    parser.add_argument('--is_vit_ln', type=int, default=0, help="TF Vit")

    parser.add_argument('--is_fusion_before', type=int, default=0, help="TF Vit")

    parser.add_argument('--num_tokens', type=int, default=32, help="num of MBT tokens")

    parser.add_argument(
        '--early_stop', type=int, default=3, help='weight and bias setup')

    parser.add_argument(
        '--lr_block', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')

    parser.add_argument(
        '--tmp_av', type=float, default=0.1, help='tmp for nce')
    parser.add_argument(
        '--tmp_tv', type=float, default=0.1, help='tmp for nce')

    parser.add_argument(
        '--coff_av', type=float, default=0.5, help='tmp for nce')
    parser.add_argument(
        '--coff_tv', type=float, default=0.5, help='tmp for nce')
    parser.add_argument(
        '--backbone_type', type=str, default='audioset', help='the backbone of htsat')    

    args = parser.parse_args()   
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    if args.model == 'AVQA_AVatt_Grounding':
        model = AVQA_AVatt_Grounding(args)
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label_data=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        val_dataset = AVQA_dataset(label_data=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ##### frozen
        for name, parameters in model.named_parameters():
            layer=str(name).split('.')[1]
            if(layer=='swin'):
                parameters.requires_grad= False
            if(layer=='htsat'):
                parameters.requires_grad= False

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        best_F = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(args, model, val_loader, epoch=epoch)
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + "_best.pt")
            torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + str(epoch) + ".pt")

    elif args.mode == 'val':
        test_dataset = AVQA_dataset(label_data=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_best.pt"))
        eval(args, model, test_loader, epoch=epoch)
    else:
        test_dataset = AVQA_dataset(label_data=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir)
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_best.pt"))
        test(args, model, test_loader)

if __name__ == '__main__':
    main()