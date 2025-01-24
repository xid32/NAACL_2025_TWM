from __future__ import print_function
import sys 
sys.path.append("/home/guangyao_li/projects/music_avqa/") 

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from grounding_gen.dataloader_grd_gen_vis import *
from grounding_gen.nets_grd_gen_vis import AVQA_AVatt_Grounding
import ast
import json
import numpy as np
import torch.nn.functional as F

# from sklearn import cluster, metrics
# from skimage import img_as_ubyte
# from sklearn.preprocessing import normalize
import cv2


import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/grounding_gen_vis/'+TIMESTAMP)




print("\n ------------------- grounding_gen visualization ---------------------\n")


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    label = np.array(1)
    for batch_idx, sample in enumerate(train_loader):
        video_id, audio, video, target = sample['video_id'], sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['label'].to('cuda')
        optimizer.zero_grad()

        feat = model(video_id, audio, video)
        B = target.shape[0]
        C = target.shape[1]
        target = target.view(-1, B*C).squeeze()
        target = target.type(torch.LongTensor).cuda()
        # output.clamp_(min=1e-7, max=1 - 1e-7)
        
        loss = criterion(feat, target)
        writer.add_scalar('train/grd_gen_loss_vis',loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            video_id, audio, video_org, video, target = sample['video_id'], sample['audio'].to('cuda'), sample['pos_frame_org'].to('cuda'), sample['video_s'].to('cuda'), sample['label'].to('cuda')
            
            av_atten, preds = model(video_id, audio, video)

            print("\n\nvideo name: ", video_id)
            print("video_org type: ", type(video_org))

            # 正负样本交替的, 隔一个取一个mask
            obj_localization = av_atten.detach().cpu().numpy()  # (2, 1, 196)
            obj_localization = obj_localization[::2]            # (1, 1, 196)

            posi_img_data = video_org                            # [1, 3, 224, 224]

            obj_len = obj_localization.shape[0]
            print("obj_len: ", obj_len)
            for j in range(obj_len):
                print("obj: ", obj_localization.shape)
                map = obj_localization[j, :, :].squeeze()

                print("map: ", map.shape)
                map = (map-map.min()) / (map.max()-map.min())
                map=cv2.resize(map.reshape(14,14),(224,224))
                map=map/map.max()
                map=np.uint8(map*255)
                heatmap = cv2.applyColorMap(map, cv2.COLORMAP_JET)

                print("map type: ", type(map))

                current_img = posi_img_data[j].cpu().numpy()
                print("current_img type: ", type(current_img))
                current_img = cv2.resize(current_img, (224, 224))
                print("current_img: ", current_img.shape)

                result = heatmap * 0.4 + current_img * 0.6
 
                file_name = '%04d_' % batch_idx + '%04d_0' % j + '.jpg'
                print("file_name: ", file_name)
                cv2.imwrite(os.path.join('grounding_gen/models_grd_vis/vis_h4_c6', file_name), result)



def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('../data/json/avqa-test_real.json', 'r'))
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

            preds = model(video_id, audio, video)
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
        "--audio_dir", type=str, default='./data/feats/vggish', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='./data/frames', help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='./data/feats/r2plus1d_18', help="video dir")

    parser.add_argument(
        "--label_train", type=str, default="../dataset/avqa-train_real.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="../dataset/avqa-val_real.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="../dataset/avqa-test_real.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
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
        "--checkpoint", type=str, default='main_grounding_gen', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')


    args = parser.parse_args()   
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    if args.model == 'AVQA_AVatt_Grounding':
        model = AVQA_AVatt_Grounding()
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label_data=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        val_dataset = AVQA_dataset(label_data=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        ##### frozen
        for name, parameters in model.named_parameters():
            layer=str(name).split('.')[1]
            if(layer=='visual_net'):
                parameters.requires_grad= False

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        best_F = 0
        for epoch in range(1, args.epochs + 1):
            # train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader)
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + "_best.pt")
            torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + str(epoch) + ".pt")

    elif args.mode == 'val':
        test_dataset = AVQA_dataset(label_data=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_best.pt"))
        eval(model, test_loader)
    else:
        test_dataset = AVQA_dataset(label_data=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir)
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_best.pt"))
        test(model, test_loader)

if __name__ == '__main__':
    main()