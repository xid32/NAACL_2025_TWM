import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import ast
import json

from PIL import Image
from munch import munchify

import time
import random



def TransformImage(img):

    transform_list = []
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    transform_list.append(transforms.Resize([224,224]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    trans = transforms.Compose(transform_list)
    frame_tensor = trans(img)
    
    return frame_tensor


def load_frame_info(img_path, img_file):

    img_info = os.path.join(img_path, img_file)
    img = Image.open(img_info).convert('RGB')
    frame_tensor = TransformImage(img)

    return frame_tensor


def image_info(video_name, frame_flag):

    # path = "./data/frames-8fps"
    path = "./data/frames"
    img_path = os.path.join(path, video_name)

    img_list = os.listdir(img_path)
    img_list.sort()

    frame_idx = img_list[0 + frame_flag]
    img_tensor = load_frame_info(img_path, frame_idx)
    select_img = img_tensor.cpu().numpy()

    return select_img

def audio_info(audio_dir, audeo_name, aud_flag):

    audio = np.load(os.path.join(audio_dir, audeo_name + '.npy'))
    select_aud = audio[aud_flag]

    return select_aud

class AVQA_dataset(Dataset):

    def __init__(self, label_data, audio_dir, video_dir, transform=None):

        samples = json.load(open('./data/avqa-train_real.json', 'r'))

        self.samples = json.load(open(label_data, 'r'))

        video_list = []
        for sample in samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list
        self.audio_len = 10 * len(video_list)
        self.video_len = 10 * len(video_list)

        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.transform = transform

    
    def __len__(self):
        return self.video_len

    def __getitem__(self, idx):

        pos_frame_id = idx
        pos_video_id = int(idx / 10)
        pos_frame_flag = idx % 10
        pos_video_name = self.video_list[pos_video_id]
        # print("pos name: ", pos_video_name)

        while(1):
            neg_frame_id = random.randint(0, self.video_len - 1)
            if int(neg_frame_id/10) != int(pos_frame_id/10):
                break
        neg_video_id = int(neg_frame_id / 10)
        neg_frame_flag = neg_frame_id % 10
        neg_video_name = self.video_list[neg_video_id]

        aud_frame_id = pos_frame_id
        aud_id = pos_video_id
        aud_flag = pos_frame_flag

        # print(pos_video_id, neg_video_id, aud_id)
        pos_frame = torch.Tensor(image_info(pos_video_name, pos_frame_flag)).unsqueeze(0)
        neg_frame = torch.Tensor(image_info(neg_video_name, neg_frame_flag)).unsqueeze(0)

        sec_audio = torch.Tensor(audio_info(self.audio_dir, pos_video_name, aud_flag)).unsqueeze(0)

        video_s = torch.cat((pos_frame, neg_frame), dim=0)
        audio = torch.cat((sec_audio, sec_audio), dim=0)

        label  = torch.Tensor(np.array([1, 0]))

        video_id = pos_video_name
        sample = {'video_id':video_id, 'audio': audio, 'video_s': video_s, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
        