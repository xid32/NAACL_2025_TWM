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

from ipdb import set_trace

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision
import torchaudio
import glob
import warnings
warnings.filterwarnings('ignore')

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

        samples = json.load(open('/root/autodl-tmp/data/AVQA/json/avqa-train_real.json', 'r'))

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
        
        self.my_normalize = Compose([
            # Resize([384,384], interpolation=Image.BICUBIC),
            Resize([192,192], interpolation=Image.BICUBIC),
            # Resize([224,224], interpolation=Image.BICUBIC),
            # CenterCrop(224),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        ### ---> yb calculate stats for AVQA
        self.norm_mean =  -5.385333061218262
        self.norm_std =  3.5928637981414795
        
    
    def __len__(self):
        return self.video_len
    
    def _wav2fbank(self, filename, filename2=None, idx=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
    

        ## yb: align ##
        if waveform.shape[1] > 16000*(1.95+0.1):
            sample_indx = np.linspace(0, waveform.shape[1] -16000*(1.95+0.1), num=10, dtype=int)
            waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*1.95)]

        ## align end ##



        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=10)



        # target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
        target_length = 192 ## yb: overwrite for swin

        
        ########### ------> very important: audio normalized
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ### <--------

        # target_length = 512 ## 5s
        # target_length = 256 ## 2.5s
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda
    
    def __getitem__(self, idx):

        pos_frame_id = idx
        pos_video_id = int(idx / 10)
        pos_frame_flag = idx % 10
        pos_video_name = self.video_list[pos_video_id]
        # print("pos name: ", pos_video_name)
        
        
        ### ---> video frame process 
        total_num_frames = len(glob.glob(os.path.join(self.video_dir, pos_video_name, '*.jpg')))
        sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
        total_img = []
        for tmp_idx in sample_indx:
            tmp_img = torchvision.io.read_image(os.path.join(self.video_dir, pos_video_name, str("{:08d}".format(tmp_idx))+ '.jpg'))/255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)
        pos_frame = total_img[pos_frame_flag].unsqueeze(0)
        ### <---
        
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
        
        
        ### ---> video frame process 
        total_num_frames = len(glob.glob(os.path.join(self.video_dir, neg_video_name, '*.jpg')))
        sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
        total_img = []
        for tmp_idx in sample_indx:
            tmp_img = torchvision.io.read_image(os.path.join(self.video_dir, neg_video_name, str("{:08d}".format(tmp_idx))+ '.jpg'))/255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)
        neg_frame = total_img[neg_frame_flag].unsqueeze(0)
        ### <---
        
        pos_wave = np.load('/root/autodl-tmp/data/AVQA/audio_wave/{}.npy'.format(pos_video_name))
        pos_wave = torch.from_numpy(pos_wave)
        pos_wave = pos_wave.view(10, 32000)
        while pos_wave.size(-1) < 32000 * 10:
            pos_wave = torch.cat((pos_wave, pos_wave), dim=-1)     
        pos_wave = pos_wave[:, :32000*10]
        
        sec_audio = pos_wave[aud_flag].unsqueeze(0)

        video_s = torch.cat((pos_frame, neg_frame), dim=0)
        audio = torch.cat((sec_audio, sec_audio), dim=0)

        label  = torch.Tensor(np.array([1, 0]))

        video_id = pos_video_name
        sample = {'video_id':video_id, 'audio': audio, 'video_s': video_s, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
        