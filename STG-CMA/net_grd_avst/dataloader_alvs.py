import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
import time
import random
from ipdb import set_trace
import pickle

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from einops import rearrange, repeat

import torchvision
import torchaudio
import glob
import cv2
import warnings

warnings.filterwarnings('ignore')

import sys

IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def ids_to_multinomial(id, categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]


class AVQA_dataset(Dataset):

    def __init__(self, root_path, label, audio_dir, video_res14x14_dir, transform=None, mode_flag='train'):

        self.root_path = root_path

        samples = json.load(
            open('./datasets/avqa/MUSIC_AVQA/balance_full_set/train_balance.json', 'r'))

        # nax =  nne
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(
            open('./datasets/avqa/MUSIC_AVQA/balance_full_set/train_balance.json', 'r'))
        self.max_len = 14  # question length

        self.audio_dir = audio_dir
        self.video_res14x14_dir = video_res14x14_dir
        self.transform = transform

        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list

        self.video_len = 10 * len(video_list)

        self.my_normalize = Compose([
            # Resize([384,384], interpolation=Image.BICUBIC),
            Resize([192, 192], interpolation=Image.BICUBIC),
            # Resize([224,224], interpolation=Image.BICUBIC),
            # CenterCrop(224),
            Normalize(-5.385333061218262, 3.5928637981414795),
        ])

        self.norm_mean = -5.385333061218262
        self.norm_std = 3.5928637981414795

    ### <----

    def __len__(self):
        return len(self.samples)

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
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        if waveform.shape[1] > 16000 * (1.95 + 0.1):
            sample_indx = np.linspace(0, waveform.shape[1] - 16000 * (1.95 + 0.1), num=10, dtype=int)
            waveform = waveform[:, sample_indx[idx]:sample_indx[idx] + int(16000 * 1.95)]

        ## align end ##

        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=10)

        # target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
        target_length = 192

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

    def sample_frames_from_video(self, video_path, fps=20):
        # Load the video
        video_capture = cv2.VideoCapture(video_path)

        video_fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Calculate how many frames to skip between each saved frame
        frame_skip = int(video_fps / fps)

        frame_number = 0
        saved_frame_number = 0

        frames = []

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            # Save the frame if it's the correct one (based on frame skip logic)
            if frame_number % frame_skip == 0:
                frames.append(self.my_normalize(frame))
                saved_frame_number += 1

            frame_number += 1

        # Release the video capture object
        video_capture.release()
        return frames

    def __getitem__(self, idx):

        sample = self.samples[idx]
        name = sample['video_id']


        ### ---> video frame process
        total_img = self.sample_frames_from_video(os.path.join("./datasets/avqa/MUSIC_AVQA/videos", name))

        total_img = torch.stack(total_img)
        ### <---


        # visual nega [60, 512, 14, 14]

        # question
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # answer
        answer = sample['anser']
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        wave = np.load(os.path.join('./datasets/avqa/MUSIC_AVQA/audio_wave/{}.npy'.format(name)))
        wave = torch.from_numpy(wave)
        wave = wave.view(10, 32000)
        while wave.size(-1) < 32000 * 10:
            wave = torch.cat((wave, wave), dim=-1)
        wave = wave[:, :32000 * 10]
        wave = wave.reshape([wave.shape[0], 10, 32000])

        sample = {'visual_posi': total_img, 'question': ques, 'label': label, 'wave': wave}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        visual_posi = sample['visual_posi']
        label = sample['label']
        wave = sample['wave']

        return {
            'visual_posi': sample['visual_posi'],
            'question': sample['question'],
            'label': label,
            'wave': wave}


def sample_frames_from_video(video_path, fps=20):
    my_normalize = Compose([
            # Resize([384,384], interpolation=Image.BICUBIC),
            Resize([192, 192], interpolation=Image.BICUBIC),
            # Resize([224,224], interpolation=Image.BICUBIC),
            # CenterCrop(224),
            Normalize(-5.385333061218262, 3.5928637981414795),
        ])
    # Load the video
    video_capture = cv2.VideoCapture(video_path)

    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate how many frames to skip between each saved frame
    frame_skip = int(video_fps / fps)

    frame_number = 0
    saved_frame_number = 0

    frames = []

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Save the frame if it's the correct one (based on frame skip logic)
        if frame_number % frame_skip == 0:
            frames.append(my_normalize(frame))
            saved_frame_number += 1

        frame_number += 1

    # Release the video capture object
    video_capture.release()
    return frames


def inference_dataloader(mode="visual"):
    samples = json.load(open('./datasets/avqa/MUSIC_AVQA/balance_full_set/train_balance.json', 'r'))
    ques_vocab = ['<pad>']
    ans_vocab = []
    i = 0
    N = 20
    for sample in samples:
        i += 1
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1

        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)
        if sample['anser'] not in ans_vocab:
            ans_vocab.append(sample['anser'])
    word_to_ix = {word: i for i, word in enumerate(ques_vocab)}

    samples = json.load(
            open('./datasets/avqa/MUSIC_AVQA/balance_full_set/test_balance.json', 'r'))
    max_len = 14  # question length
    transform = transforms.Compose([ToTensor()])
    video_list = []
    for sample in samples:
        video_name = sample['video_id']
        if video_name not in video_list:
            video_list.append(video_name)

    all_data = []
    for sample in samples:
        name = sample['video_id']

        ### ---> video frame process
        total_img = sample_frames_from_video(os.path.join("./datasets/avqa/MUSIC_AVQA/videos", name), N)
        total_img = torch.stack(total_img)
        if mode == "audio":
            with open(os.path.join("./datasets/avqa/MUSIC_AVQA/visual_idxs", f"{name}.pkl"), "rb") as f:
                s_inx = pickle.load(f)
            total_img = total_img[s_inx]

            ### <---

        # question
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]
        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < max_len:
            n = max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # answer
        answer = sample['anser']
        label = ids_to_multinomial(answer, ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        wave = np.load(os.path.join('./datasets/avqa/MUSIC_AVQA/audio_wave/{}.npy'.format(name)))
        wave = torch.from_numpy(wave)
        wave = wave.view(10, 32000)
        while wave.size(-1) < 32000 * 10:
            wave = torch.cat((wave, wave), dim=-1)
        wave = wave[:, :32000 * 10]
        wave = wave.reshape([wave.shape[0], 10, 32000])

        sample = {'visual_posi': total_img, 'question': ques, 'label': label, 'wave': wave}
        sample = transform(sample)
        sample['name'] = name
        all_data.append(sample)
    return all_data




