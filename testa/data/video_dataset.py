import logging

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import copy
from PIL import Image
import math
import pickle
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
import random
from data.utils import pre_caption, pre_question
from .randaugment import TemporalConsistentRandomAugment
import pandas as pd
import collections
decord.bridge.set_bridge("torch")


class VideoRandomSquareCrop(object):
    def __init__(self, crop_size, p=0.5):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        self.p = p

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be cropped.

        Returns:
            torch.tensor: cropped video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                b, t, h, w = video.shape
            else:
                raise RuntimeError('Expecting 4-dimensional tensor of shape (b,t,h,w), got {}'.format(video.shape))

            # if random.uniform(0, 1) < self.p:
            #     video = torch.flip(video, (3,))

            x = random.randint(0, h - self.crop_size)
            y = random.randint(0, w - self.crop_size)

            return video[:, :, x: x + self.crop_size, y: y + self.crop_size]

        else:
            raise NotImplementedError('Support only torch.Tensor as input, got {}'.format(type(video)))


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, img):
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def load_video_from_path_decord(video_path, frm_sampling_strategy, num_frm, height=None, width=None, start_time=None,
                                end_time=None, fps=-1):
    try:
        if not height or not width:
            vr = VideoReader(video_path)
        else:
            vr = VideoReader(video_path, width=width, height=height)

        vlen = len(vr)

        if start_time or end_time:
            assert fps > 0, 'must provide video fps if specifying start and end time.'

            start_idx = min(int(start_time * fps), vlen)
            end_idx = min(int(end_time * fps), vlen)
        else:
            start_idx, end_idx = 0, vlen

        if frm_sampling_strategy == 'uniform':
            frame_indices = np.arange(start_idx, end_idx, vlen / num_frm, dtype=int)
        elif frm_sampling_strategy == 'nlvl_uniform':
            frame_indices = np.arange(start_idx, end_idx, vlen / num_frm).astype(int)
        elif frm_sampling_strategy == 'nlvl_rand':
            frame_indices = np.arange(start_idx, end_idx, vlen / num_frm).astype(int)

            # generate some random perturbations
            strides = [frame_indices[i] - frame_indices[i - 1] for i in range(1, len(frame_indices))] + [vlen - frame_indices[-1]]
            pertube = np.array([np.random.randint(0, stride) for stride in strides])

            frame_indices = frame_indices + pertube
        elif frm_sampling_strategy == 'rand':
            frame_indices = sorted(random.sample(range(vlen), num_frm))
        elif frm_sampling_strategy == 'headtail':
            frame_indices_head = sorted(random.sample(range(vlen // 2), num_frm // 2))
            frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), num_frm // 2))
            frame_indices = frame_indices_head + frame_indices_tail
        else:
            raise NotImplementedError('Invalid sampling strategy {} '.format(frm_sampling_strategy))

        raw_sample_frms = vr.get_batch(frame_indices)
    except Exception as e:
        return None

    raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)  # (N, H, W, C) to (N, C, H, W)

    return raw_sample_frms


class VideoDataset(Dataset):

    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/msrvtt_test.jsonl'
        filename = 'msrvtt_test.jsonl'

        download_url(url, ann_root)
        self.annotation = load_jsonl(os.path.join(ann_root, filename))

        print('number of instances: %s' % len(self.annotation))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        # self.text = [pre_caption(ann['caption'], 40) for ann in self.annotation]
        self.txt2video = [i for i in range(len(self.annotation))]
        self.video2txt = self.txt2video

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt)

        if not os.path.exists(video_path):
            print('not exist %s' % video_path)
            return

        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)

        video = self.img_norm(vid_frm_array.float())

        return video, ann['clip_name']



class caption_video(Dataset):

    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=224,
                 split='test', max_words=30, prompt='', video_resize=256, input_segments=False, input_asr=False,
                 asr_drop=0.0, seg_drop=0.0):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''

        filename = '%s.caption_coco_format.json' % split
        with open(os.path.join(ann_root, filename), 'r') as f:
            self.annotation = json.load(f)['annotations']

        if split == 'train':
            print('number of instances: %s in %s dataset' % (len(self.annotation), split))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_resize = video_resize
        self.video_random_cropper = VideoRandomSquareCrop(max_img_size)
        self.video_rand_aug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])  
        self.video_root = video_root
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        self.max_words = max_words
        self.prompt = prompt
        self.input_segments = input_segments
        self.input_asr = input_asr
        self.asr_drop = asr_drop
        self.seg_drop = seg_drop

        # remove invalid files
        remove_cnt = 0
        valid_annotation = []
        for ann in self.annotation:
            video_path = os.path.join(self.video_root, ann['image_id'])
            if not os.path.exists(video_path):
                remove_cnt += 1
                continue
            valid_annotation.append(ann)
        self.annotation = valid_annotation
        print('remove {} invalid video-text pairs'.format(remove_cnt))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        video_path = os.path.join(self.video_root, ann['image_id'])

        if not os.path.exists(video_path):
            print('not exist %s' % video_path)
            return

        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.video_resize, width=self.video_resize)
        vid_frm_array = self.video_random_cropper(vid_frm_array) # (N, C, H, W)
        vid_frm_array = self.video_rand_aug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # (N, H, W, C) to (N, C ,H. W)
        video = self.img_norm(vid_frm_array.float())

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        seg_prompt = ''
        if 'seg_prompt' in ann and self.input_segments == True:
            seg_prompt = ann['seg_prompt']
            #caption = seg_prompt + caption
        asr_text = ''
        if 'asr_text' in ann and self.input_asr == True:
            asr_text = ann['asr_text']
        seg_text = ''
        if 'seg_text' in ann and self.input_segments == True:
            seg_text  = ann['seg_text']
        
        if self.asr_drop > 0:
            if random.random() < self.asr_drop:
                asr_text = ''
        if self.seg_drop > 0:
            if random.random() < self.seg_drop:
                seg_text = ''


        return video, caption, ann['image_id'], seg_prompt, asr_text, seg_text # (N, C, H, W)

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'nlvl_uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm).astype(int)[:self.num_frm]
                if frame_indices[-1] >= vlen:
                    frame_indices[-1] = vlen-1
            elif self.frm_sampling_strategy == 'nlvl_rand':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm).astype(int)

                # generate some random perturbations
                strides = [frame_indices[i] - frame_indices[i-1] for i in range(1, len(frame_indices))] + [vlen - frame_indices[-1]]
                pertube = np.array([np.random.randint(0, stride) for stride in strides])

                frame_indices = frame_indices + pertube
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            print(video_path, e)
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # (N, H, W, C) to (N, C, H, W)

        return raw_sample_frms


class caption_video_eval(caption_video):

    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384, split='test', 
            input_segments=False, input_asr=False, asr_drop=0.0, seg_drop=0.0):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        super(caption_video_eval, self).__init__(video_root, ann_root, num_frm, frm_sampling_strategy, max_img_size,
                                                 split=split, input_segments=input_segments, input_asr=input_asr,
                                                 asr_drop=asr_drop, seg_drop=seg_drop)

        # delete duplicate items for evaluation

        annotation_removed_duplication = []
        image_ids = set()
        for ann in self.annotation:
            if ann['image_id'] in image_ids:
                continue
            else:
                image_ids.add(ann['image_id'])
                annotation_removed_duplication.append(ann)
        self.annotation = annotation_removed_duplication
        self.input_segments = input_segments
        print('number of instances: %s in %s dataset' % (len(self.annotation), split))

    def __getitem__(self, index):

        ann = self.annotation[index]

        video_path = os.path.join(self.video_root, ann['image_id'])
        seg_prompt= ''
        if 'seg_prompt' in ann and self.input_segments == True:
            seg_prompt = ann['seg_prompt']
        asr_text = ''
        if 'asr_text' in ann and self.input_asr == True:
            asr_text = ann['asr_text']
        seg_text = ''
        if 'seg_text' in ann and self.input_segments == True:
            seg_text  = ann['seg_text']

        if self.asr_drop > 0:
            if random.random() < self.asr_drop:
                asr_text = ''
        if self.seg_drop > 0:
            if random.random() < self.seg_drop:
                seg_text = ''

        if not os.path.exists(video_path):
            print('not exist %s' % video_path)
            return

        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

        video = self.img_norm(vid_frm_array.float())
        #print('asr_text' in ann, self.input_asr )
        #print('asr_text', asr_text)
        return video, ann['image_id'], seg_prompt, asr_text, seg_text


class CoinDataset(Dataset):
    split_map = {
        "train": "training",
        "valid": "testing",
        "test": "testing",
    }

    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 split='test', video_fmt='.mp4', sliding_window=16, sliding_window_size=32, max_video_len=32, clip_len=128):
        """
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        """
        filename = 'COIN_clean_with_frame_num.json'
        with open(os.path.join(ann_root, filename), 'r') as f:
            database = json.load(f)['database']

        id2label = {}
        data = []
        # filter the data by split.
        for video_id, rec in database.items():
            # if not os.path.isfile(os.path.join(config.vfeat_dir, video_id + ".npy")):
            #     continue
            if rec["subset"] == CoinDataset.split_map[split]:
                recipe_type = rec["recipe_type"]
                starts, ends, labels = [], [], []
                for segment in rec["annotation"]:
                    start, end = segment["segment"]
                    label = int(segment["id"])
                    starts.append(start)
                    ends.append(end)
                    labels.append(label)
                for clip_start in range(0, int(rec["frame_num"]), clip_len):
                    data.append({"video_id": video_id, "recipe_type": recipe_type, "start": starts, "end": ends,
                                 "label": labels, "clip_start": clip_start})

            # always use testing to determine label_set
            if rec["subset"] == "testing":
                for segment in rec["annotation"]:
                    id2label[int(segment["id"])] = segment["label"]

        # text_labels is used for ZS setting
        self.text_labels = ["none"] * len(id2label)
        for label_id in id2label:
            self.text_labels[label_id - 1] = id2label[label_id]

        id2label[0] = "O"
        print("num of labels", len(id2label))

        self.annotation = data[::3]  # TODO only use 1/3 data

        print('number of instances: %s' % len(self.annotation))
        self.clip_len = clip_len
        self.sliding_window = sliding_window
        self.sliding_window_size = sliding_window_size
        self.max_video_len = max_video_len
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        video_path = os.path.join(self.video_root, str(ann["recipe_type"]), ann['video_id'] + self.video_fmt)

        if not os.path.exists(video_path):
            print('not exist %s' % video_path)
            return

        clip_start = ann["clip_start"]
        vid_frm_array, vid_ts_array = self._load_video_from_path_decord(video_path, height=self.max_img_size,
                                                                        width=self.max_img_size,
                                                                        start_idx=clip_start,
                                                                        end_idx=clip_start + self.clip_len,
                                                                        fps=30)
        video = self.img_norm(vid_frm_array.float())

        N, C, W, H = video.size()
        starts, ends, label_ids = ann["start"], ann["end"], ann["label"]
        # sliding window.
        video_len = N

        video_targets = torch.full((video_len,), 0)
        for frame_id in range(video_len):
            timestamp = vid_ts_array[frame_id][1]
            for start, end, label_id in zip(starts, ends, label_ids):
                if timestamp >= start and timestamp <= end:
                    video_targets[frame_id] = label_id
                    continue

        vfeats, vmasks, targets = [], [], []
        # sliding window on video features and targets.
        for window_start in range(0, N, self.sliding_window):
            video_start = 0
            video_end = min(video_len - window_start, self.sliding_window_size)
            video_clip = {"start": [video_start], "end": [video_end]}
            vfeat, vmask = self._build_video_seq(
                video[window_start: window_start + video_end],
                video_clip
            )
            # covers video length only.
            target = torch.full_like(vmask, -100, dtype=torch.long)
            target[vmask] = 0
            target[:video_end] = video_targets[window_start: window_start + video_end]
            vfeats.append(vfeat)
            vmasks.append(vmask)
            targets.append(target)
            if (video_len - window_start) <= self.sliding_window_size:
                break

        vfeats = torch.stack(vfeats)
        vmasks = torch.stack(vmasks)
        targets = torch.stack(targets)

        del video, vid_frm_array, vmasks

        return vfeats, targets, video_targets, video_len

    def _build_video_seq(self, video_feature, video_clips=None):
        """
        `video_feature`: available video tokens.
        `video_clips`: video clip sequence to build.
        """
        if video_clips is None:
            # this is borrowed from DSAligner
            video_start = 0
            video_end = min(len(video_feature), self.max_video_len)
            # the whole sequence is a single clip.
            video_clips = {"start": [video_start], "end": [video_end]}

        vfeats = torch.zeros(
            (self.max_video_len, video_feature.size(1), video_feature.size(2), video_feature.size(3)), dtype=torch.float32
        )
        vmasks = torch.zeros((self.max_video_len,), dtype=torch.bool)
        video_len = 0
        for start, end in zip(video_clips["start"], video_clips["end"]):
            clip_len = min(self.max_video_len - video_len, (end - start))
            if clip_len > 0:
                vfeats[video_len: video_len + clip_len] = video_feature[
                    start: start + clip_len
                ]
                vmasks[video_len: video_len + clip_len] = 1
                video_len += clip_len

        return vfeats, vmasks

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_idx=None, end_idx=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)
            if start_idx or end_idx:
                start_idx = min(start_idx, vlen)
                end_idx = min(end_idx, vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            elif self.frm_sampling_strategy == 'all':
                frame_indices = np.arange(start_idx, end_idx, dtype=int)
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
            raw_sample_timestamp = vr.get_frame_timestamp(frame_indices)
        except Exception as e:
            logging.error(f"video_path: {video_path}, vlen: {vlen}, start_idx: {start_idx}, end_idx: {end_idx}")
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)
        return raw_sample_frms, raw_sample_timestamp


class QuerYD_train(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='train'):
        """
        video_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        filenames = {'train': 'exists_train_list.txt', 'val': 'exists_val_list.txt', 'test': 'exists_test_list.txt'}

        with open(os.path.join(ann_root, filenames[split]), 'r') as f:
            self.video_names = f.readlines()
            self.video_names = [a.strip() for a in self.video_names]

        with open(os.path.join(ann_root, 'raw_captions_combined_filtered.pkl'), 'rb') as f:
            annotations = pickle.load(f)

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        self.text = []
        self.video = []
        self.txt2video = {}
        self.video2txt = {}
        self.annotation = []

        txt_id = 0
        for video_id, video_name in enumerate(self.video_names):
            self.video.append(f"{video_name}")
            texts = annotations[video_name]
            text = ' '.join([' '.join(tokens) for tokens in texts])
            text = pre_caption(text, max_words)
            self.text.append(text)
            self.video2txt[video_id] = [txt_id]
            self.txt2video[txt_id] = video_id
            self.annotation.append({'video_id': video_id, 'caption': text, 'video_name': video_name})
            txt_id += 1
        assert len(self.txt2video) == len(self.video2txt)
        print('number of instances: %s' % len(self.annotation))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        caption = ann['caption']
        video_id = ann['video_id']
        video_path = os.path.join(self.video_root, ann['video_name'])
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())

        return video, caption, video_id


class QuerYD_retrieval_eval(QuerYD_train):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='test'):
        super(QuerYD_retrieval_eval, self).__init__(video_root, ann_root, num_frm, frm_sampling_strategy, max_img_size,
                                                    video_fmt, max_words, split)

    def __getitem__(self, index):
        video_path = os.path.join(self.video_root, self.video[index])
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())

        return video, index


class MSRVTT_retrieval_train(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        filename = f'{split}.jsonl'
        self.annotation = load_jsonl(os.path.join(ann_root, filename))

        print('number of instances: %s' % len(self.annotation))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        # remove invalid files
        remove_cnt = 0
        valid_annotation = []
        video_ids = set()
        self.text = []
        self.video = []
        self.txt2video = {}
        self.video2txt = {}
        txt_id, video_id = 0, -1
        for ann in self.annotation:
            video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt)
            if not os.path.exists(video_path):
                remove_cnt += 1
                continue
            if ann['clip_name'] not in video_ids:  # new video
                video_ids.add(ann['clip_name'])
                video_id += 1
                self.video.append(ann['clip_name'])
                self.video2txt[video_id] = []
            valid_annotation.append(ann)
            self.text.append(ann['caption'])
            self.video2txt[video_id].append(txt_id)
            self.txt2video[txt_id] = video_id
            txt_id += 1
        self.annotation = valid_annotation
        print('remove {} invalid video-text pairs'.format(remove_cnt))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt)
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = int(ann['clip_name'].strip('video'))

        return video, ann['caption'], video_id


class MSRVTT_retrieval_eval(MSRVTT_retrieval_train):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='test'):
        super(MSRVTT_retrieval_eval, self).__init__(video_root, ann_root, num_frm, frm_sampling_strategy, max_img_size,
                                                    video_fmt, max_words, split)
        annotation_removed_duplication = []
        video_ids = set()
        for ann in self.annotation:
            if ann['clip_name'] in video_ids:
                continue
            else:
                video_ids.add(ann['clip_name'])
                annotation_removed_duplication.append(ann)
        self.annotation = annotation_removed_duplication
        print('number of instances: %s in %s dataset' % (len(self.annotation), split))

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt)
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = int(ann['clip_name'].strip('video'))

        return video, video_id


class MSRVTT_qa(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        filename = f'{split}.jsonl'
        self.split = split
        self.annotation = load_jsonl(os.path.join(ann_root, filename))

        print('number of instances: %s' % len(self.annotation))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.question_ids = [i for i in range(len(self.annotation))]
        self.answer_list = list(set([ann["answer"] for ann in self.annotation]))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.video_root, ann['video_id'] + self.video_fmt)
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = int(ann['video_id'].strip('video'))

        if self.split == 'val' or self.split == 'test':
            question = pre_question(ann['question'])
            question_id = self.question_ids[index]  # TODO
            return video, question, question_id
        elif self.split == 'train':
            question = pre_question(ann['question'])
            answers = [ann['answer']]
            weights = [0.2]

            return video, question, answers, weights


class DiDeMo_retrieval_train(MSRVTT_retrieval_train):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        super(DiDeMo_retrieval_train, self).__init__(video_root, ann_root, num_frm, frm_sampling_strategy, max_img_size,
                                                    video_fmt, max_words, split)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt)
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = range(len(self.annotation))[index]

        return video, ann['caption'], video_id


class DiDeMo_retrieval_eval(MSRVTT_retrieval_eval):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='test'):
        super(DiDeMo_retrieval_eval, self).__init__(video_root, ann_root, num_frm, frm_sampling_strategy, max_img_size,
                                                    video_fmt, max_words, split)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt)
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = range(len(self.annotation))[index]

        return video, video_id


class ActivityNet_qa(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        self.split = split
        with open(os.path.join(ann_root, f'{split}_q.json'), 'r') as f:
            self.questions = json.load(f)
        with open(os.path.join(ann_root, f'{split}_a.json'), 'r') as f:
            self.answers = json.load(f)
        self.qid2ans = {a["question_id"]: a["answer"] for a in self.answers}
        self.annotation = []
        for item in self.questions:
            answer = self.qid2ans[item["question_id"]]
            item["answer"] = answer
            self.annotation.append(item)

        if split == 'val':
            random.shuffle(self.annotation)
            self.annotation = self.annotation[::50]  # use 1/50 val data
        print('number of instances: %s' % len(self.annotation))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.answer_list = list(set([ann["answer"] for ann in self.annotation]))
        qids = set([ann["question_id"] for ann in self.annotation])
        assert len(qids) == len(self.annotation)
        self.question_ids = [i for i in range(len(self.annotation))]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_name = "v_" + ann["video_name"]
        video_path = os.path.join(self.video_root, video_name + self.video_fmt)
        if not os.path.exists(video_path):
            video_path = os.path.join(self.video_root, video_name + '.webm')
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())

        if self.split == 'val' or self.split == 'test':
            question = pre_question(ann['question'])
            question_id = self.question_ids[index]  # TODO
            return video, question, question_id
        elif self.split == 'train':
            question = pre_question(ann['question'])
            answers = [ann['answer']]
            weights = [0.2]

            return video, question, answers, weights


class ActivityNet_retrieval_train(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        filename = f'{split}.json'
        self.annotation = load_jsonl(os.path.join(ann_root, filename))[0]

        print('number of instances: %s' % len(self.annotation))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        # remove invalid files
        remove_cnt = 0
        valid_annotation = []
        video_names = set()
        self.text = []
        self.video = []
        self.videoname2videoid = {}
        self.txt2video = {}
        self.video2txt = {}
        txt_id, video_id = 0, -1
        for ann in self.annotation:
            video_path = os.path.join(self.video_root, ann['video'])
            if not os.path.exists(video_path):
                remove_cnt += 1
                continue
            if ann['video'] not in video_names:  # new video
                video_names.add(ann['video'])
                video_id += 1
                self.video.append(ann['video'])
                self.videoname2videoid[ann['video']] = video_id
                self.video2txt[video_id] = []
            valid_annotation.append(ann)
            caption = pre_caption(' '.join(ann['caption']))
            self.text.append(caption)
            self.video2txt[video_id].append(txt_id)
            self.txt2video[txt_id] = video_id
            txt_id += 1
        self.annotation = valid_annotation
        print('remove {} invalid video-text pairs'.format(remove_cnt))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        caption = pre_caption(' '.join(ann['caption']))
        video_path = os.path.join(self.video_root, ann['video'])
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = self.videoname2videoid[ann['video']]

        return video, caption, video_id


class ActivityNet_retrieval_eval(ActivityNet_retrieval_train):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', max_words=30, split='val'):
        super(ActivityNet_retrieval_eval, self).__init__(video_root, ann_root, num_frm, frm_sampling_strategy,
                                                         max_img_size, video_fmt, max_words, split)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.video_root, ann['video'])
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm,
                                                    height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = self.videoname2videoid[ann['video']]

        return video, video_id


class CondensedMovies_retrieval_train(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mkv', max_words=30, split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        if split in ['train', 'val']:
            filename = 'train_val_challf0.csv'
        else:
            filename = 'test_challf0.csv'

        df = pd.read_csv(os.path.join(ann_root, filename))
        df = df[df.split == split]
        self.annotation = [row.to_dict() for i, row in df.iterrows()]

        print('number of instances: %s' % len(self.annotation))

        self.video_root = video_root
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        # remove invalid files
        remove_cnt = 0
        valid_annotation = []
        video_names = set()
        self.text = []
        self.video = []
        self.videoname2videoid = {}
        self.txt2video = {}
        self.video2txt = {}
        txt_id, video_id = 0, -1
        for ann in self.annotation:
            video_path = os.path.join(self.video_root, f"{ann['videoid']}{self.video_fmt}")
            if not os.path.exists(video_path):
                remove_cnt += 1
                continue
            if ann['videoid'] not in video_names:  # new video
                video_names.add(ann['videoid'])
                video_id += 1
                self.video.append(ann['videoid'])
                self.videoname2videoid[ann['videoid']] = video_id
                self.video2txt[video_id] = []
            valid_annotation.append(ann)
            caption = pre_caption(ann['caption'])
            self.text.append(caption)
            self.video2txt[video_id].append(txt_id)
            self.txt2video[txt_id] = video_id
            txt_id += 1
        self.annotation = valid_annotation
        print('remove {} invalid video-text pairs'.format(remove_cnt))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        caption = pre_caption(ann['caption'])
        video_path = os.path.join(self.video_root, f"{ann['videoid']}{self.video_fmt}")
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm,
                                                    height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = self.videoname2videoid[ann['videoid']]

        return video, caption, video_id


class CondensedMovies_retrieval_eval(CondensedMovies_retrieval_train):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mkv', max_words=30, split='val'):
        super(CondensedMovies_retrieval_eval, self).__init__(video_root, ann_root, num_frm, frm_sampling_strategy,
                                                             max_img_size, video_fmt, max_words, split)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_path = os.path.join(self.video_root, f"{ann['videoid']}{self.video_fmt}")
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm,
                                                    height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())
        video_id = self.videoname2videoid[ann['videoid']]

        return video, video_id


class ivqa(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.webm', split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        self.split = split
        filename = f"{split}.csv"
        df = pd.read_csv(os.path.join(ann_root, filename))
        annotation = [row.to_dict() for i, row in df.iterrows()]

        # self.a2id = json.load(open(os.path.join(ann_root, 'vocab.json'), "r"))
        # self.id2a = {v: k for k, v in self.a2id.items()}
        remove_cnt = 0
        self.annotation = []
        for i, ann in enumerate(annotation):
            video_path = os.path.join(video_root, f"{ann['video_id']}_{ann['start']}_{ann['end']}{video_fmt}")
            if not os.path.exists(video_path):
                remove_cnt += 1
                continue
            answer_cnt = collections.Counter(
                [ann['answer1'], ann['answer2'], ann['answer3'], ann['answer4'], ann['answer5']])
        #     answer = answer_txt.most_common(1)[0][0]
        #     answer_id = torch.zeros(len(self.a2id))
        #     for x in answer_txt:
        #         if x in self.a2id:
        #             answer_id[self.a2id[x]] = answer_txt[x]
        #     answer_txt = ", ".join([str(x) + "(" + str(answer_txt[x]) + ")" for x in answer_txt])
            ann['answer_cnt'] = answer_cnt
            self.annotation.append(ann)
        print('remove {} invalid video-text pairs'.format(remove_cnt))
        print('number of instances: %s' % len(self.annotation))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.answer_list = list(json.load(open(os.path.join(ann_root, 'vocab.json'), "r")).keys())  # TODO
        self.question_ids = [i for i in range(len(self.annotation))]
        

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_name = ann["video_id"]
        video_path = os.path.join(self.video_root, f"{video_name}_{ann['start']}_{ann['end']}{self.video_fmt}")
        vid_frm_array = load_video_from_path_decord(video_path, self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        video = self.img_norm(vid_frm_array.float())

        if self.split == 'val' or self.split == 'test':
            question = pre_question(ann['question'])
            question_id = self.question_ids[index]  # TODO
            return video, question, question_id

        elif self.split == 'train':
            question = pre_question(ann['question'])
            answers = [ann['answer1'], ann['answer2'], ann['answer3'], ann['answer4'], ann['answer5']]
            answer_weight = {}
            for answer in answers:
                if not isinstance(answer, str):
                    continue
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(answers)
                else:
                    answer_weight[answer] = 1 / len(answers)
            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            return video, question, answers, weights


class nextqa(Dataset):
    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384,
                 video_fmt='.mp4', split='train'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        self.split = split
        if split == 'test': #TODO: acquiring the hidden test data
            split = 'val'
            print('use the validation set for test')
        filename = f"{split}.csv"
        df = pd.read_csv(os.path.join(ann_root, filename))
        annotation = [row.to_dict() for i, row in df.iterrows()]

        # self.a2id = json.load(open(os.path.join(ann_root, 'vocab.json'), "r"))
        # self.id2a = {v: k for k, v in self.a2id.items()}
        remove_cnt = 0
        self.annotation = []
        with open(os.path.join(ann_root, 'map_vid_vidorID.json'),'r') as f:
            idMap = json.load(f)
        for i, ann in enumerate(annotation):
            ori_video_id = str(ann['video'])
            if ori_video_id not in idMap:
                print(f'invalid id: {ori_video_id}')
                remove_cnt += 1
                continue
            video_path = os.path.join(video_root, f"{idMap[ori_video_id]}{video_fmt}")
            if not os.path.exists(video_path):
                print(f'invalid path: {video_path}')
                remove_cnt += 1
                continue
            ann['video_path'] = video_path
            self.annotation.append(ann)
        print('remove {} invalid video-text pairs'.format(remove_cnt))
        print('number of instances: %s' % len(self.annotation))

        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.question_ids = [i for i in range(len(self.annotation))]
        self.video_rand_aug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])  

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        vid_frm_array = load_video_from_path_decord(ann['video_path'], self.frm_sampling_strategy, self.num_frm, height=self.max_img_size, width=self.max_img_size)
        if self.split == 'train':
            vid_frm_array = self.video_rand_aug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (N, H, W, C) to (N, C ,H. W)
        video = self.img_norm(vid_frm_array.float())
        question = pre_caption(ann['question'])
        #question += '? ' + ', '.join([pre_caption(ans) for ans in [ann['a0'],ann['a1'],ann['a2'],ann['a3'],ann['a4']]]) + '; '
        answers = [pre_caption(ans) for ans in [ann['a0'],ann['a1'],ann['a2'],ann['a3'],ann['a4']]]
        label = int(ann['answer'])
        return video, question, answers, label

