import json
import os
import random
import torch
from pandas import Categorical
import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import decord
from decord import VideoReader
from data.utils import pre_caption
import os,glob

from .randaugment import TemporalConsistentRandomAugment

decord.bridge.set_bridge('torch')

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform): 

        self.ann_pretrain = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann_pretrain += ann
        
        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path,'*.json'))

            print('loading '+self.laion_files[0])
            with open(self.laion_files[0],'r') as f:
                self.ann_laion = json.load(f)  

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain
            
        self.transform = transform


    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]   
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        
        return image, caption
'''
config['video_files'], config['image_files'], transform_image_pseudo_video_train,
            num_frm=config['num_frm_train'], max_img_size=config['image_size'], max_words=config['max_words'], 
            frm_sampling_strategy='uniform'
'''
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


class pretrain_video_dataset(Dataset):

    def __init__(self, video_files, image_transform, num_frm=8, frm_sampling_strategy="uniform",
        max_img_size=224, max_words=30, video_resize=256,):

        self.ann_pretrain = []
        for f in video_files:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann_pretrain += ann

        print('number of instances: %s' % (len(self.ann_pretrain)))
        self.size = len(self.ann_pretrain)
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_resize = video_resize
        self.video_random_cropper = VideoRandomSquareCrop(max_img_size)
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.video_rand_aug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        self.max_words = max_words
        self.transform = image_transform


    def __len__(self):
        return len(self.ann_pretrain)

    def __getitem__(self, index):
        ann = self.ann_pretrain[index]

        video_path = ann['path']

        try:
            video_flag = False
            common_video_formats = ['mp4', 'avi', 'mov', 'flv', 'webm']
            for suffix in common_video_formats:
                if video_path.endswith(suffix):
                    video_flag = True
                    break
            if video_flag == True:
                vid_frm_array = self._load_video_from_path_decord(video_path, height=self.video_resize, width=self.video_resize)
                vid_frm_array = self.video_random_cropper(vid_frm_array)
                vid_frm_array = self.video_rand_aug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                video = self.img_norm(vid_frm_array.float())
            else:
                image = Image.open(video_path).convert('RGB')   
                image = self.transform(image)
                video = image.repeat(self.num_frm, 1, 1, 1)
            caption = pre_caption(ann['caption'], self.max_words)
        except Exception as e:
            print(e)
            return self.__getitem__(random.choice(list(range(self.size))))

        return video, caption

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
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm).astype(int)
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
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms