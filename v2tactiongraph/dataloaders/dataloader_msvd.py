from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from rawvideo_util import RawVideoExtractor

# Based on https://github.com/ArrowLuo/CLIP4Clip

class MSVD_Loader(Dataset):
    """Implementation of the dataloader for MSVD. Mainly used in the feature extraction process.
    Params:
        data_path: Path to the MSVD folder.
        features_path: Path to the extracted feature file.
        max_words: Max word length retained. Any more than the value will be truncated. Default: 30
        feature_framerate: sampling rate in second. Default: 1.0
        max_frames: Max frame sampled. Any more than the value will be ignored. Default: 100
        image_resolution: Processed image's width and height, in pixel. If param transform_type = 0 and
            the original image is greater than this value, it will be resized and center cropped. Default: 224
        frame_order: 0: ordinary order; 1: reverse order; 2: random order. Default: 0
        slice_framepos: 0: sample from the first frames; 1: sample from the last frames;
            2: sample uniformly. Default: 0
        transform_type: 0: default transformation; 1: transformation for objects, iou, temporal, action;
            2: transformation for i3d;. Default: 0
    """

    def __init__(
            self,
            data_path,
            features_path,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            transform_type =0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.id_dict = {}
        self.transform_type = transform_type

        
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        url2id = {}
        for line in open(data_path, 'r').readlines():
            url2id[line.strip().split(' ')[0]] = line.strip().split(' ')[-1]

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = url2id[".".join(video_file.split(".")[:-1])]

                self.id_dict[len(self.id_dict)] = video_id_

                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        
       
        print("Video number: {}".format(len(self.video_dict)))
        print("Id number: {}".format(len(self.id_dict)))



        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution,type = self.transform_type)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.id_dict)

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)



        for i, video_id in enumerate(choice_video_ids):

            video_path = self.video_dict[video_id]

            raw_video_data,shapes = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            
              # Pair x L x T x 3 x H x W
            video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                              shapes[2], shapes[3]), dtype=np.float)

            # Pair x L x T x 3 x H x W
            video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                              shapes[2], shapes[3]), dtype=np.float)

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice

            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        #print(idx)
        if idx >=len(self):raise IndexError
        video_id = self.id_dict[idx]
        choice_video_ids = [video_id]
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return video_id, video, video_mask
