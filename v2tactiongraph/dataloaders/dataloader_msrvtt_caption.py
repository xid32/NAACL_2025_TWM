from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
import json
import random
import h5py
from tqdm import tqdm
import dgl
from scipy import sparse
import h5py
import glob

class MSRVTT_Caption_DataLoader(Dataset):
    """Implementation of the dataloader for MSRVTT. Mainly used in the model training and evaluation.
    Params:
        json_path: Path to the MSRVTT_data.json file.
        features_path: Path to the extracted feature file.
        inference_path: Path to the inference folder that contains videos to predict.
        tokenizer: Tokenizer used for tokenizing the caption.
        fo_path: Path to the object feature file. Default: None
        stgraph_path: Path to the STGraph feature file. Default: None
        data_geometric_path: Path to the geometric feature file. Default: None
        use_geometric_hdf5: Whether the geometric feature file is in hdf5 format. Default: False
        data_object_dgl_path: Path to the object DGL feature file. Default: None
        max_words: Max word length retained. Any more than the value will be truncated. Default: 30
        feature_framerate: sampling rate in second. Default: 1.0
        max_frames: Max frame sampled. Any more than the value will be ignored. Default: 100
        split_type: Either "train", "val", or "test". Default: ""
        node_features: Either "dgl", "geometric", or "both". Default: ""
    """
    def __init__(
            self,
            json_path,
            features_path,
            inference_path,
            tokenizer,
            fo_path=None,
            stgraph_path=None,
            data_geometric_path=None,
            use_geometric_hdf5=False,
            data_object_dgl_path=None,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
            node_features = "", # 'dgl' or 'geometric' or 'both'
    ):
        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.node_features = node_features


        self.feature_size = self.feature_dict[next(iter(self.feature_dict))].shape[-1]
        self.inference_path = inference_path

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        if self.inference_path is not None:
            inference_files = glob.glob(self.inference_path + "/*")
            inference_files = [x.split("/")[-1].split(".mp4")[0] for x in inference_files]
            choiced_video_ids = [x for x in choiced_video_ids if x in inference_files]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

        
        if self.node_features == 'dgl' or self.node_features=="both":
            with open(data_object_dgl_path, 'rb') as fdgl:
                self.data_object_dgl = pickle.load(fdgl)

        self.use_geometric_hdf5 = use_geometric_hdf5
        if self.node_features == 'geometric' or self.node_features=="both":
            if not self.use_geometric_hdf5:
                geo_graph = []
                print(data_geometric_path)
                with (open(data_geometric_path, "rb")) as openfile:
                    while True:
                        try:
                            geo_graph.append(pickle.load(openfile))
                        except Exception as e:
                            print(e)
                            break
                # geo_graph = {
                #     key.replace("video", ""): value for key, value in geo_graph[0].items()
                # }
                # geo_graph = {int(k):v for k,v in geo_graph.items()}
                geo_graph = geo_graph[0]
                self.data_object_geo =  {}
                for k in geo_graph.keys():
                    if k in choiced_video_ids:
                        self.data_object_geo[k] = {'x': geo_graph[k].x, 'edge_index': geo_graph[k].edge_index, 'edge_attr': geo_graph[k].edge_attr }
                geo_graph = None
                del geo_graph
            else:
                self.data_object_geo = h5py.File(data_geometric_path, 'r')

            
    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]


            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
            assert len(input_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, np.array([]), np.array([]), np.array([]), np.array([]), \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)

        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length


        return video, video_mask, np.array([]), np.array([])

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        

        pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, masked_video, video_labels_index = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

        

        # data_object = {'fo':sparse.csr_matrix([]),'weights':sparse.csr_matrix([]),'uv': sparse.csr_matrix([])}
        if self.node_features == 'dgl':
            data_object = self.data_object_dgl[video_id]

            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               data_object['fo'].toarray(), data_object['weights'].toarray(), data_object['uv'].toarray()

        elif self.node_features == 'geometric':
            if not self.use_geometric_hdf5:
                data_object = {'x':self.data_object_geo[video_id]['x'],'edge_index':self.data_object_geo[video_id]['edge_index'],'edge_attr':self.data_object_geo[video_id]['edge_attr'].todense()}
                # print(type(self.data_object_geo[video_id]['edge_attr']))
            else:
                data_object = {'x': self.data_object_geo[video_id+'-x'][:], 'edge_index': self.data_object_geo[video_id+'-edge_index'][:],'edge_attr':self.data_object_geo[video_id+'-edge_attr'][:]}

            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               data_object['x'], data_object['edge_index'], data_object['edge_attr']
        elif self.node_features == "both":
            data_object_dgl = self.data_object_dgl[video_id]
            if not self.use_geometric_hdf5:
                data_object_geo = {'x':self.data_object_geo[video_id]['x'],'edge_index':self.data_object_geo[video_id]['edge_index'],'edge_attr':self.data_object_geo[video_id]['edge_attr'].todense()}
            else:
                data_object_geo = {'x': self.data_object_geo[video_id+'-x'][:], 'edge_index': self.data_object_geo[video_id+'-edge_index'][:],'edge_attr':self.data_object_geo[video_id+'-edge_attr'][:]}

            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               data_object_dgl['fo'].toarray(), data_object_dgl['weights'].toarray(), data_object_dgl['uv'].toarray(), \
               data_object_geo['x'], data_object_geo['edge_index'], data_object_geo['edge_attr']

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids


    def __stack_object_features(self, pathfile):

        fo_input = h5py.File(pathfile, "r")
        fo_list = {}
        for i,key in tqdm(enumerate(fo_input.keys()), total=len(fo_input.keys())):
            a = key.split('-')

            if a[0] not in fo_list:
                fo_list[a[0]] = {}
            fo_list[a[0]][int(a[1])] = fo_input[key][:]

        fo_stacked = {}
        for key in fo_list.keys():
            stacked = []
            for k_fr in sorted(fo_list[key].keys()):
                stacked.append(fo_list[key][k_fr])
            fo_stacked[key] = np.vstack(stacked)

        return fo_stacked
    
    def __generate_object_features(self, fo_path, stgraph_path):
    
        fo_file = self.__stack_object_features(fo_path)
        stgraph_file = h5py.File(stgraph_path, "r")

        # Create a video_id query
        chosen_keys = ["video%s" % x for x in range(len(fo_file))]

        fo_input, stgraph = {}, {}
        for key in chosen_keys:
            fo_input[key] = sparse.csr_matrix(np.array(fo_file.get(key)).astype(np.float))
            stgraph[key] = stgraph_file.get(key)

        return fo_input, stgraph


class MSRVTT_ALVS_DataLoader(Dataset):
    """Implementation of the dataloader for MSRVTT. Mainly used in the model training and evaluation.
    Params:
        json_path: Path to the MSRVTT_data.json file.
        features_path: Path to the extracted feature file.
        inference_path: Path to the inference folder that contains videos to predict.
        tokenizer: Tokenizer used for tokenizing the caption.
        fo_path: Path to the object feature file. Default: None
        stgraph_path: Path to the STGraph feature file. Default: None
        data_geometric_path: Path to the geometric feature file. Default: None
        use_geometric_hdf5: Whether the geometric feature file is in hdf5 format. Default: False
        data_object_dgl_path: Path to the object DGL feature file. Default: None
        max_words: Max word length retained. Any more than the value will be truncated. Default: 30
        feature_framerate: sampling rate in second. Default: 1.0
        max_frames: Max frame sampled. Any more than the value will be ignored. Default: 100
        split_type: Either "train", "val", or "test". Default: ""
        node_features: Either "dgl", "geometric", or "both". Default: ""
    """

    def __init__(
            self,
            json_path,
            features_path,
            inference_path,
            tokenizer,
            fo_path=None,
            stgraph_path=None,
            data_geometric_path=None,
            use_geometric_hdf5=False,
            data_object_dgl_path=None,
            max_words=30,
            feature_framerate=1.0,
            max_frames=1200,
            split_type="",
            node_features="",  # 'dgl' or 'geometric' or 'both'
    ):
        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.node_features = node_features

        self.feature_size = self.feature_dict[next(iter(self.feature_dict))].shape[-1]
        self.inference_path = inference_path

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        if self.inference_path is not None:
            inference_files = glob.glob(self.inference_path + "/*")
            inference_files = [x.split("/")[-1].split(".mp4")[0] for x in inference_files]
            choiced_video_ids = [x for x in choiced_video_ids if x in inference_files]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

        if self.node_features == 'dgl' or self.node_features == "both":
            with open(data_object_dgl_path, 'rb') as fdgl:
                self.data_object_dgl = pickle.load(fdgl)

        self.use_geometric_hdf5 = use_geometric_hdf5
        if self.node_features == 'geometric' or self.node_features == "both":
            if not self.use_geometric_hdf5:
                geo_graph = []
                print(data_geometric_path)
                with (open(data_geometric_path, "rb")) as openfile:
                    while True:
                        try:
                            geo_graph.append(pickle.load(openfile))
                        except Exception as e:
                            print(e)
                            break
                # geo_graph = {
                #     key.replace("video", ""): value for key, value in geo_graph[0].items()
                # }
                # geo_graph = {int(k):v for k,v in geo_graph.items()}
                geo_graph = geo_graph[0]
                self.data_object_geo = {}
                for k in geo_graph.keys():
                    if k in choiced_video_ids:
                        self.data_object_geo[k] = {'x': geo_graph[k].x, 'edge_index': geo_graph[k].edge_index,
                                                   'edge_attr': geo_graph[k].edge_attr}
                geo_graph = None
                del geo_graph
            else:
                self.data_object_geo = h5py.File(data_geometric_path, 'r')

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
            assert len(input_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, np.array([]), np.array([]), np.array([]), np.array([]), \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)

        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask, np.array([]), np.array([])

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        pairs_text, pairs_mask, pairs_segment, \
            pairs_masked_text, pairs_token_labels, \
            pairs_input_caption_ids, pairs_decoder_mask, \
            pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)

        pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, masked_video, video_labels_index = np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # data_object = {'fo':sparse.csr_matrix([]),'weights':sparse.csr_matrix([]),'uv': sparse.csr_matrix([])}
        if self.node_features == 'dgl':
            data_object = self.data_object_dgl[video_id]

            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
                data_object['fo'].toarray(), data_object['weights'].toarray(), data_object['uv'].toarray()

        elif self.node_features == 'geometric':
            if not self.use_geometric_hdf5:
                data_object = {'x': self.data_object_geo[video_id]['x'],
                               'edge_index': self.data_object_geo[video_id]['edge_index'],
                               'edge_attr': self.data_object_geo[video_id]['edge_attr'].todense()}
                # print(type(self.data_object_geo[video_id]['edge_attr']))
            else:
                data_object = {'x': self.data_object_geo[video_id + '-x'][:],
                               'edge_index': self.data_object_geo[video_id + '-edge_index'][:],
                               'edge_attr': self.data_object_geo[video_id + '-edge_attr'][:]}

            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
                data_object['x'], data_object['edge_index'], data_object['edge_attr']
        elif self.node_features == "both":
            data_object_dgl = self.data_object_dgl[video_id]
            if not self.use_geometric_hdf5:
                data_object_geo = {'x': self.data_object_geo[video_id]['x'],
                                   'edge_index': self.data_object_geo[video_id]['edge_index'],
                                   'edge_attr': self.data_object_geo[video_id]['edge_attr'].todense()}
            else:
                data_object_geo = {'x': self.data_object_geo[video_id + '-x'][:],
                                   'edge_index': self.data_object_geo[video_id + '-edge_index'][:],
                                   'edge_attr': self.data_object_geo[video_id + '-edge_attr'][:]}

            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
                data_object_dgl['fo'].toarray(), data_object_dgl['weights'].toarray(), data_object_dgl['uv'].toarray(), \
                data_object_geo['x'], data_object_geo['edge_index'], data_object_geo['edge_attr']

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, video_id

    def __stack_object_features(self, pathfile):

        fo_input = h5py.File(pathfile, "r")
        fo_list = {}
        for i, key in tqdm(enumerate(fo_input.keys()), total=len(fo_input.keys())):
            a = key.split('-')

            if a[0] not in fo_list:
                fo_list[a[0]] = {}
            fo_list[a[0]][int(a[1])] = fo_input[key][:]

        fo_stacked = {}
        for key in fo_list.keys():
            stacked = []
            for k_fr in sorted(fo_list[key].keys()):
                stacked.append(fo_list[key][k_fr])
            fo_stacked[key] = np.vstack(stacked)

        return fo_stacked

    def __generate_object_features(self, fo_path, stgraph_path):

        fo_file = self.__stack_object_features(fo_path)
        stgraph_file = h5py.File(stgraph_path, "r")

        # Create a video_id query
        chosen_keys = ["video%s" % x for x in range(len(fo_file))]

        fo_input, stgraph = {}, {}
        for key in chosen_keys:
            fo_input[key] = sparse.csr_matrix(np.array(fo_file.get(key)).astype(np.float))
            stgraph[key] = stgraph_file.get(key)

        return fo_input, stgraph
