from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
import time
from einops import rearrange, repeat
import argparse
from torch.utils.data import DataLoader
from alvs_inter import *
import pickle
from tqdm import tqdm

from dataloader import MSRVTT_ALVS_DataLoader
from v2tactiongraph.main_alvs import get_args

torch.distributed.init_process_group(backend="nccl")

global logger





def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_ALVS_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        data_geometric_path=args.data_geometric_path,
        max_words=args.max_words,
        use_geometric_hdf5=args.use_geometric_hdf5,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        node_features=args.node_features,
        inference_path=None
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_val_test(args, tokenizer, split_type="test", ):
    msrvtt_testset = MSRVTT_ALVS_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        # fo_path=args.fo_path,
        # stgraph_path=args.stgraph_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        inference_path=None
        # node_features=args.node_features
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)







DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train, "val": dataloader_msrvtt_val_test,
                             "test": dataloader_msrvtt_val_test}

device = "cuda"

class ALVS(nn.Module):
    def __init__(self):
        super(ALVS, self).__init__()
        self.visual_mapping = nn.Linear(768, 30)

    def forward(self, f_v, f_t):
        bs, cs, t, d = f_v.shape
        f_v = rearrange(f_v.squeeze(), 'bs t d -> (bs t) d')
        f_v = self.linear(f_v)
        f_t = repeat(f_t, 'b dim -> b t dim', t=t)
        f_t = rearrange(f_t.squeeze(), 'b t d -> (b t) d')
        return f_v, f_t

    def forward_visual(self, f_v):
        bs, cs, t, d = f_v.shape
        f_v = rearrange(f_v.squeeze(), 'bs t d -> (bs t) d')
        f_v = self.linear(f_v)
        f_v = rearrange(f_v.squeeze(), '(bs t) d -> bs t d', bs=bs, t=t)
        return f_v


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


def inference(model, args, tokenizer, device):
    save_path = f"./datasets/caption/MSRVTT/idxs"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer, "test")
    model.frozen_model()
    for step, batch_ in test_dataloader:
        input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, geo_x, geo_edge_index, geo_edge_attr, video_id = batch_
        batch = (input_ids, input_mask, segment_ids, video, video_mask, \
                 pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                 pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, geo_x, geo_edge_index,
                 geo_edge_attr)
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, geo_x, geo_edge_index, geo_edge_attr = batch
        f_v = model.forward_visual(video).squeeze()
        s_inx = iterative_sampling(f_v, input_ids.squeeze(), 3, 3, 0.5, 0.5)
        with open(os.path.join(save_path, f"{video_id}.pkl"), "wb") as f:
            pickle.dump(s_inx, f)
    return





def main():
    args = get_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    loss_info = InfoNCELoss(1)
    model = ALVS().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(0):
        for step, batch_ in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, geo_x, geo_edge_index, geo_edge_attr, video_id = batch_
            batch = (input_ids, input_mask, segment_ids, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, geo_x, geo_edge_index, geo_edge_attr)
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

            input_ids, input_mask, segment_ids, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, geo_x, geo_edge_index, geo_edge_attr = batch

            optimizer.zero_grad()

            f_v, f_t = model(video, input_ids)
            loss = loss_info(f_v, f_t)

            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f"./alvs.pt")
    model.load_state_dict(torch.load(f"./datasets/caption/MSRVTT/models/alvs.pt"))
    model.eval()
    inference(model, args, tokenizer, device)
    return






if __name__ == "__main__":
    main()
