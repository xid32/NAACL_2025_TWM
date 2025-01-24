import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ipdb import set_trace
import timm
from einops import rearrange, repeat
import os

import loralib as lora
from transformers.activations import get_activation
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint

from htsat import HTSAT_Swin_Transformer
import esc_config as esc_config
from utils import do_mixup, get_mix_lambda, do_mixup_label

def batch_organize(out_match_posi, out_match_nega):
	# audio B 512
	# posi B 512
	# nega B 512

	out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
	batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
	for i in range(out_match_posi.shape[0]):
		out_match[i * 2, :] = out_match_posi[i, :]
		out_match[i * 2 + 1, :] = out_match_nega[i, :]
		batch_labels[i * 2] = 1
		batch_labels[i * 2 + 1] = 0
	
	return out_match, batch_labels


class AVQA_AVatt_Grounding(nn.Module):

    def __init__(self, opt):
        super(AVQA_AVatt_Grounding, self).__init__()
        self.opt = opt
        # for features
        self.fc_a1 =  nn.Linear(768, 1536)
        self.fc_a2=nn.Linear(1536, 1536)

        # combine
        self.fc1 = nn.Linear(1536+1536, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_gl=nn.Linear(1536+1536, 1536)
        self.tanh = nn.Tanh()
        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
        
        
        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 50
            esc_config.checkpoint_path = "../checkpoints/ESC-50/"
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 527
            esc_config.checkpoint_path = "../checkpoints/AudioSet/"
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = "../checkpoints/SCV2/"
            esc_config.checkpoint = "HTSAT_SCV2_Saved_2.ckpt"
        else:
            raise NotImplementedError
    
        self.htsat = HTSAT_Swin_Transformer(
            spec_size=esc_config.htsat_spec_size,
            patch_size=esc_config.htsat_patch_size,
            in_chans=1,
            num_classes=esc_config.classes_num,
            window_size=esc_config.htsat_window_size,
            config = esc_config,
            depths = esc_config.htsat_depth,
            embed_dim = esc_config.htsat_dim,
            patch_stride=esc_config.htsat_stride,
            num_heads=esc_config.htsat_num_head
        )
        
        checkpoint_path = os.path.join(esc_config.checkpoint_path, esc_config.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        tmp = {k[10:]:v for k, v in tmp['state_dict'].items()}
        self.htsat.load_state_dict(tmp, strict=True)
        
    def forward(self, video_id, audio, visual, mixup_lambda=None):
        
        bs,t,c,h,w = visual.shape
        
        audio = audio[:, 0] # [B, 32000]
        waveform = audio
        
        audio = self.htsat.spectrogram_extractor(audio)
        audio = self.htsat.logmel_extractor(audio)        
        audio = audio.transpose(1, 3)
        audio = self.htsat.bn0(audio)
        audio = audio.transpose(1, 3)
        if self.htsat.training:
            audio = self.htsat.spec_augmenter(audio)
        if self.htsat.training and mixup_lambda is not None:
            audio = do_mixup(audio, mixup_lambda)

        if audio.shape[2] > self.htsat.freq_ratio * self.htsat.spec_size:
            audio = self.htsat.crop_wav(audio, crop_size=self.htsat.freq_ratio * self.htsat.spec_size)
            audio = self.htsat.reshape_wav2img(audio)
        else: # this part is typically used, and most easy one
            audio = self.htsat.reshape_wav2img(audio)
        frames_num = audio.shape[2]
        f_a = self.htsat.patch_embed(audio)
        if self.htsat.ape:
            f_a = f_a + self.htsat.absolute_pos_embed
        f_a = self.htsat.pos_drop(f_a) 

        for htsat_blk in self.htsat.layers:
            for blk in htsat_blk.blocks:
                f_a, _ = blk(f_a)
            if htsat_blk.downsample is not None:
                f_a = htsat_blk.downsample(f_a)
        
        with torch.no_grad():
            visual_posi = visual[:, 0]
            visual_nega = visual[:, 1]
            visual_nega = self.swin.forward_features(visual_nega)
            visual_posi = self.swin.forward_features(visual_posi)
            
        visual = torch.cat((visual_posi.unsqueeze(1), visual_nega.unsqueeze(1)), dim=1)
        
        ### -------> yb: cvpr use
        f_a = f_a.mean(dim=1) # [B, 768]
        audio = f_a.unsqueeze(1)
        audio = torch.cat((audio, audio), dim=1)
        ### <-----
        
        ## audio features
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)                      # [B, 2, 1536]
        (B, T, C) = audio_feat.size() # [B, 2, 1536]
        audio_feat = audio_feat.view(B*T, C)                # [B*2, 1536]
        
        B, T, hw, C = visual.size()
        visual = visual.view(B*T, hw, C)
        visual = visual.permute(0, 2, 1)
        visual = visual.view(B*T, C, 6, 6)
        v_feat_out_swin = visual                  # [B*2, 1536, 6, 6]
        v_feat=self.avgpool(v_feat_out_swin)
        visual_feat_before_grounding=v_feat.squeeze()     # B*2 1536
        
        (B, C, H, W) = v_feat_out_swin.size()
        v_feat = v_feat_out_swin.view(B, C, H * W)
        v_feat = v_feat.permute(0, 2, 1)  # B, HxW, C
        visual = nn.functional.normalize(v_feat, dim=2)
             
        ## audio-visual grounding
        audio_feat_aa = audio_feat.unsqueeze(-1)            # [B*2, 512, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)
        visual_feat = visual
        x2_va = torch.matmul(visual_feat, audio_feat_aa).squeeze()

        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)       # [320, 1, 36]
        visual_feat_grd = torch.matmul(x2_p, visual_feat)
        visual_feat_grd = visual_feat_grd.squeeze()         # [320, 1536]   

        visual_gl=torch.cat((visual_feat_before_grounding, visual_feat_grd),dim=-1)
        visual_feat_grd=self.tanh(visual_gl)
        visual_feat_grd=self.fc_gl(visual_feat_grd)

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd), dim=-1)     # [320, 3072]

        feat = F.relu(self.fc1(feat))   # (3072, 512)
        feat = F.relu(self.fc2(feat))   # (512, 256)
        feat = F.relu(self.fc3(feat))   # (256, 128)
        feat = self.fc4(feat)   # (128, 2)

        return  feat
