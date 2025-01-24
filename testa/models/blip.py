'''
Adapted from https://github.com/salesforce/BLIP
'''
import warnings

warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from models.timesformer.models.vit import TimeSformer
from configs.config import basic_check_arguments, shared_configs, restore_training_settings
from utils import str_to_bool
from einops import rearrange


def get_custom_args(base_config):
    parser = base_config.parser
    '''
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--kinetics", type=str, default='600', help="400 or 600")
    parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')  # change base to tiny
    parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1,
                        help="0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--att_mask_expansion', type=int, default=-1,
                        help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    '''
    args = base_config.parse_args()  # change parse_args() to parse_known_args()
    return args


class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

    def forward(self, image, caption, mode):

        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)

        if mode == 'image':
            # return image features
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == 'text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            return text_output.last_hidden_state

        elif mode == 'multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return output.last_hidden_state


class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/bert_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 num_image_with_temporal_embedding=None,
                 model_cfg=None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, model_cfg=model_cfg)
        self.timesformer = False
        if vit == 'base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            # msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
        elif vit == 'large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder, default_cfgs['vit_large_patch16_224_in21k'])
        elif 'timesformer' in vit:
            self.latent_feat_size = 768
            self.timesformer = True

        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel.from_pretrained(config=med_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
        self.num_image_with_temporal_embedding = None
        if num_image_with_temporal_embedding:
            self.num_image_with_temporal_embedding = num_image_with_temporal_embedding
            self.temporal_embedding = nn.Parameter(torch.zeros(num_image_with_temporal_embedding, self.visual_encoder.embed_dim))

    def forward(self, image, caption, video=False, B=0):
        '''
        if video == True and self.timesformer == True:
            num_frames = int(image.shape[0]/B)
            num_channels = image.shape[1]
            h, w = image.shape[2], image.shape[3]
            image = image.reshape(B, num_channels, num_frames, h, w)
        '''
        image_embeds = self.visual_encoder(image)

        if video:
            if self.timesformer is True:
                pass  # (B,T*W*H,C)
            else:
                image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1])
                if self.num_image_with_temporal_embedding:
                    embs_per_image = int(image_embeds.shape[1]/self.num_image_with_temporal_embedding)
                    for frame_idx in range(self.num_image_with_temporal_embedding):
                        image_embeds[:, frame_idx*embs_per_image:(frame_idx+1)*embs_per_image] += self.temporal_embedding[frame_idx, :]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(
            image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss

        return loss_lm

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)  # [bsz, (image_size/patch_size)^2+1, 768]

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,
                                                          dim=0)  # [bsz*num_beams, (image_size/patch_size)^2+1, 768]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device)  # [bsz, (image_size/patch_size)^2+1]
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions

    def generate_based_on_video(self, video_embeds, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                                repetition_penalty=1.0):

        bsz = video_embeds.size(0)
        if not sample:
            video_embeds = video_embeds.repeat_interleave(num_beams, dim=0)

        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video_embeds.device)
        model_kwargs = {"encoder_hidden_states": video_embeds, "encoder_attention_mask": video_atts}

        prompt = [self.prompt] * bsz
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(video_embeds.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions


def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        # if model.timesformer == False:
        model, msg = load_checkpoint(model, pretrained)
        # else:
        #    model, msg = load_timesformer_checkpoint(model, pretrained)
        print(msg)
        # assert (len(msg.missing_keys) == 0)
    return model


def blip_feature_extractor(pretrained='', **kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0, model_cfg=None):
    if 'timesformer' in vit:
        vision_width = model_cfg['vision_width']
        model_cfg['use_grad_checkpointing'] = use_grad_checkpointing
        visual_encoder = TimeSformer(model_cfg)
    else:
        assert vit in ['base', 'large'], "vit parameter must be base or large"
        if vit == 'base':
            vision_width = 768
            visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                               num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                               ckpt_layer=ckpt_layer,
                                               drop_path_rate=0 or drop_path_rate
                                               )
        elif vit == 'large':
            vision_width = 1024
            visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                               num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                               ckpt_layer=ckpt_layer,
                                               drop_path_rate=0.1 or drop_path_rate
                                               )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)

    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
    if model.timesformer is False:
        for key in model.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != model.state_dict()[key].shape:
                    del state_dict[key]
    else:
        '''
        num_patches = model.visual_encoder.num_patches
        if num_patches + 1 != state_dict['visual_encoder.pos_embed'].size(1):
            pos_embed = state_dict['visual_encoder.pos_embed']
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            state_dict['visual_encoder.pos_embed'] = new_pos_embed
        '''
        if 'visual_encoder.time_embed' in state_dict and model.num_frames != state_dict['visual_encoder.time_embed'].size(1):
            time_embed = state_dict['visual_encoder.time_embed'].transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(model.num_frames), mode='nearest')
            state_dict['visual_encoder.time_embed'] = new_time_embed.transpose(1, 2)
            if 'visual_encoder_m.time_embed' in state_dict and model.num_frames != state_dict['visual_encoder_m.time_embed'].size(1):
                state_dict['visual_encoder_m.time_embed'] = new_time_embed.transpose(1, 2)

        ## Initializing temporal attention
        attention_type = model.visual_encoder.attention_type
        if attention_type == 'divided_space_time':
            new_state_dict = state_dict.copy()
            for key in state_dict:
                if 'bert' in key:
                    continue
                if 'blocks' in key and 'attn' in key:
                    new_key = key.replace('attn', 'temporal_attn')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                if 'blocks' in key and 'norm1' in key:
                    new_key = key.replace('norm1', 'temporal_norm1')
                    if not new_key in state_dict:
                        new_state_dict[new_key] = state_dict[key]
                    else:
                        new_state_dict[new_key] = state_dict[new_key]
                if 'queue' in key:
                    del new_state_dict[key]
            state_dict = new_state_dict
        else:
            new_state_dict = state_dict.copy()
            for key in state_dict:
                if 'queue' in key:
                    del new_state_dict[key]
            state_dict = new_state_dict

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg
