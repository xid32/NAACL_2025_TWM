# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. <https://arxiv.org/abs/1810.04805>"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from modules.until_module import PreTrainedModel, LayerNorm, CrossEn
from modules.module_bert import BertModel, BertConfig
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_decoder import DecoderModel, DecoderConfig

logger = logging.getLogger(__name__)


class CaptionGeneratorPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, decoder_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(CaptionGeneratorPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.decoder_config = decoder_config

        self.visual = None
        self.decoder = None

    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, decoder_config, *inputs, **kwargs)

        # assert model.bert is not None
        assert model.visual is not None

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

class NormalizeVideo(nn.Module):
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CaptionGenerator(CaptionGeneratorPreTrainedModel):
    def __init__(self, bert_config, visual_config, decoder_config, task_config):
        super(CaptionGenerator, self).__init__(bert_config, visual_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings

        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        bert = BertModel(bert_config)
        bert_word_embeddings_weight = bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder

  
        # Decoder ===>
        decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                   self.task_config, "decoder_num_hidden_layers")
        self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
        # <=== End of Decoder

        self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.normalize_video = NormalizeVideo(task_config)


        self.apply(self.init_weights)

    def forward(self, video, video_mask=None,
                input_caption_ids=None, decoder_mask=None):

        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = self.normalize_video(video)

        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        visual_output = self.get_visual_output(video, video_mask, shaped=True)

        if self.training:
            loss = 0.

            if (input_caption_ids is not None):
                decoder_scores, res_tuples = self._get_decoder_score(visual_output, video_mask,
                                                                         input_caption_ids, decoder_mask, shaped=True)
                # output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                # decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size), output_caption_ids.view(-1))
                # loss += decoder_loss

            return decoder_scores
        else:
            return None

    def get_visual_output(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]

        return visual_output


    def _get_decoder_score(self, visual_output, video_mask, input_caption_ids, decoder_mask, shaped=False):

        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        decoder_scores = self.decoder(input_caption_ids, encoder_outs=visual_output, answer_mask=decoder_mask, encoder_mask=video_mask)

        return decoder_scores, res_tuples

    def decoder_caption(self, visual_output, video_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores, _ = self._get_decoder_score(visual_output,
                                                    video_mask,
                                                    input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)

        return decoder_scores_result