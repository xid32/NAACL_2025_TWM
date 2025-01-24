from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import testa


class TESTA_VQA(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 token_merging=False,
                 testa_r=0,
                 merging_type=None,
                 model_cfg=None,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0.1, model_cfg)
        self.timesformer = False
        if 'timesformer' in vit:
            self.latent_feat_size = vision_width
            self.timesformer = True
        self.num_frames = model_cfg["num_frames"]

        self.token_merging = token_merging
        self.merging_type = merging_type
        if token_merging:
            if self.timesformer:
                testa.patch.timesformer(self.visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,
                                        merging_type=merging_type, num_patches=self.visual_encoder.num_patches)
            else:
                testa.patch.vit(self.visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,
                                merging_type=merging_type)
            self.visual_encoder.r = testa_r

        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, video, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):

        video_embeds = self.visual_encoder(video)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video.device)

        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                  return_tensors="pt").to(video.device)
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        if train:
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(video.device)
            answer.input_ids[:, 0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=video_embeds,
                                                encoder_attention_mask=video_atts,
                                                return_dict=True)

            question_states = []
            question_atts = []
            for b, n in enumerate(n):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [question.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

            loss = weights * answer_output.loss
            loss = loss.sum() / video.size(0)

            return loss
        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=video_embeds,
                                                encoder_attention_mask=video_atts,
                                                return_dict=True)

            if inference == 'generate':
                num_beams = 3
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
                question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask": question_atts}

                bos_ids = torch.full((video.size(0), 1), fill_value=self.tokenizer.bos_token_id, device=video.device)

                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id,
                                                     **model_kwargs)

                answers = []
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)
                    answers.append(answer)
                return answers

            elif inference == 'rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                           answer.input_ids, answer.attention_mask, k_test)
                return max_ids

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        return max_ids


def testa_vqa(pretrained='', **kwargs):
    model = TESTA_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))


class TESTA_ChoiceVQA(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 token_merging=False,
                 testa_r=0,
                 merging_type=None,
                 model_cfg=None,
                 num_choices=5,
                 classifier_dropout=0.1
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0.1, model_cfg)
        self.timesformer = False
        if 'timesformer' in vit:
            self.latent_feat_size = vision_width
            self.timesformer = True
        self.num_frames = model_cfg["num_frames"]
        self.num_choices = num_choices

        self.token_merging = token_merging
        self.merging_type = merging_type
        if token_merging:
            if self.timesformer:
                testa.patch.timesformer(self.visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,
                                        merging_type=merging_type, num_patches=self.visual_encoder.num_patches)
            else:
                testa.patch.vit(self.visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,
                                merging_type=merging_type)
            self.visual_encoder.r = testa_r

        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(encoder_config.hidden_size, 1)

    def forward(self, video, questions, answers_list, labels, train=True):

        video_embeds = self.visual_encoder(video)
        video_embeds = video_embeds.repeat_interleave(self.num_choices, dim=0)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video.device)

        qa_texts = []
        for question, answers in zip(questions, answers_list):
            for answer in answers:
                qa_texts.append(question + "? " + answer)
        qa_pairs = self.tokenizer(qa_texts, padding='longest', truncation=True, max_length=50,
                                  return_tensors="pt").to(video.device)
        qa_pairs.input_ids[:, 0] = self.tokenizer.enc_token_id

        question_outputs = self.text_encoder(qa_pairs.input_ids,
                                             attention_mask=qa_pairs.attention_mask,
                                             encoder_hidden_states=video_embeds,
                                             encoder_attention_mask=video_atts,
                                             return_dict=True)
        hidden_states = question_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(hidden_states))
        reshaped_logits = logits.view(-1, self.num_choices)
        if train is True:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            preds = np.argmax(reshaped_logits.cpu().numpy(), axis=1)
            return preds


def testa_vqa(pretrained='', **kwargs):
    model = TESTA_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model


def testa_choice_vqa(pretrained='', **kwargs):
    model = TESTA_ChoiceVQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model
