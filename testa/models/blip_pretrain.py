'''
Adapted from https://github.com/salesforce/BLIP
'''
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers

transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from models.blip import create_vit, init_tokenizer, load_checkpoint


class BLIP_Pretrain(nn.Module):
    def __init__(self,
                 med_config='configs/bert_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                 model_cfg=None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0, model_cfg)
        if vit == 'base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        elif vit == 'large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder, default_cfgs['vit_large_patch16_224_in21k'])
        elif 'timesformer' in vit:
            self.latent_feat_size = 768
            self.timesformer = True

        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=encoder_config,
                                                      add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0, model_cfg)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')

    def forward(self, image, caption, alpha, video=False, B=0):
        # image: [batch_size* num_frames, channels, width, height]
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)

        if video == False:
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]),
                                     dim=-1)  # image: (batch_size, hidden_dim)
        else:
            if self.timesformer is True:
                image_feat = self.vision_proj(
                    torch.mean(image_embeds, dim=1))  # mean pooling --> (batch_size, hidden_dim)
                image_feat = F.normalize(image_feat, dim=-1)
            else:
                image_feat = self.vision_proj(image_embeds[:, 0, :])  # video: (batch_size*num_frames, hidden_dim)
                image_feat = image_feat.reshape(B, -1, image_feat.shape[-1])
                image_feat = torch.mean(image_feat, dim=1)  # video: (batch_size, hidden_size)
                image_feat = F.normalize(image_feat, dim=-1)
                image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30,
                              return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            if video == False:
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            else:
                if self.timesformer is True:
                    image_feat_m = self.vision_proj_m(
                        torch.mean(image_embeds_m, dim=1))  # mean pooling --> (batch_size, hidden_dim)
                    image_feat_m = F.normalize(image_feat_m, dim=-1)
                else:
                    image_feat_m = self.vision_proj_m(
                        image_embeds_m[:, 0, :])  # video: (batch_size*num_frames, hidden_dim)
                    image_feat_m = image_feat_m.reshape(B, -1, image_feat_m.shape[-1])
                    image_feat_m = torch.mean(image_feat_m, dim=1)  # video: (batch_size, hidden_size)
                    image_feat_m = F.normalize(image_feat_m, dim=-1)
                    image_embeds_m = image_embeds.reshape(B, -1, image_embeds_m.shape[-1])
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        if video == False:
            bs = image.size(0)
        else:
            bs = B
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

            # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= LM ========================##     
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        decoder_output = self.text_decoder(decoder_input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )

        loss_lm = decoder_output.loss
        return loss_ita, loss_itm, loss_lm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


class BLIP_VideoImage_Pretrain(BLIP_Pretrain):
    def __init__(self,
                 med_config='configs/bert_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                 model_cfg=None
                 ):
        super(BLIP_VideoImage_Pretrain, self).__init__(med_config, image_size,
                                                       vit, vit_grad_ckpt, vit_ckpt_layer, embed_dim, queue_size,
                                                       momentum, model_cfg)

    def forward(self, video, caption, B1, video_2, caption_2, B2, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        video_embeds = self.visual_encoder(video)
        video_embeds_2 = self.visual_encoder(video_2)

        if self.timesformer is True:
            video_feat = self.vision_proj(torch.mean(video_embeds, dim=1))  # mean pooling --> (batch_size, hidden_dim)
            video_feat = F.normalize(video_feat, dim=-1)
            video_feat_2 = self.vision_proj(torch.mean(video_embeds_2, dim=1))
            video_feat_2 = F.normalize(video_feat_2, dim=-1)
        else:
            video_feat = self.vision_proj(video_embeds[:, 0, :])  # video: (batch_size*num_frames, hidden_dim)
            video_feat = video_feat.reshape(B1, -1, video_feat.shape[-1])
            video_feat = torch.mean(video_feat, dim=1)  # video: (batch_size, hidden_size)
            video_feat = F.normalize(video_feat, dim=-1)
            video_embeds = video_embeds.reshape(B1, -1, video_embeds.shape[-1])

            video_feat_2 = self.vision_proj(video_embeds_2[:, 0, :])  # video: (batch_size*num_frames, hidden_dim)
            video_feat_2 = video_feat_2.reshape(B2, -1, video_feat_2.shape[-1])
            video_feat_2 = torch.mean(video_feat_2, dim=1)  # video: (batch_size, hidden_size)
            video_feat_2 = F.normalize(video_feat_2, dim=-1)
            video_embeds_2 = video_embeds_2.reshape(B2, -1, video_embeds_2.shape[-1])

        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video.device)
        video_atts_2 = torch.ones(video_embeds_2.size()[:-1], dtype=torch.long).to(video_2.device)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            video.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                        mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        text_2 = self.tokenizer(caption_2, padding='max_length', truncation=True, max_length=30,
                                return_tensors="pt").to(video_2.device)
        text_output_2 = self.text_encoder(text_2.input_ids, attention_mask=text_2.attention_mask, return_dict=True,
                                          mode='text')
        text_feat_2 = F.normalize(self.text_proj(text_output_2.last_hidden_state[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            video_embeds_m = self.visual_encoder_m(video)
            video_embeds_m_2 = self.visual_encoder_m(video_2)

            if self.timesformer is True:
                video_feat_m = self.vision_proj(
                    torch.mean(video_embeds_m, dim=1))  # mean pooling --> (batch_size, hidden_dim)
                video_feat_m = F.normalize(video_feat_m, dim=-1)

                video_feat_m_2 = self.vision_proj(
                    torch.mean(video_embeds_m_2, dim=1))  # mean pooling --> (batch_size, hidden_dim)
                video_feat_m_2 = F.normalize(video_feat_m_2, dim=-1)

            else:
                video_feat_m = self.vision_proj(video_embeds_m[:, 0, :])  # video: (batch_size*num_frames, hidden_dim)
                video_feat_m = video_feat_m.reshape(B1, -1, video_feat_m.shape[-1])
                video_feat_m = torch.mean(video_feat_m, dim=1)  # video: (batch_size, hidden_size)
                video_feat_m = F.normalize(video_feat_m, dim=-1)
                video_embeds_m = video_embeds.reshape(B1, -1, video_embeds_m.shape[-1])

                video_feat_m_2 = self.vision_proj(
                    video_embeds_m_2[:, 0, :])  # video: (batch_size*num_frames, hidden_dim)
                video_feat_m_2 = video_feat_m_2.reshape(B2, -1, video_feat_m_2.shape[-1])
                video_feat_m_2 = torch.mean(video_feat_m_2, dim=1)  # video: (batch_size, hidden_size)
                video_feat_m_2 = F.normalize(video_feat_m_2, dim=-1)
                video_embeds_m_2 = video_embeds_m2.reshape(B2, -1, video_embeds_m_2.shape[-1])

            video_feat_all = torch.cat([video_feat_m.t(), video_feat_m_2.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                                mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_output_m_2 = self.text_encoder_m(text_2.input_ids, attention_mask=text_2.attention_mask,
                                                  return_dict=True, mode='text')
            text_feat_m_2 = F.normalize(self.text_proj_m(text_output_m_2.last_hidden_state[:, 0, :]), dim=-1)

            text_feat_all = torch.cat([text_feat_m.t(), text_feat_m_2.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = torch.cat([video_feat_m, video_feat_m_2], dim=0) @ text_feat_all / self.temp
            sim_t2i_m = torch.cat([text_feat_m, text_feat_m_2], dim=0) @ video_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(video.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = torch.cat([video_feat, video_feat_2], dim=0) @ text_feat_all / self.temp
        sim_t2i = torch.cat([text_feat, text_feat_2], dim=0) @ video_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(torch.cat([video_feat_m, video_feat_m_2], dim=0),
                                  torch.cat([text_feat_m, text_feat_m_2], dim=0))

        ###============== video-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        encoder_input_ids_2 = text_2.input_ids.clone()
        encoder_input_ids_2[:, 0] = self.tokenizer.enc_token_id

        # forward the positve video-text pair

        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=video_embeds,
                                       encoder_attention_mask=video_atts,
                                       return_dict=True,
                                       )
        output_pos_2 = self.text_encoder(encoder_input_ids_2,
                                         attention_mask=text_2.attention_mask,
                                         encoder_hidden_states=video_embeds_2,
                                         encoder_attention_mask=video_atts_2,
                                         return_dict=True,
                                         )
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :B1 + B2], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:, :B1 + B2], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

        neg_text_ids_shape1 = []
        neg_text_atts_shape1 = []
        neg_video_embs_shape1 = []
        neg_video_atts_shape1 = []
        neg_text_ids_shape2 = []
        neg_text_atts_shape2 = []
        neg_video_embs_shape2 = []
        neg_video_atts_shape2 = []

        # select a negative video for each text
        for b in range(B1 + B2):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            # video_embeds_neg.append(video_embeds[neg_idx])
            if neg_idx < B1:
                if b < B1:
                    neg_text_ids_shape1.append(encoder_input_ids[b])
                    neg_text_atts_shape1.append(text.attention_mask[b])
                else:
                    neg_text_ids_shape1.append(encoder_input_ids_2[b - B1])
                    neg_text_atts_shape1.append(text_2.attention_mask[b - B1])
                neg_video_embs_shape1.append(video_embeds[neg_idx - B1])
                neg_video_atts_shape1.append(video_atts[neg_idx - B1])
            else:
                if b < B1:
                    neg_text_ids_shape2.append(encoder_input_ids[b])
                    neg_text_atts_shape2.append(text.attention_mask[b])
                else:
                    neg_text_ids_shape2.append(encoder_input_ids_2[b - B1])
                    neg_text_atts_shape2.append(text_2.attention_mask[b - B1])
                neg_video_embs_shape2.append(video_embeds_2[neg_idx - B1])
                neg_video_atts_shape2.append(video_atts_2[neg_idx - B1])

        # video_embeds_neg = torch.stack(video_embeds_neg,dim=0)

        # select a negative text for each video
        # text_ids_neg = []
        # text_atts_neg = []
        for b in range(B1):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            if neg_idx < B1:
                neg_text_ids_shape1.append(encoder_input_ids[neg_idx])
                neg_text_atts_shape1.append(text.attention_mask[neg_idx])
            else:
                neg_text_ids_shape1.append(encoder_input_ids_2[neg_idx - B1])
                neg_text_atts_shape1.append(text_2.attention_mask[neg_idx - B1])
            neg_video_embs_shape1.append(video_embeds[b])
            neg_video_atts_shape1.append(video_atts[b])

        for b in range(B1, B1 + B2):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            if neg_idx < B1:
                neg_text_ids_shape2.append(encoder_input_ids[neg_idx])
                neg_text_atts_shape2.append(text.attention_mask[neg_idx])

            else:
                neg_text_ids_shape2.append(encoder_input_ids_2[neg_idx - B1])
                neg_text_atts_shape2.append(text_2.attention_mask[neg_idx - B1])
            neg_video_embs_shape2.append(video_embeds_2[b - B1])
            neg_video_atts_shape2.append(video_atts_2[b - B1])

        output_neg_shape1 = self.text_encoder(torch.stack(neg_text_ids_shape1, dim=0),
                                              attention_mask=torch.stack(neg_text_atts_shape1, dim=0),
                                              encoder_hidden_states=torch.stack(neg_video_embs_shape1, dim=0),
                                              encoder_attention_mask=torch.stack(neg_video_atts_shape1, dim=0),
                                              return_dict=True,
                                              )
        output_neg_shape2 = self.text_encoder(torch.stack(neg_text_ids_shape2, dim=0),
                                              attention_mask=torch.stack(neg_text_atts_shape2, dim=0),
                                              encoder_hidden_states=torch.stack(neg_video_embs_shape2, dim=0),
                                              encoder_attention_mask=torch.stack(neg_video_atts_shape2, dim=0),
                                              return_dict=True,
                                              )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_pos_2.last_hidden_state[:, 0, :],
                                   output_neg_shape1.last_hidden_state[:, 0, :],
                                   output_neg_shape2.last_hidden_state[:, 0, :]], dim=0)

        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(B1 + B2, dtype=torch.long), torch.zeros(2 * (B1 + B2), dtype=torch.long)],
                               dim=0).to(video.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= LM ========================##     
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        decoder_output = self.text_decoder(decoder_input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=video_embeds,
                                           encoder_attention_mask=video_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )

        decoder_input_ids_2 = text_2.input_ids.clone()
        decoder_input_ids_2[:, 0] = self.tokenizer.bos_token_id
        decoder_targets_2 = decoder_input_ids_2.masked_fill(decoder_input_ids_2 == self.tokenizer.pad_token_id, -100)

        decoder_output_2 = self.text_decoder(decoder_input_ids_2,
                                             attention_mask=text_2.attention_mask,
                                             encoder_hidden_states=video_embeds_2,
                                             encoder_attention_mask=video_atts_2,
                                             labels=decoder_targets_2,
                                             return_dict=True,
                                             )

        loss_lm = decoder_output.loss * B1 / (B1 + B2) + decoder_output_2.loss * B2 / (B1 + B2)
        return loss_ita, loss_itm, loss_lm


from .blip import load_checkpoint


def blip_pretrain(pretrained='', **kwargs):
    model = BLIP_Pretrain(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print(msg)
    return model


def blip_videoimage_pretrain(pretrained='', **kwargs):
    model = BLIP_VideoImage_Pretrain(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print(msg)
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


from typing import List


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            skip_key: str,
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + ' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)
