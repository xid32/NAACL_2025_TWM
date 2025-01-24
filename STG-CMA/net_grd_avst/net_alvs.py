import torch

import torch.nn as nn
import torch.nn.functional as F



import timm
from einops import rearrange, repeat
import os
from net_encoders import AVQA_Fusion_Net, AMS



# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)  # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

        return qst_feature


class AVQA_Net(nn.Module):

    def __init__(self, opt, mode="visual"):
        super(AVQA_Net, self).__init__()

        self.opt = opt
        self.mode = mode
        self.visual_encoder = AVQA_Fusion_Net(opt)


        # ===================================== load pretrained model ===============================================
        ####### concat model
        pretrained_file = "grounding_gen/models_grounding_gen/lavish_grounding_gen_best.pt"
        checkpoint = torch.load(pretrained_file)
        print("\n-------------- loading pretrained models --------------")
        model_dict = self.visual_encoder.state_dict()
        tmp = ['module.fc_a1.weight', 'module.fc_a1.bias', 'module.fc_a2.weight', 'module.fc_a2.bias',
               'module.fc_gl.weight', 'module.fc_gl.bias', 'module.fc1.weight', 'module.fc1.bias', 'module.fc2.weight',
               'module.fc2.bias', 'module.fc3.weight', 'module.fc3.bias', 'module.fc4.weight', 'module.fc4.bias']
        tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias', 'module.fc_a2.weight', 'module.fc_a2.bias']
        pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
        pretrained_dict2 = {str(k).split('.')[0] + '.' + str(k).split('.')[1] + '_pure.' + str(k).split('.')[-1]: v for
                            k, v in checkpoint.items() if k in tmp2}

        model_dict.update(pretrained_dict1)  # 利用预训练模型的参数，更新模型
        model_dict.update(pretrained_dict2)  # 利用预训练模型的参数，更新模型
        self.visual_encoder.load_state_dict(model_dict)

        print("\n-------------- load pretrained models --------------")

        # ===================================== load pretrained model ===============================================




        self.visual_encoder.frozen_model()
        self.audio_encoder = AMS(1024, 1536, 8, "cuda")
        self.visual_mapping = nn.Linear(1536, 1536)

    def frozen_model(self):
        for p in self.parameters():
            p.requires_grad = False
        return


    def forward(self, audio, visual_posi, question):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''

        bs, t, c, h, w = visual_posi.shape

        audio_re = audio.view(audio.size(0) * audio.size(1) * 10, 32000)
        bs = visual_posi.size(0)
        f_v, _ = self.visual_encoder.visual_audio_feature_forward(audio, visual_posi)
        f_v = f_v.squeeze()
        f_v = self.visual_mapping(f_v)
        if self.mode == "visual":
            f_qst_ = self.visual_encoder.qst_feature_forward(question)
            rep_f_qst = repeat(f_qst_, 'b d -> b t d', t=t)
            f_qst = rearrange(rep_f_qst, 'b t d -> (b t) d')
            f_a = None
        else:
            f_a, _ = self.audio_encoder(audio_re, f_v)
            f_qst = None

        return f_v, f_a, f_qst


    def forward_visual(self, audio, visual_posi, question, mixup_lambda, stage='eval'):
        audio, visual_posi, question = audio.unsqueeze(0), visual_posi.unsqueeze(0), question.unsqueeze(0)
        bs, t, c, h, w = visual_posi.shape

        audio_re = audio.view(audio.size(0) * audio.size(1) * 10, 32000)
        bs = visual_posi.size(0)
        f_v, _ = self.visual_encoder.visual_audio_feature_forward(audio, visual_posi)
        f_v = f_v.squeeze()
        f_v = self.visual_mapping(f_v)
        if self.mode == "visual":
            f_qst_ = self.visual_encoder.qst_feature_forward(question)
            rep_f_qst = repeat(f_qst_, 'b d -> b t d', t=t)
            f_qst = rearrange(rep_f_qst, 'b t d -> (b t) d')
            f_a = None
        else:
            f_a, _ = self.audio_encoder(audio_re, f_v)
            f_qst = None
        return f_v, f_a, f_qst
