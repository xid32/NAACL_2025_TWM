import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18


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

# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class AVQA_Fusion_Net(nn.Module):

    def __init__(self):
        super(AVQA_Fusion_Net, self).__init__()

        # for features
        self.fc_a1 =  nn.Linear(128, 512)
        self.fc_a2=nn.Linear(512,512)

        self.fc_a1_pure =  nn.Linear(128, 512)
        self.fc_a2_pure=nn.Linear(512,512)
        self.visual_net = resnet18(pretrained=True)

        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.fc = nn.Linear(1024, 512)
        self.fc_aq = nn.Linear(512, 512)
        self.fc_vq = nn.Linear(512, 512)

        self.linear11 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(512, 512)

        self.linear21 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(512, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(512)

        self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)

        # question
        self.question_encoder = QstEncoder(93, 512, 512, 1, 512)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(512, 42)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl=nn.Linear(1024,512)

        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()


    def forward(self, audio, visual_posi, visual_nega, question):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''

        ## question features
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        ## audio features  [2*B*T, 128]
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)  
        audio_feat_pure = audio_feat
        B, T, C = audio_feat.size()             # [B, T, C]
        audio_feat = audio_feat.view(B*T, C)    # [B*T, C]

        ## visual posi [2*B*T, C, H, W]
        B, T, C, H, W = visual_posi.size()
        temp_visual = visual_posi.view(B*T, C, H, W)            # [B*T, C, H, W]
        v_feat = self.avgpool(temp_visual)                      # [B*T, C, 1, 1]
        visual_feat_before_grounding_posi = v_feat.squeeze()    # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)                      # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)                            # [B, HxW, C]
        visual_feat_posi = nn.functional.normalize(v_feat, dim=2)   # [B, HxW, C]

        ## audio-visual grounding posi
        audio_feat_aa = audio_feat.unsqueeze(-1)                        # [B*T, C, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)   # [B*T, C, 1]
        x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze() # [B*T, HxW]

        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_posi)
        visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()    # [B*T, C]   

        visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi),dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*T, C]

        feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*T, C*2], [B*T, 1024]

        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_posi = self.fc4(feat)     # (128, 2)

        ###############################################################################################
        # visual nega
        B, T, C, H, W = visual_nega.size()
        temp_visual = visual_nega.view(B*T, C, H, W)
        v_feat = self.avgpool(temp_visual)
        visual_feat_before_grounding_nega = v_feat.squeeze() # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)        # [B, HxW, C]
        visual_feat_nega = nn.functional.normalize(v_feat, dim=2)

        ##### av grounding nega
        x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_nega)
        visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()    # [B*T, C]   

        visual_gl=torch.cat((visual_feat_before_grounding_nega,visual_feat_grd_after_grounding_nega),dim=-1)
        visual_feat_grd=self.tanh(visual_gl)
        visual_feat_grd_nega=self.fc_gl(visual_feat_grd)    # [B*T, C]

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)   # [B*T, C*2], [B*T, 1024]

        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_nega = self.fc4(feat)     # (128, 2)

        ###############################################################################################

        # out_match=None
        # match_label=None

        B = xq.shape[1]
        visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 512)   # [B, T, 512]
        visual_feat_grd=visual_feat_grd_be.permute(1,0,2)
        
        ## attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)
    
        # attention, question as query on audio
        audio_feat_be=audio_feat_pure.view(B, -1, 512)
        audio_feat = audio_feat_be.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None,key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)
        
        feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(), visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        ## fusion with question
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)              # [batch_size, ans_vocab_size]

        return out_qa, out_match_posi,out_match_nega
