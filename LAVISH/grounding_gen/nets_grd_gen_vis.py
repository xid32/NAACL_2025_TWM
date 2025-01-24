import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from visual_net import resnet18


class AVQA_AVatt_Grounding(nn.Module):

    def __init__(self):
        super(AVQA_AVatt_Grounding, self).__init__()

        # for features
        self.fc_a1 =  nn.Linear(128, 512)
        self.fc_a2=nn.Linear(512,512)

        # visual
        self.visual_net = resnet18(pretrained=True)

        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_gl=nn.Linear(1024,512)
        self.tanh = nn.Tanh()


    def forward(self, video_id, audio, visual):

        ## audio features
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat=self.fc_a2(audio_feat)                      # [16, 20, 512]
        (B, T, C) = audio_feat.size()
        audio_feat = audio_feat.view(B*T, C)                # [320, 512]

        ## visual, input: [16, 20, 3, 224, 224]
        (B, T, C, H, W) = visual.size()
        visual = visual.view(B * T, C, H, W)                # [320, 3, 224, 224]

        v_feat_out_res18 = self.visual_net(visual)                    # [320, 512, 14, 14]
        v_feat=self.avgpool(v_feat_out_res18)
        visual_feat_before_grounding=v_feat.squeeze()     # 320 512
        
        (B, C, H, W) = v_feat_out_res18.size()
        v_feat = v_feat_out_res18.view(B, C, H * W)
        v_feat = v_feat.permute(0, 2, 1)  # B, HxW, C
        visual = nn.functional.normalize(v_feat, dim=2)
             
        ## audio-visual grounding
        audio_feat_aa = audio_feat.unsqueeze(-1)            # [320, 512, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)
        visual_feat = visual
        x2_va = torch.matmul(visual_feat, audio_feat_aa).squeeze()

        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)       # [320, 1, 196]
        visual_feat_grd = torch.matmul(x2_p, visual_feat)
        visual_feat_grd = visual_feat_grd.squeeze()         # [320, 512]   

        visual_gl=torch.cat((visual_feat_before_grounding,visual_feat_grd),dim=-1)
        visual_feat_grd=self.tanh(visual_gl)
        visual_feat_grd=self.fc_gl(visual_feat_grd)

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd), dim=-1)     # [320, 1024]

        feat = F.relu(self.fc1(feat))   # (1024, 512)
        feat = F.relu(self.fc2(feat))   # (512, 256)
        feat = F.relu(self.fc3(feat))   # (256, 128)
        feat = self.fc4(feat)   # (128, 2)

        return  x2_p, feat
