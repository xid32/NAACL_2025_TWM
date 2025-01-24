import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
from einops import rearrange, repeat

class VisualAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True):
		super().__init__()
		self.adapter_kind = adapter_kind
		self.use_bn = use_bn
		self.is_multimodal = opt.is_multimodal
		self.opt = opt

		if use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // reduction_factor


			self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
			self.gate_av = nn.Parameter(torch.zeros(1))

			### <------

			self.activation = nn.ReLU(inplace=True)
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)
			
			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // reduction_factor
			self.activation = nn.ReLU(inplace=True)
			
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "basic":
			self.activation = nn.ReLU(inplace=True)
			self.conv = nn.Linear(input_dim, output_dim, bias=False)

			if use_bn:
				self.bn = nn.BatchNorm1d(output_dim)

		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			
			

			### -------> high dim att
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))


			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

			att_v2tk = F.softmax(att_v2tk, dim=-1)
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0,2,1))

			rep_token = rep_token + rep_token_res
			

			att_tk2x = torch.bmm(x.squeeze(-1).permute(0,2,1), rep_token.permute(0,2,1))

			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0,2,1).unsqueeze(-1)



			x = x + self.gate_av*x_res.contiguous()

			### <----------
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)
		
			## <----

			if self.use_bn:
				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
	
		elif self.adapter_kind == "bottleneck":
			
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
	
			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
			

		elif self.adapter_kind == "basic":
			output = self.conv(x)
			if self.use_bn:
				output = self.bn(rearrange(output, 'N C L -> N L C') )
				output = rearrange(output, 'N L C -> N C L')


		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

		if self.gate is not None:
			output = self.gate * output



	

		return output



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

	def __init__(self, opt):
		super(AVQA_Fusion_Net, self).__init__()

		self.opt = opt

		# for features
		self.fc_a1 =  nn.Linear(128, 512)
		self.fc_a2=nn.Linear(512,512)

		self.fc_a1_pure =  nn.Linear(128, 512)
		self.fc_a2_pure=nn.Linear(512,512)
		# self.visual_net = resnet18(pretrained=True)

		self.fc_v = nn.Linear(2048, 512)
		# self.fc_v = nn.Linear(1536, 512)
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

		self.yb_fc_v = nn.Linear(1536, 512)
		self.yb_fc_a = nn.Linear(1536, 512)

		# self.resnet = timm.create_model('resnet18', pretrained=True)
		self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)


		### ------------> for swin 
		hidden_list = []
		down_in_dim = []
		down_out_dim = []
		for idx_layer, my_blk in enumerate(self.swin.layers) :
			if not isinstance(my_blk.downsample, nn.Identity):
				down_in_dim.append(my_blk.downsample.reduction.in_features)
				down_out_dim.append(my_blk.downsample.reduction.out_features)

			for blk in my_blk.blocks:
				hidden_d_size = blk.norm1.normalized_shape[0]
				hidden_list.append(hidden_d_size)
		### <--------------


		self.audio_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		self.vis_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])

		self.audio_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		self.vis_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])

	def forward(self, audio, visual_posi, visual_nega, question, stage='eval'):
		'''
			input question shape:    [B, T]
			input audio shape:       [B, T, C]
			input visual_posi shape: [B, T, C, H, W]
			input visual_nega shape: [B, T, C, H, W]
		'''
		
	
		bs,t,c,h,w = visual_posi.shape



		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)
		audio = rearrange(audio, 'b t c w h -> (b t) c w h')

		###### ---------------->
		f_a = self.swin.patch_embed(audio)


		visual_posi = rearrange(visual_posi, 'b t c w h -> (b t) c w h')
		f_v = self.swin.patch_embed(visual_posi)
		# f_v_neg = self.swin.patch_embed(visual_nega)
		idx_layer = 0
		multi_scale = []
		
		idx_block = 0

		for _, my_blk in enumerate(self.swin.layers) :

			for blk in my_blk.blocks:
					
				f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path1(blk.norm1(blk._attn(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)

				

				f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path2(blk.norm2(blk.mlp(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)

				idx_layer = idx_layer +1
			#####
			f_v = my_blk.downsample(f_v)
			f_a = my_blk.downsample(f_a)

		f_v = self.swin.norm(f_v)
		f_a = self.swin.norm(f_a)

		f_v = self.yb_fc_v(f_v)
		f_a = self.yb_fc_a(f_a)
			

		with torch.no_grad():

			visual_nega = rearrange(visual_nega, 'b t c h w -> (b t) c h w')
			visual_nega = self.swin.forward_features(visual_nega)


		############## <----------

		visual_nega = self.yb_fc_v(visual_nega)
		visual_posi = rearrange(f_v, '(b t) (h w) c -> b t c h w', b=bs ,t=t, h=6 ,w=6)
		visual_nega = rearrange(visual_nega, '(b t) (h w) c -> b t c h w', b=bs ,t=t, h=6 ,w=6)


		### -------> yb: cvpr use
		f_a = f_a.mean(dim=1)
		audio = rearrange(f_a, '(b t) c -> b t c', b=bs ,t=t)
		### <-----

		## question features
		qst_feature = self.question_encoder(question)
		xq = qst_feature.unsqueeze(0)

		## audio features  [2*B*T, 128]
		# audio_feat = F.relu(self.fc_a1(audio))
		audio_feat = F.relu(audio) # yb: for swin only
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
