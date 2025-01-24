#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
# from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument(
			"--audio_dir", type=str, default='/home/guangyao_li/dataset/avqa-features/feats/vggish', help="audio dir")
		# parser.add_argument(
		#     "--video_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-1fps', help="video dir")
		self.parser.add_argument(
			"--video_res14x14_dir", type=str, default='/home/guangyao_li/dataset/avqa-features/visual_14x14', help="res14x14 dir")
		
		self.parser.add_argument(
			"--label_train", type=str, default="data/AVQA/json/avqa-train.json", help="train csv file")
		self.parser.add_argument(
			"--label_val", type=str, default="data/AVQA/json/avqa-val.json", help="val csv file")
		self.parser.add_argument(
			"--label_test", type=str, default="data/AVQA/json/avqa-test.json", help="test csv file")
		self.parser.add_argument(
			'--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 16)')
		self.parser.add_argument(
			'--epochs', type=int, default=12, metavar='N', help='number of epochs to train (default: 60)')
		self.parser.add_argument(
			'--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
		self.parser.add_argument(
			"--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
		self.parser.add_argument(
			"--mode", type=str, default='train', help="with mode to use")
		self.parser.add_argument(
			'--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
		self.parser.add_argument(
			'--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
		self.parser.add_argument(
			"--model_save_dir", type=str, default='net_grd_avst/avst_models/', help="model save dir")
		self.parser.add_argument(
			"--checkpoint", type=str, default='alst', help="save model name")
		self.parser.add_argument(
			'--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu device number')

		### for AV-ada ###
		self.parser.add_argument(
			'--wandb', type=int, default=0, help='weight and bias setup')

		
		self.parser.add_argument('--audio_length', type=float, default= 1, help='audio length')
		self.parser.add_argument('--num_workers', type=int, default= 8, help='worker for dataloader')
		self.parser.add_argument('--model_name', type=str, default=None, help="for log")



		self.parser.add_argument('--adapter_kind', type=str, default='bottleneck', help="for log")

		self.parser.add_argument('--Adapter_downsample', type=int, default=16, help="tune top k")


		self.parser.add_argument('--num_conv_group', type=int, default=4, help="group conv")


		self.parser.add_argument('--is_audio_adapter_p1', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_audio_adapter_p2', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_audio_adapter_p3', type=int, default=0, help="TF audio adapter")

		self.parser.add_argument('--is_bn', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_gate', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_multimodal', type=int, default=1, help="TF audio adapter")
		self.parser.add_argument('--is_before_layernorm', type=int, default=1, help="TF audio adapter")
		self.parser.add_argument('--is_post_layernorm', type=int, default=1, help="TF audio adapter")

		self.parser.add_argument('--is_vit_ln', type=int, default=0, help="TF Vit")

		self.parser.add_argument('--is_fusion_before', type=int, default=0, help="TF Vit")

		self.parser.add_argument('--num_tokens', type=int, default=32, help="num of MBT tokens")

		self.parser.add_argument(
			'--early_stop', type=int, default=3, help='weight and bias setup')

		self.parser.add_argument(
			'--lr_block', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')

		self.parser.add_argument(
			'--tmp_av', type=float, default=0.1, help='tmp for nce')
		self.parser.add_argument(
			'--tmp_tv', type=float, default=0.1, help='tmp for nce')

		self.parser.add_argument(
			'--coff_av', type=float, default=0.5, help='tmp for nce')
		self.parser.add_argument(
			'--coff_tv', type=float, default=0.5, help='tmp for nce')
		self.parser.add_argument('--backbone_type', type=str, help="the backbone of htsat")
		self.parser.add_argument('--root_path', type=str)
		
		

		

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		str_ids = self.opt.gpu.split(',')
		self.opt.gpu = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu.append(id)

		# # set gpu ids
		# if len(self.opt.gpu_ids) > 0:
		# 	torch.cuda.set_device(self.opt.gpu_ids[0])


		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		# expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		# util.mkdirs(expr_dir)
		# file_name = os.path.join(expr_dir, 'opt.txt')
		# with open(file_name, 'wt') as opt_file:
		# 	opt_file.write('------------ Options -------------\n')
		# 	for k, v in sorted(args.items()):
		# 		opt_file.write('%s: %s\n' % (str(k), str(v)))
		# 	opt_file.write('-------------- End ----------------\n')
		return self.opt
