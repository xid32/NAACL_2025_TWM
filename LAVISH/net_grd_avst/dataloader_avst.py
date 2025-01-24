import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
import time
import random
from ipdb import set_trace

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision
import torchaudio
import glob
import warnings
warnings.filterwarnings('ignore')

def ids_to_multinomial(id, categories):
	""" label encoding
	Returns:
	  1d array, multimonial representation, e.g. [1,0,1,0,0,...]
	"""
	id_to_idx = {id: index for index, id in enumerate(categories)}

	return id_to_idx[id]

class AVQA_dataset(Dataset):

	def __init__(self, label, audio_dir, video_res14x14_dir, transform=None, mode_flag='train'):


		samples = json.load(open('./data/json/avqa-train.json', 'r'))

		# nax =  nne
		ques_vocab = ['<pad>']
		ans_vocab = []
		i = 0
		for sample in samples:
			i += 1
			question = sample['question_content'].rstrip().split(' ')
			question[-1] = question[-1][:-1]

			p = 0
			for pos in range(len(question)):
				if '<' in question[pos]:
					question[pos] = ast.literal_eval(sample['templ_values'])[p]
					p += 1

			for wd in question:
				if wd not in ques_vocab:
					ques_vocab.append(wd)
			if sample['anser'] not in ans_vocab:
				ans_vocab.append(sample['anser'])

		self.ques_vocab = ques_vocab
		self.ans_vocab = ans_vocab
		self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

		self.samples = json.load(open(label, 'r'))
		self.max_len = 14    # question length

		self.audio_dir = audio_dir
		self.video_res14x14_dir = video_res14x14_dir
		self.transform = transform

		video_list = []
		for sample in self.samples:
			video_name = sample['video_id']
			if video_name not in video_list:
				video_list.append(video_name)

		self.video_list = video_list

		self.video_len = 60 * len(video_list)


		self.my_normalize = Compose([
			# Resize([384,384], interpolation=Image.BICUBIC),
			Resize([192,192], interpolation=Image.BICUBIC),
			# Resize([224,224], interpolation=Image.BICUBIC),
			# CenterCrop(224),
			Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
		])

		### ---> yb calculate stats for AVQA
		self.norm_mean =  -5.385333061218262
		self.norm_std =  3.5928637981414795

		### <----
		

	def __len__(self):
		return len(self.samples)


	def _wav2fbank(self, filename, filename2=None, idx=None):
		# mixup
		if filename2 == None:
			waveform, sr = torchaudio.load(filename)
			waveform = waveform - waveform.mean()
		# mixup
		else:
			waveform1, sr = torchaudio.load(filename)
			waveform2, _ = torchaudio.load(filename2)

			waveform1 = waveform1 - waveform1.mean()
			waveform2 = waveform2 - waveform2.mean()

			if waveform1.shape[1] != waveform2.shape[1]:
				if waveform1.shape[1] > waveform2.shape[1]:
					# padding
					temp_wav = torch.zeros(1, waveform1.shape[1])
					temp_wav[0, 0:waveform2.shape[1]] = waveform2
					waveform2 = temp_wav
				else:
					# cutting
					waveform2 = waveform2[0, 0:waveform1.shape[1]]

			# sample lambda from uniform distribution
			#mix_lambda = random.random()
			# sample lambda from beta distribtion
			mix_lambda = np.random.beta(10, 10)

			mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
			waveform = mix_waveform - mix_waveform.mean()
		

		## yb: align ##
		if waveform.shape[1] > 16000*(1.95+0.1):
			sample_indx = np.linspace(0, waveform.shape[1] -16000*(1.95+0.1), num=10, dtype=int)
			waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*1.95)]

		## align end ##



		# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
		fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=10)



		# target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
		target_length = 192 ## yb: overwrite for swin

		
		########### ------> very important: audio normalized
		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
		### <--------

		# target_length = 512 ## 5s
		# target_length = 256 ## 2.5s
		n_frames = fbank.shape[0]

		p = target_length - n_frames

		# cut and pad
		if p > 0:
			m = torch.nn.ZeroPad2d((0, 0, 0, p))
			fbank = m(fbank)
		elif p < 0:
			fbank = fbank[0:target_length, :]

		if filename2 == None:
			return fbank, 0
		else:
			return fbank, mix_lambda

	def __getitem__(self, idx):
		
		sample = self.samples[idx]
		name = sample['video_id']
		# audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
		# audio = audio[::6, :]

		# visual_out_res18_path = '/home/guangyao_li/dataset/avqa-features/visual_14x14'
		

		### ---> video frame process 
		
		total_num_frames = len(glob.glob(os.path.join(self.video_res14x14_dir,'video_frames',name,'*.jpg')))
		sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
		total_img = []
		for tmp_idx in sample_indx:
			tmp_img = torchvision.io.read_image(os.path.join(self.video_res14x14_dir,'video_frames',name, str("{:04d}".format(tmp_idx))+ '.jpg'))/255
			tmp_img = self.my_normalize(tmp_img)
			total_img.append(tmp_img)
		total_img = torch.stack(total_img)
		### <---

		video_idx=self.video_list.index(name)
		visual_nega = []
		for i in range(total_img.shape[0]):
			while(1):
				neg_frame_id = random.randint(0, self.video_len - 1)
				if (int(neg_frame_id/60) != video_idx):
					break

			neg_video_id = int(neg_frame_id / 60)
			neg_frame_flag = neg_frame_id % 60
			neg_video_name = self.video_list[neg_video_id]
			
			


			total_num_frames = len(glob.glob(os.path.join(self.video_res14x14_dir,'video_frames',neg_video_name,'*.jpg')))
			sample_indx = np.linspace(1, total_num_frames , num=60, dtype=int)


			tmp_img = torchvision.io.read_image(os.path.join(self.video_res14x14_dir,'video_frames',neg_video_name, str("{:04d}".format(sample_indx[neg_frame_flag]))+ '.jpg'))/255
			visual_nega_clip = self.my_normalize(tmp_img)
			# visual_nega_out_res18=np.load(os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'))
			# visual_nega_out_res18 = torch.from_numpy(visual_nega_out_res18)

			# visual_nega_clip=visual_nega_out_res18[neg_frame_flag,:,:,:].unsqueeze(0)


			# visual_nega_clip = total_img_neg[]
			# set_trace()


			
			visual_nega.append(visual_nega_clip)
		visual_nega = torch.stack(visual_nega)

		# visual nega [60, 512, 14, 14]

		# question
		question_id = sample['question_id']
		question = sample['question_content'].rstrip().split(' ')
		question[-1] = question[-1][:-1]

		p = 0
		for pos in range(len(question)):
			if '<' in question[pos]:
				question[pos] = ast.literal_eval(sample['templ_values'])[p]
				p += 1
		if len(question) < self.max_len:
			n = self.max_len - len(question)
			for i in range(n):
				question.append('<pad>')
		idxs = [self.word_to_ix[w] for w in question]
		ques = torch.tensor(idxs, dtype=torch.long)

		# answer
		answer = sample['anser']
		label = ids_to_multinomial(answer, self.ans_vocab)
		label = torch.from_numpy(np.array(label)).long()



		### ---> loading all audio frames
		total_audio = []
		for audio_sec in range(10):
			
			fbank, mix_lambda = self._wav2fbank(os.path.join(self.video_res14x14_dir,'raw_audio',name+'.wav'), idx=audio_sec)
			total_audio.append(fbank)
		total_audio = torch.stack(total_audio)
		### <----


		sample = {'audio': total_audio, 'visual_posi': total_img, 'visual_nega': visual_nega, 'question': ques, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample

class ToTensor(object):

	def __call__(self, sample):

		audio = sample['audio']
		visual_posi = sample['visual_posi']
		visual_nega = sample['visual_nega']
		label = sample['label']


		return { 
				'audio': sample['audio'], 
				'visual_posi': sample['visual_posi'],
				'visual_nega': sample['visual_nega'],
				'question': sample['question'],
				'label': label}

		# return { 
		# 		'audio': torch.from_numpy(audio), 
		# 		'visual_posi': sample['visual_posi'],
		# 		'visual_nega': sample['visual_nega'],
		# 		'question': sample['question'],
		# 		'label': label}