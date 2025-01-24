from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
from nlgeval import NLGEval
import time
import argparse
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CaptionGenerator
from modules.graph_gat_modelling import GraphGAT
from modules.graph_gatv2_modelling import GraphGATv2
from modules.graph_transformer_modelling import GraphTransformer
from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
from feature_extractor.util import get_logger
import dgl
from torch_geometric.data import Data
from tqdm import tqdm
from dataloaders.dataloader_msvd_caption import MSVD_Caption_DataLoader
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CaptionGenerator'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--data_path', type=str, default='data/MSRVTT_data.json',
                        help='caption and transcription file path')
    parser.add_argument('--features_path', type=str, default='data/msrvtt_videos_feature.pickle',
                        help='feature path for CLIP features')
    parser.add_argument('--fo_path', type=str, default='data/fo.hdf5',
                        help='feature node path from can be either from object or grid features. This is not required for training our method.')
    parser.add_argument('--stgraph_path', type=str, default='data/stgraph.hdf5',
                        help='spatio-temporal graph path. This is not required for training our method.')
    parser.add_argument('--node_features', type=str, default='',
                        help='type of graph features, dgl or geometric')
    parser.add_argument('--data_geometric_path', type=str, default='',
                        help='geometric data path')
    parser.add_argument('--use_geometric_hdf5', action='store_true', help='use hdf5 version geometric')

    parser.add_argument('--gnn_model_type', type=str, default='gat',
                        help='gat, gatv2, or transformer')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-model", type=str, required=True, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset `msrvtt` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")
    parser.add_argument('--num_object', type=int, default=9, help="number of object / node")
    parser.add_argument('--node_feat_dim', type=int, default=2048, help="dim of node features.")
    parser.add_argument('--d_model', type=int, default=512, help="dim of gcn model.")
    parser.add_argument('--tradeoff_theta_2', type=float, default=1.0, help='theta_2 loss weight')
    parser.add_argument('--tradeoff_distill', type=float, default=4.0, help='distill loss weight')

    parser.add_argument('--edge_dim', type=int, default=1024, help="dim of edge features.")

    parser.add_argument('--no_skip', action='store_true', help="Whether to remove the additional skip connection in the gnn model")
    parser.add_argument('--last_average', action='store_true', help="Whether to use average or concatenation in the last layer of gnn model")
    parser.add_argument('--no_beta_transformer', action='store_true', help="Whether to not use beta on the transformer model")

    parser.add_argument('--patience', type=int, default=10, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--patience_metric', type=str, default="CIDEr", help="Metric which is used for early stopping.")
    parser.add_argument('--target_metric', type=str, default="CIDEr", help="Target metric which is used to select the best model.")
    parser.add_argument('--project_edge_dim', type=int, default=None, help="projection of edge dim")

    args = parser.parse_args()

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    # n_gpu = torch.cuda.device_count()
    n_gpu = torch.distributed.get_world_size()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model_theta_2(args, device, n_gpu, local_rank):
    """
        weight initialization of theta_2 network
    """

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def init_model_theta_1(args, device, n_gpu, local_rank):
    """
        weight initialization of theta_1 network
    """
    if args.init_model:
        caption_generator_model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        caption_generator_model_state_dict = None


    caption_generator_cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    
    # Prepare model
    if args.gnn_model_type=='transformer':
        model = GraphTransformer(caption_generator_state_dict=caption_generator_model_state_dict, caption_generator_cache_dir=caption_generator_cache_dir, args=args)
    elif args.gnn_model_type=='gatv2':
    	model = GraphGATv2(caption_generator_state_dict=caption_generator_model_state_dict, caption_generator_cache_dir=caption_generator_cache_dir, args=args)
    else: # gat
        model = GraphGAT(caption_generator_state_dict=caption_generator_model_state_dict, caption_generator_cache_dir=caption_generator_cache_dir, args=args)
    model.to(device)

    return model

def prep_optimizer(args, theta_2_model, theta_1_model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(theta_2_model, 'module'):
        theta_2_model = theta_2_model.module

    if hasattr(theta_1_model, 'module'):
        theta_1_model = theta_1_model.module

    param_optimizer = list(theta_2_model.named_parameters()) + list(theta_1_model.named_parameters()) 
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    theta_2_model = torch.nn.parallel.DistributedDataParallel(theta_2_model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    theta_1_model = torch.nn.parallel.DistributedDataParallel(theta_1_model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, theta_2_model, theta_1_model

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Caption_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        data_geometric_path=args.data_geometric_path,
        max_words=args.max_words,
        use_geometric_hdf5=args.use_geometric_hdf5,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        node_features=args.node_features,
        inference_path = None
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_val_test(args, tokenizer, split_type="test",):
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        # fo_path=args.fo_path,
        # stgraph_path=args.stgraph_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        inference_path = None
        # node_features=args.node_features
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)
def dataloader_msvd_val_test(args, tokenizer, split_type="val",):
    msvd = MSVD_Caption_DataLoader(
        data_path=args.data_path,

        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        inference_path = None
    )

    sampler = SequentialSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        sampler=sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msvd, len(msvd)
def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def dataloader_msvd(args, tokenizer, split_type="train",):
    msvd = MSVD_Caption_DataLoader(
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        node_features=args.node_features,
        data_geometric_path=args.data_geometric_path,
        feature_framerate=args.feature_framerate,
        inference_path = None
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        sampler=train_sampler,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        drop_last=True,)
    return dataloader_msvd, len(msvd), train_sampler

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores
    
def train_epoch(epoch, args, theta_2_model, theta_1_model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0):
    """
        For training theta_2 and theta_1 model simultaneously.
    """
    global logger
    torch.cuda.empty_cache()
    theta_2_model.train()
    theta_1_model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    offset = []
    for step, batch in enumerate(train_dataloader):
        # if n_gpu == 1:
        #     # multi-gpu does scattering it-self
        #     batch = tuple(t.to(device) for t in batch)
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, geo_x, geo_edge_index, geo_edge_attr = batch

        batch_sz, n_node, _ = geo_x.shape

        x_batch = geo_x.reshape(geo_x.shape[0]*geo_x.shape[1], geo_x.shape[2])
        
        # this is to convert the edge index in batch format, as required by the geometric library
        # please check how the dataloader of geometric returns the batched data
        if len(offset)==0:
            for i in range(batch_sz):
                offset.append(torch.full(geo_edge_index[0].shape, int(np.sqrt(geo_edge_index[0].shape[1]))*i))
            offset = torch.stack(offset)
            offset = offset.to(device)
        
        geo_graph_batch_offset = geo_edge_index + offset
        new_dim = geo_graph_batch_offset.shape[0]*geo_graph_batch_offset.shape[2]
        edge_index_batch = geo_graph_batch_offset.permute(1,0,2).reshape(2, new_dim)
        

        edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0]*geo_edge_attr.shape[1], geo_edge_attr.shape[2]).float()
        
        data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
        data_geo_graph_batch = data_geo_graph_batch.to(device)
        

        # pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,
        # input_mask, segment_ids
        # the above data are unnecessary to train clip4caption

        decoder_scores_theta_2 = theta_2_model(video, video_mask,
                    input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask)

        decoder_scores_theta_1 = theta_1_model(data_geo_graph_batch, video_mask,
                    input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask, batch_size=batch_sz, n_node=n_node)

        pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])

        loss_theta_2 = theta_2_model.module.decoder_loss_fct(decoder_scores_theta_2.view(-1, theta_2_model.module.bert_config.vocab_size), pairs_output_caption_ids.view(-1))

        loss_theta_1 = theta_1_model.module.caption_generator_model.decoder_loss_fct(decoder_scores_theta_1.view(-1, theta_1_model.module.caption_generator_model.bert_config.vocab_size), pairs_output_caption_ids.view(-1))


                
        logprob_theta_1 = F.log_softmax(decoder_scores_theta_1, 2)
        prob_theta_2 = F.softmax(decoder_scores_theta_2, 2)

        # logprob_theta_1 = F.log_softmax(decoder_scores_theta_1, 2)
        # prob_theta_2 = F.softmax(decoder_scores_theta_2, 2)

        # reshape
        # logprob_theta_1 = logprob_theta_1.reshape(-1, logprob_theta_1.shape[2])
        # prob_theta_2 = prob_theta_2.reshape(-1, prob_theta_2.shape[2])

        # Calculate distillation loss
        loss_distill = theta_1_model.module.kl_loss_fct(logprob_theta_1, prob_theta_2)

        if n_gpu > 1:
            loss_theta_2 = loss_theta_2.mean()  # mean() to average on multi-gpu.
            loss_theta_1 = loss_theta_1.mean()  # mean() to average on multi-gpu.
            loss_distill = loss_distill.mean()

        if args.gradient_accumulation_steps > 1:
            loss_theta_2 = loss_theta_2 / args.gradient_accumulation_steps
            loss_theta_1 = loss_theta_1 / args.gradient_accumulation_steps
            loss_distill = loss_distill / args.gradient_accumulation_steps

        loss = (args.tradeoff_theta_2*loss_theta_2) + loss_theta_1 + (args.tradeoff_distill*loss_distill)

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(theta_2_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(theta_1_model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, WD: %f, WS: %f, Loss: %f, Theta 2 Loss: %f, Theta 1 Loss: %f, Distill loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(args.tradeoff_distill),float(args.tradeoff_theta_2),
                            float(loss), float(loss_theta_2), float(loss_theta_1), float(loss_distill),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor


def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)
    visual_output_rpt, video_mask_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return (active_visual_output_rpt, active_video_mask_rpt), \
           active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples):
        visual_output_rpt, video_mask_rpt = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        dec_output = decoder(visual_output_rpt, video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
# >----------------------------------------

def eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None):
    """
        to evaluate the theta_2 model. Only theta_2 model is needed for inference. the sentecence is generated via beam search
    """
    if hasattr(model, 'module'):
        model = model.module.to(device)

    all_result_lists = []
    all_caption_lists = []
    model.eval()
    for batch in tqdm(test_dataloader, desc='validation'):
        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch


        with torch.no_grad():
            visual_output = model.get_visual_output(video, video_mask)

            # -- Repeat data for beam search
            n_bm = 5 # beam_size
            device = visual_output.device
            n_inst, len_v, v_h = visual_output.size()

            decoder = model.decoder_caption

            # Note: shaped first, then decoder need the parameter shaped=True
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # -- Decode
            for len_dec_seq in range(1, args.max_words + 1):
                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (visual_output_rpt, video_mask_rpt))

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (visual_output_rpt, video_mask_rpt), \
                inst_idx_to_position_map = collate_active_info((visual_output_rpt, video_mask_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            for re_idx, re_list in enumerate(result_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                all_result_lists.append(decode_text)

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                all_caption_lists.append(decode_text)

    # Save full results
    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")
        with open(hyp_path, "w", encoding='utf-8') as writer:
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists):
                video_id, sub_id = test_set.iter2video_pairs_dict[idx]
                start_time = test_set.data_dict[video_id]['start'][sub_id]
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    # Save pure results
    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    if args.datatype == "msrvtt"or args.datatype == "msvd":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        video_ids = set()
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            video_ids.add(video_id)
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        if args.datatype != "msvd": # if number of caption for each video is different, use this
            all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    # Evaluate
    if args.datatype == "msvd":
        all_result_dict = {}
        all_caption_dict = {}
        for i in range(len(all_result_lists)):
            all_result_dict[i] = [all_result_lists[i]]
        for i in range(len(all_caption_lists)):
            all_caption_dict[i]=all_caption_lists[i]
        # Evaluate
        metrics_nlg = score(all_caption_dict,all_result_dict)
    else:
        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=all_result_lists)
    logger.info(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    logger.info(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))

    return metrics_nlg

DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_val_test, "test":dataloader_msrvtt_val_test}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd, "val":dataloader_msvd_val_test, "test":dataloader_msvd_val_test}

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    theta_2_model = init_model_theta_2(args, device, n_gpu, args.local_rank)
    theta_1_model = init_model_theta_1(args, device,n_gpu, args.local_rank)

    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)

    assert args.datatype in DATALOADER_DICT
    val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, "val")
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer, "test")
    if args.local_rank == 0:
        logger.info("***** Running val *****") 
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))
        logger.info("***** Running test *****") 
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, theta_2_model, theta_1_model = prep_optimizer(args, theta_2_model, theta_1_model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = {"CIDEr": 0.00001}
        best_output_model_file = {"CIDEr": None}
        assert args.target_metric in best_score.keys()
        assert args.patience_metric in best_score.keys()
        global_step = 0
        stop_signal = torch.zeros(2).cuda() # index 0 for early stopping based on num of epoch, index 1 for low metrics
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, theta_2_model, theta_1_model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, theta_2_model, type_name="")
                if epoch > 0:
                    metric_scores = eval_epoch(args, theta_2_model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

                    for met in best_score.keys():
                        if met == args.target_metric:
                            if metric_scores[met] <= 0.001:
                                logger.warning("One of the metrics is less than 0.001. The training will be stopped.")
                                stop_signal[1] = 1
                                break
                            if best_score[met] <= metric_scores[met]:
                                best_score[met] = metric_scores[met]
                                best_output_model_file[met] = output_model_file
                                if met==args.patience_metric:
                                    stop_signal[0] = 0
                            else:
                                if met==args.patience_metric:
                                    stop_signal[0] += 1
                            logger.info("The best model based on {} is: {}, the {} is: {:.4f}".format(met, best_output_model_file[met], met, best_score[met]))

                    for gp in range(1, args.n_gpu): # TODO, try to use broadcast function of pytorch
                        torch.distributed.send(stop_signal, dst=gp)

                    if stop_signal[0]>=args.patience:
                        logger.warning("Early stopping, no improvement after {} epochs at local rank {}".format(args.patience, args.local_rank))
                        break
                    if stop_signal[1] == 1:
                        break
                else:
                    for gp in range(1, args.n_gpu): # TODO, try to use broadcast function of pytorch
                        torch.distributed.send(stop_signal, dst=gp)
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))
            else:
                torch.distributed.recv(stop_signal, src=0)
                if stop_signal[0]>=args.patience:
                    logger.warning("Early stopping, no improvement after {} epochs at local rank {}".format(args.patience, args.local_rank))
                    break
                elif stop_signal[1]==1:
                    logger.warning("One of the metrics is less than 0.001")
                    break

        if args.local_rank == 0:
            test_scores = {}
            for met in best_score.keys():
                if met == args.target_metric:
                    theta_2_model = None
                    theta_2_model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file[met])
                    
                    metric_scores = eval_epoch(args, theta_2_model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    test_scores[met] = metric_scores
            for met in test_scores.keys():
                logger.info("Test score based on the best {} model ({}) : {}".format(met, best_output_model_file[met], str(test_scores[met])))
    elif args.do_eval:
       
        eval_epoch(args, theta_2_model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

if __name__ == "__main__":
    main()
