import dgl
import h5py
import os
import sys
import logging
import re
import spacy
import torch

import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utility.vocabulary import *

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from sklearn.metrics import precision_score, f1_score, recall_score

tqdm.pandas()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducible result
def init_seed(seed=1, use_cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


# Initialize file handler object
def init_log(save_dir='saved/log/', filename='log.txt', log_format='%(message)s'):
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        create_folder(save_dir)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)


def init_tensorboard(save_dir='saved/tensorboard/'):
    create_folder(save_dir)
    writer = SummaryWriter(save_dir)
    return writer


def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
def convert_batched_graph_feat(features, adj):
    graphs = []
    weights = []
    for k in range(len(adj)):
        u = []
        v = []
        w = []
        for i in range(100):
            for j in range(100):
                u.append(i)
                v.append(j)
                w.append(adj[k][i][j])
        g = dgl.graph((u, v))
        graphs.append(g)
        weights.append(torch.stack(w))

    weights = torch.stack(weights)
    weights = torch.flatten(weights)
    features = torch.flatten(features, end_dim=1)
    graphs = dgl.batch(graphs)
    return graphs, features, weights

def calculate_longest_sentence(series):
    tokenizer = spacy.load('en_core_web_sm')
    longest_sentence = 0
    for sentence in tqdm(series):
        sentence_len = 0
        for word in tokenizer(sentence):
            sentence_len += 1
        if sentence_len > longest_sentence:
            longest_sentence = sentence_len

    return longest_sentence

def generate_caption_data(data='msvd_train', n_video=5, vocab=None, device="cuda",path="dataset/MSVD/captions/sents_%s_lc_nopunc.txt",
                         min_word_count=0):
    
    if 'msvd' in data:
        used_captions = pd.read_csv(path % data.split("_")[1],\
                                    sep='\t', header=None, names=["vid_id", "caption"])
        
    # Start index for MSVD data
    start_index = {"msvd_train": 1, "msvd_val": 1201, "msvd_test": 1301} 
    
    # Create a video_id query
    chosen_keys = ["vid%s" % x for x in range(start_index[data], start_index[data]+n_video)]
    used_captions = used_captions[used_captions['vid_id'].isin(chosen_keys)]
    
    if vocab is None:
        # Instantiate new vocabulary
        vocab = Vocabulary()

        # Populate vocabulary
        print("Populating vocab with %s..." % data)
        for caption in tqdm(used_captions['caption']):
            vocab.add_sentence(caption)
            
        print("Original number of words:",vocab.num_words)
        if min_word_count>0:
            vocab.filter_vocab(min_word_count)
        print("Filtered number of words:",vocab.num_words)
        
        # Create vector caption
        print("Converting sentences to indexes...")
        used_captions['vector'] = used_captions['caption'].progress_apply(lambda x: vocab.generate_vector(x))
        longest_sentence = vocab.longest_sentence
        
    else:
        # If using val_data/test_data
        longest_sentence = calculate_longest_sentence(used_captions['caption'])
        used_captions['vector'] = used_captions['caption'].progress_apply(lambda x: vocab.generate_vector(x, longest_sentence))
    
    flatten_captions = torch.tensor(used_captions['vector']).to(device=device)
    captions_vector = used_captions.groupby("vid_id", sort=False)['vector'].sum()\
                                                                .apply(lambda x: torch.tensor(x).reshape(-1, longest_sentence+2)\
                                                                .to(device=device)).to_dict()
    
    return captions_vector, flatten_captions, vocab, used_captions

def generate_2d_3d_features(data='msvd_train', n_video=5,
                            f2d_path="MSVD-2D.hdf5", f3d_path="MSVD-3D.hdf5", device="cuda"):
    scn_2d = h5py.File(f2d_path, "r")
    scn_3d = h5py.File(f3d_path, "r")
    
    # Start index for MSVD data
    start_index = {"msvd_train": 1, "msvd_val": 1201, "msvd_test": 1301} 
    
    # Create a video_id query
    chosen_keys = ["vid%s" % x for x in range(start_index[data], start_index[data]+n_video)]
    
    scn_2d_src, scn_3d_src = [], []
    for key in chosen_keys:
        scn_2d_src.append(scn_2d.get(key))
        scn_3d_src.append(scn_3d.get(key))
        
    return torch.tensor(scn_2d_src).to(device=device), torch.tensor(scn_3d_src).to(device=device)

def generate_node_features(data="msvd_train", n_video=5,
                             fo_path="MSVD_FO_FASTERRCNN_RESNET50.hdf5",
                             stgraph_path="MSVD_IOU_STG_FASTERRCNN_RESNET50.hdf5", device="cuda", generate_fo=True):
    
    if generate_fo:
        fo_file = stack_node_features(fo_path)
    stgraph_file = h5py.File(stgraph_path, "r")

    # Start index for MSVD data
    start_index = {"msvd_train": 1, "msvd_val": 1201, "msvd_test": 1301} 
    
    # Create a video_id query
    excluded_keys = []
    for vid in fo_file.keys():
        if len(fo_file[vid]) != 100:
            excluded_keys.append(vid)
    chosen_keys = ["vid%s" % x for x in range(start_index[data], start_index[data]+n_video) if "vid%s" % x not in excluded_keys]
    
    
    fo_input, stgraph = [], []
    for key in chosen_keys:
        if generate_fo:
            fo_input.append(fo_file.get(key))
        stgraph.append(stgraph_file.get(key))
    
    if generate_fo:
        return torch.tensor(fo_input).to(device=device), torch.tensor(stgraph).to(device=device), excluded_keys
    return torch.tensor(stgraph).to(device=device)

def stack_node_features(pathfile):

    fo_input = h5py.File(pathfile, "r")
    fo_list = {}
    for i,key in tqdm(enumerate(fo_input.keys()), total=len(fo_input.keys())):
        a = key.split('-')

        if a[0] not in fo_list:
            fo_list[a[0]] = {}
        fo_list[a[0]][int(a[1])] = fo_input[key][:]

    fo_stacked = {}
    for key in fo_list.keys():
        stacked = []
        for k_fr in sorted(fo_list[key].keys()):
            stacked.append(fo_list[key][k_fr])
        fo_stacked[key] = np.vstack(stacked)
        
    return fo_stacked

def score(ref, hypo, metrics=[]):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
    
    metrics, eg. ['bleu', 'meteor','rouge_l','cider']
    """
    scorers = {
        "bleu" : (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        "meteor" : (Meteor(),"METEOR"),
        "rouge_l" : (Rouge(), "ROUGE_L"),
        "cider" : (Cider(), "CIDEr")
    }
    final_scores = {}
    for key in metrics:
        scorer, method = scorers[key]
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
        'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
        'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
        
        'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
        'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
        
        'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
        'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
        'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }