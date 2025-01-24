import torch
from modules.modeling import CaptionGenerator
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch import nn
from torch.nn import KLDivLoss
from modules.gnn.TransformerConvolution import TransC


class GraphTransformer(nn.Module):
	"""Implementation of the graph model using Transformer head.
		Params:
			caption_generator_state_dict: model's saved state.
			caption_generator_cache_dir: cache directory location.
			args: args variable from main script.
	"""
	def __init__(self, caption_generator_state_dict=None, caption_generator_cache_dir=None, args=None):
		super().__init__()
		self.transc = TransC(node_feat_dim=args.node_feat_dim, d_model=args.d_model, edge_dim=args.edge_dim,
								project_edge_dim=args.project_edge_dim, more_skip=args.no_skip==False, last_average=args.last_average, beta=args.no_beta_transformer==False)
		self.caption_generator_model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                   cache_dir=caption_generator_cache_dir, state_dict=caption_generator_state_dict, task_config=args)
		self.avgPool = nn.AvgPool2d((args.num_object,1)) # number of patches or objects
		self.kl_loss_fct = KLDivLoss(reduction='batchmean')
		self.lp = nn.Linear(1024, 512)

	def forward(self, geo_graph, video_mask=None,
                input_caption_ids=None, decoder_mask=None, batch_size=None, n_node=None):
		fo_convolved = self.transc(geo_graph)
		
		fo_convolved = fo_convolved.unflatten(0, (batch_size,n_node))
		fo_convolved = self.avgPool(fo_convolved)

		decoder_scores = self.caption_generator_model(fo_convolved, video_mask, input_caption_ids=input_caption_ids, decoder_mask=decoder_mask)
		return decoder_scores

	def get_visual_output(self, geo_graph, video_mask=None, batch_size=None, n_node=None, action=None):
		fo_convolved = self.transc(geo_graph)

		fo_convolved = fo_convolved.unflatten(0, (batch_size, n_node))
		fo_convolved = self.avgPool(fo_convolved)

		visual_output = self.caption_generator_model.get_visual_output(fo_convolved, video_mask)

		return visual_output
