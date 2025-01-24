from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.blip import create_vit, init_tokenizer, load_checkpoint
import testa


class TESTA_Retrieval(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                 negative_all_rank=False,
                 token_merging=False,
                 testa_r=0,
                 merging_type=None,
                 model_cfg=None,
                 max_words=128
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0, model_cfg)
        self.timesformer = False
        if 'timesformer' in vit:
            self.latent_feat_size = vision_width
            self.timesformer = True
        self.num_frames = model_cfg["num_frames"]

        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.max_words = max_words

        # print("Turning off gradients in the text encoder")
        # for name, param in self.text_encoder.named_parameters():
        #     param.requires_grad_(False)
        # print("Turning off gradients in the visual encoder")
        # for name, param in self.visual_encoder.named_parameters():
        #     param.requires_grad_(False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0, model_cfg)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.token_merging = token_merging
        self.merging_type = merging_type
        if token_merging:
            if self.timesformer:
                testa.patch.timesformer(self.visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,  # todo check trace_source: frame in merging_type?
                                        merging_type=merging_type, num_patches=self.visual_encoder.num_patches)
                testa.patch.timesformer(self.visual_encoder_m, trace_source=(merging_type == 'frame'), prop_attn=False,
                                        merging_type=merging_type, num_patches=self.visual_encoder_m.num_patches)
            else:
                testa.patch.vit(self.visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,
                                merging_type=merging_type)
                testa.patch.vit(self.visual_encoder_m, trace_source=(merging_type == 'frame'), prop_attn=False,
                                merging_type=merging_type)
            self.visual_encoder.r = testa_r
            self.visual_encoder_m.r = testa_r

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.negative_all_rank = negative_all_rank

    def forward(self, video, caption, alpha, idx, bsz=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        video_embeds = self.visual_encoder(video)
        video_feat = self.vision_proj(video_embeds)  # mean pooling --> (batch_size, hidden_dim)
        video_feat = F.normalize(video_feat, dim=-1)


        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words,
                              return_tensors="pt").to(video.device)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return video_feat, text_feat

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
    def _dequeue_and_enqueue(self, video_feat, text_feat, idxs):
        # gather keys before updating queue
        video_feats = concat_all_gather(video_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = video_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = video_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.ptr_queue[0] = ptr


def testa_retrieval(pretrained='', **kwargs):
    model = TESTA_Retrieval(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
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


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
