import logging

import torch
import torch.nn.functional as F
from einops import rearrange

from .utils import tile
from .vindlu import VindLU

logger = logging.getLogger(__name__)


class VindLU_TVQA(VindLU):
    """docstring for VindLU_TVQA"""

    def __init__(self, config, tokenizer, is_pretrain=False):
        super().__init__(config, tokenizer, is_pretrain)

    def forward(self, image, text, answer, train=True):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (TODO): tokenized text. 5*B. Each image has 5 text.
            answer (torch.Tensor): The answers. Shape: [B,]. Each value
                is between 0 and 4.

        """
        # ================= Dual Encoder ITC loss ================ #
        bsz = len(image)
        num_options_per_q = 5

        image_embeds, pooled_image_embeds = self.encode_vision(image)  # (N, )
        text_embeds, pooled_text_embeds = self.encode_text(text)  # (5N, )
        image_embeds = rearrange(image_embeds, "b t l c -> b (t l) c")

        # ================= Cross Encoder ITM loss ================ #
        image_embeds = tile(image_embeds, 0, num_options_per_q)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device, non_blocking=True
        )

        output = self.get_text_encoder()(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )
        itm_embeds = output.last_hidden_state[:, 0]  # [CLS]  (5N, )

        score = self.itm_head(itm_embeds)[:, 1]  # (5N, )
        score = score.view(-1, num_options_per_q)  # (N, 5)
        if train:
            loss_qa = F.cross_entropy(score, answer)

            return_dict = dict(loss_qa=loss_qa)

            return return_dict
        else:
            pred_ans = score.max(1)[1].cpu()
            return pred_ans
