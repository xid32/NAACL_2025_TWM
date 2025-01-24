import copy
import logging
import os.path as osp

import torch
from torch.utils.data import ConcatDataset, DataLoader

from models.backbones.beit.builder import interpolate_pos_embed_beit
from models.backbones.bert.tokenization_bert import BertTokenizer
from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, has_decoder=False, pretrain=False, find_unused_parameters=False
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    else:
        raise ValueError(f"Not supported text encoder.")

    model = model_cls(config=config, tokenizer=tokenizer, is_pretrain=pretrain)

    model = model.to(torch.device(config.device))
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters,  # `False` for image-only task
        )

    optimizer = create_optimizer(config.optimizer, model)
    scheduler = create_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    start_epoch = 0
    global_step = 0
    if osp.isfile(config.pretrained_path):
        logger.info(f"Loading checkpoint from {config.pretrained_path}")
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        state_dict = checkpoint["model"]

        if config.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
        elif not pretrain:  # downstream init from pretrained ckpt

            # interpolate positional embeddings.
            if "beit" in config.model.vision_encoder.name:
                state_dict = interpolate_pos_embed_beit(state_dict, model_without_ddp)
            else:
                raise ValueError(
                    f" vision encoder: {config.model.vision_encoder.name} not implelented"
                )
            if not config.evaluate:  # finetuning from a pretarined weights.
                for key in list(state_dict.keys()):
                    if "bert" in key:
                        encoder_key = key.replace("bert.", "")
                        state_dict[encoder_key] = state_dict[key]
                        if not has_decoder:
                            del state_dict[key]

                    # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                    # only for generation tasks like VQA
                    if has_decoder and "text_encoder" in key:
                        if "layer" in key:
                            encoder_keys = key.split(".")
                            layer_num = int(encoder_keys[4])
                            if layer_num < config.model.text_encoder.fusion_layer:
                                del state_dict[key]
                                continue
                            else:
                                decoder_layer_num = layer_num - 9
                                encoder_keys[4] = str(decoder_layer_num)
                                encoder_key = ".".join(encoder_keys)
                        else:
                            encoder_key = key
                        decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                        state_dict[decoder_key] = state_dict[key]
                        del state_dict[key]

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {config.pretrained_path}")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch")

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    )
