from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
import torch
import numpy as np
import psutil
import os
from torch import nn


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def _get_model_analysis_input(cfg, use_train_input):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            lib/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    num_frames = cfg['num_frames']
    image_size = cfg['image_size']

    if use_train_input:
        input_tensors = torch.rand(
            rgb_dimension,
            num_frames,
            image_size,
            image_size,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            num_frames,
            image_size,
            image_size,
        )
    model_inputs = input_tensors.cuda(non_blocking=True).unsqueeze(0)

    inputs = (model_inputs,)
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            lib/config/defaults.py
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg, use_train_input)
    count_dict, *_ = model_stats_fun(model, inputs)
    count = sum(count_dict.values())
    model.train(model_mode)
    return count


def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            lib/config/defaults.py
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    print("Model:\n{}".format(model))
    print("Params: {:,}".format(params_count(model)))
    print("Mem: {:,} GB".format(gpu_mem_usage()))
    print(
        "Flops: {:,} G".format(
            get_model_stats(model, cfg, "flop", use_train_input)
        )
    )
    print(
        "Activations: {:,} M".format(
            get_model_stats(model, cfg, "activation", use_train_input)
        )
    )
    print("Mem: {:,} GB".format(gpu_mem_usage()))
    print("nvidia-smi")
    os.system("nvidia-smi")


if __name__ == '__main__':
    from models.blip import create_vit
    import ruamel.yaml as yaml
    import argparse
    import testa
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_queryd_timesformer.yaml')
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    visual_encoder, vision_width = create_vit(cfg['vit'], cfg['image_size'], cfg['vit_grad_ckpt'],
                                              cfg['vit_ckpt_layer'], 0, cfg)
    if cfg['token_merging']:
        merging_type = cfg['merging_type']
        if 'timesformer' in cfg['vit']:
            testa.patch.timesformer(visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,
                                    merging_type=merging_type, num_patches=visual_encoder.num_patches)
        else:
            testa.patch.vit(visual_encoder, trace_source=(merging_type == 'frame'), prop_attn=False,
                            merging_type=merging_type)
        visual_encoder.r = cfg['testa_r']
    log_model_info(visual_encoder.cuda(), cfg)
