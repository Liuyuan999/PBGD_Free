# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

import deepspeed
import numpy as np
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch.nn as nn

def get_selector_train_ds_config( #bak
    offload,
    adam_offload=True,
    stage=0,
    bf16=False,
    max_norm=10.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
):
    device = "cpu" if offload else "none"
    if stage:
        zero_opt_dict = {
            "stage": stage,
            "offload_param": {"device": device},
            "offload_optimizer": {
                "device": "cpu" if adam_offload else "none",
                "pin_memory": True,
            },
            "sub_group_size": "auto",
            "stage3_max_live_parameters": "auto",
            "stage3_max_reuse_distance": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "reduce_bucket_size": "auto",
            # ZeRO++
            "zero_hpz_partition_size": zpg,
            "zero_quantized_weights": False,
            "zero_quantized_gradients": False,
        }
    else:
        zero_opt_dict = {
            "stage": stage,
        }

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": False,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }
    

def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]



def freeze_model(model, model_param=None):
    """
    Freezes all parameters in the model except those whose names match entries in model_param,
    and except all parameters in sub‐modules named exactly in model_param.

    Args:
        model:       nn.Module
        model_param: list of str, or None.
                     If None or empty, the model is left untouched.
                     Otherwise, any parameter whose name contains any of the strings in
                     model_param, or which belongs to a sub-module whose attribute name
                     is exactly one of the strings in model_param, will be unfrozen;
                     all others will be frozen.
    """
    # import torch.nn as nn
    
    # nothing to do
    if not model_param:
        return 

    # ensure it's a list
    if isinstance(model_param, str):
        model_param = [model_param]

    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze by substring match in name
    for name, p in model.named_parameters():
        if any(substr in name for substr in model_param):
            p.requires_grad = True

    # 3) Unfreeze entire submodules named exactly in model_param
    for attr in model_param:
        submod = getattr(model, attr, None)
        if isinstance(submod, nn.Module):
            for p in submod.parameters():
                p.requires_grad = True

def get_train_params(model: nn.Module, model_param=None):
    """
    Returns a list of parameters to train.

    Args:
        model:       an nn.Module
        model_param: list of str, str, or None.
                     • If None (or empty), returns all params in model with requires_grad=True.
                     • Otherwise, returns:
                       1) any named parameter whose name contains any substring in model_param, and
                       2) all parameters in any submodule of `model` whose attribute name exactly
                          matches an entry in model_param.

    Returns:
        List[nn.Parameter]
    """
    
    # 1) If no filtering keys, just return all trainable params:
    if not model_param:
        return [p for p in model.parameters() if p.requires_grad]

    # 2) Normalize into list of strings
    if isinstance(model_param, str):
        model_param = [model_param]

    params = []

    # 3) Collect by name‐substring match
    for name, p in model.named_parameters():
        if any(substr in name for substr in model_param):
            params.append(p)

    # 4) Collect all params from sub‐modules named exactly in model_param
    for attr in model_param:
        submod = getattr(model, attr, None)
        if isinstance(submod, nn.Module):
            params.extend(list(submod.parameters()))

    # 5) Deduplicate (preserve first occurrence order)
    seen_ids = set()
    unique_params = []
    for p in params:
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            unique_params.append(p)

    return unique_params
