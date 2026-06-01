# Hugging Face wrapper for HyenaDNA
# Date added to repo: 2026-05-05
# Source Colab notebook: HyenaDNA training & inference example (Public)
# Source URL: https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL
# Local modifications:
#   - Remove genomic-benchmarks import
#   - Relative import of HyenaDNAModel (model code not inlined in this file)
#   - Use weights_only=False for torch.load() (PyTorch >= 2.6)
#   - With use_head=True, the model is a sequence-to-vector pooler that
#     returns [B, d_model] features. Callers attach their own task head
#     (e.g. classification, regression) externally; this keeps the wrapper
#     task-agnostic and lets the backbone be reused as a feature extractor.
# Modified by: Jeffrey Dick

#@title Huggingface Pretrained Wrapper
# for Huggingface integration, we use a wrapper class around the model
# to load weights
import re
import os
import json
import subprocess

import torch
from transformers import PreTrainedModel
from .standalone_hyenadna import HyenaDNAModel

def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict.

    scratch_dict: dict, a state dict from a newly initialized HyenaDNA model
    pretrained_dict: dict, a state dict from the pretrained ckpt
    checkpointing: bool, whether the gradient checkpoint flag was used in the
    pretrained model ckpt. This slightly changes state dict keys, so we patch
    that if used.

    return:
    dict, a state dict with the pretrained weights loaded (head is scratch)

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    """

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path,
        model_name,
        download=False,
        config=None,
        device='cpu',
        use_head=False,
        head_pooling_mode="pool",
    ):
        """Load HyenaDNA weights from ``path`` / ``model_name``.

        ``download=False`` (default): use the local directory when present; git-clone
        from Hugging Face when missing. ``download=True``: remove an existing directory
        and re-clone from Hugging Face. When ``use_head=True`` the model attaches a
        sequence-to-vector pooler (``head_pooling_mode``) and ``forward(input_ids)``
        returns pooled ``[B, d_model]`` features; the caller is expected to attach
        a task-specific head (classification, regression, ...) on top.
        """
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(
            **config,
            use_head=use_head,
            head_pooling_mode=head_pooling_mode,
        )
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device),
            # Needed for PyTorch >= 2.6
            weights_only=False,
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model
