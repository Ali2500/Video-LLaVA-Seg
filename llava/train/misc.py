import safetensors
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
import os
import os.path as osp

from einops import rearrange, repeat
from tqdm import tqdm
from glob import glob

from llava import distributed_utils as dist_utils
from llava.model import LlavaConfig


def ViTPatchGenerator_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if self.abs_pos:
        if f'{prefix}pos_embed' in state_dict:
            self._load_embed(state_dict[f'{prefix}pos_embed'], self.pos_embed)            


def ViTPatchLinear_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if self.bias is not None:
        if f'{prefix}bias' not in state_dict:
            print("ignoring stupid error 1")
            missing_keys.append(f'{prefix}bias')
        else:
            self.bias.data.copy_(state_dict[f'{prefix}bias'])

    if f'{prefix}weight' not in state_dict:
        missing_keys.append(f'{prefix}weight')
    else:
        chk_weight = state_dict[f'{prefix}weight']
        if chk_weight.shape != self.weight.shape:
            src_patch_size = int(math.sqrt(chk_weight.shape[1] // 3))

            assert (src_patch_size ** 2) * 3 == chk_weight.shape[1], 'Unable to interpolate non-square patch size'

            chk_weight = rearrange(chk_weight, 'b (c h w) -> b c h w', c=3, h=src_patch_size, w=src_patch_size)
            chk_weight = F.interpolate(chk_weight, size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=True, antialias=False)
            chk_weight = rearrange(chk_weight, 'b c h w -> b (c h w)')
            
        self.weight.data.copy_(chk_weight)


def load_model_weights(
        model: nn.Module, 
        restore_weights_dir: str, 
        ignore_missing_seg_head_wts: bool = False, 
        ignore_vocab_size_mismatch: bool = False
    ):
    all_model_keys = set(list(model.state_dict().keys()))
    all_seen_keys = set()

    if hasattr(model.model.vision_tower.vision_tower, "radio_model"):
        # monkey patch function to avoid annoying errors when loading partial state_dicts
        patch_generator = model.model.vision_tower.vision_tower.radio_model.model.patch_generator
        patch_generator._load_from_state_dict = functools.partial(ViTPatchGenerator_load_from_state_dict, patch_generator)

        patch_linear = model.model.vision_tower.vision_tower.radio_model.model.patch_generator.embedder
        patch_linear._load_from_state_dict = functools.partial(ViTPatchLinear_load_from_state_dict, patch_linear)

    vocab_embedding_keys = ["model.embed_tokens.weight", "lm_head.weight"]

    # later versions of huggingface use safetensors format, older versions rely on standard pytorch weights in .bin format
    weight_files = sorted(glob(osp.join(restore_weights_dir, "model-*.safetensors")))
    if weight_files:
        dist_utils.print_once(f"Restoring model weights from safetensors files...")

        for filepath in tqdm(weight_files, disable=dist_utils.get_local_rank() != 0):
            part_state_dict = {}
            with safetensors.safe_open(filepath, framework="pt", device="cpu") as fh:
                for key in fh.keys():
                    part_state_dict[key] = fh.get_tensor(key)
                    all_seen_keys.add(key)

                if ignore_vocab_size_mismatch:
                    model_state_dict = model.state_dict()
                    # allow current model to have more language tokens than what we have in the loaded checkpoint
                    for key in vocab_embedding_keys:
                        if not (key in part_state_dict and key in model_state_dict):
                            continue

                        model_shape = model_state_dict[key].shape
                        ckpt_shape = part_state_dict[key].shape
                        assert model_shape[1] == ckpt_shape[1], f"Dimension mismatch: {model_shape}, {ckpt_shape}"
                        n_pad = model_shape[0] - ckpt_shape[0]
                        if n_pad > 0:
                            dist_utils.print_once(f"Padding '{key}' with {n_pad} averaged token(s).")
                            avg_embed = part_state_dict[key].mean(0)  # [C]
                            avg_embed = repeat(avg_embed, "C -> n_pad C", n_pad=n_pad)
                            part_state_dict[key] = torch.cat((part_state_dict[key], avg_embed), 0)
                
                _, unexpected = model.load_state_dict(part_state_dict, strict=False)
                assert not unexpected, f"Encountered unexpected keys: {unexpected}"

    else:
        weight_files = sorted(glob(osp.join(restore_weights_dir, "pytorch_model-*.bin")))
        assert weight_files, f"No model weights files found in directory: {restore_weights_dir}"
        dist_utils.print_once(f"Restoring model weights from bin files...")

        for filepath in tqdm(weight_files, disable=dist_utils.get_local_rank() != 0):
            part_state_dict = torch.load(filepath, map_location='cpu')
            all_seen_keys = all_seen_keys.union(set(list(part_state_dict.keys())))
            _, unexpected = model.load_state_dict(part_state_dict, strict=False)
            assert not unexpected, f"Encountered unexpected keys: {unexpected}"

    # regardless of checkpoint format, all model keys should have been restored by now
    missing_keys = all_model_keys - all_seen_keys

    if ignore_missing_seg_head_wts:
        missing_keys = set([k for k in missing_keys if not k.startswith("model.seg_head.")])

    if missing_keys:
        raise ValueError(f"Did not encounter the following model keys: {missing_keys}")

    # ViTPatchGenerator.ALLOW_NONSTRICT_LOAD_STATE_DICT = False


def verify_config_consistency(config: LlavaConfig, restore_weights_dir):
    rw_config_path = osp.join(restore_weights_dir, "config.json")
    assert osp.exists(rw_config_path), f"No config file found at {rw_config_path}"
    rw_config = LlavaConfig.from_json_file(rw_config_path).to_dict()

    keys_to_match = [
        "vision_tower",
        "use_text_prompt"
    ]

    keys_to_warn = [
        "image_size",
        "num_frames",
        "num_slow_frames"
    ]

    config = config.to_dict()
    for k in config:
        assert k in rw_config, f"'{k}' found in current config but not in restore weights config"
        if k in keys_to_match:
            assert config[k] == rw_config[k], f"'{k}' mismatch: {config[k]} =/= {rw_config[k]}"

        if k in keys_to_warn:
            if config[k] != rw_config[k]:
                print(f"WARN: '{k}' mismatch: {config[k]} =/= {rw_config[k]}")
