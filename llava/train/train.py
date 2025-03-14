# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from ast import Import
import logging
import os
import pathlib
import shutil
import wandb
import sys
import warnings

warnings.filterwarnings("ignore", message=r"^.*find harunacompass consul*$")

from glob import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from transformers.trainer import PrinterCallback, logger as train_logger

import os.path as osp
import torch
import transformers
from time import time as current_time


from llava.paths import Paths
from llava import conversation as conversation_lib
from llava.model import *
from llava.train.llava_trainer import LLaVATrainer
from llava.train.data_classes import TrainingArguments, ModelArguments, DataArguments
from llava.train.data_parsing import make_supervised_data_module
from llava.train.misc import load_model_weights, verify_config_consistency
from llava.constants import DEFAULT_VID_SEG_TOKEN
from llava import distributed_utils as dist_utils

# try:
#     from llava.internal.hdfs import upload_model_dir_to_hdfs
#     INTERNAL_IMPORTED = True
# except ImportError as _:
#     INTERNAL_IMPORTED = False

local_rank = None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k,
                     t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True)
                 for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, qv_proj_only=False):
    if qv_proj_only:
        dist_utils.print_once('Only add LoRA to QV proj')
        return ['q_proj', 'v_proj']
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):  # pretraining (stage 1)
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        should_save = dist_utils.get_local_rank() == 0 if trainer.args.save_on_each_node else dist_utils.is_main_process()

        if should_save:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(
                    parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(
                    mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(
                    output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        dist_utils.synchronize()
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    dist_utils.synchronize()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def unfreeze_vit(vision_tower):
    for _, p in vision_tower.named_parameters():
        p.requires_grad = True


def format_bytes(size):
    billion = 10**9
    million = 10**6

    if size >= billion:
        return f"{size / billion:.2f}B"
    elif size >= million:
        return f"{size / million:.2f}M"
    else:
        return f"{size} bytes"


class ETAEstimatorCallback(transformers.TrainerCallback):
    step_times = []
    curr_step_start = 0.
    log_vars = ['loss', 'learning_rate']
    
    def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        if not state.is_local_process_zero:
            return
        
        ETAEstimatorCallback.curr_step_start = current_time()

    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        step_time = current_time() - ETAEstimatorCallback.curr_step_start
        ETAEstimatorCallback.step_times.append(step_time)
        ETAEstimatorCallback.step_times = ETAEstimatorCallback.step_times[-5000:]

    def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, logs=None, **kwargs):
        avg_step_time = sum(ETAEstimatorCallback.step_times) / float(len(ETAEstimatorCallback.step_times))
        eta = (state.max_steps - state.global_step) * avg_step_time
        days, rem = divmod(eta, 3600*24)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        eta_str = f"{int(days):02d}-{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"  # % (int(days), int(hours), int(minutes), int(seconds))

        if state.is_local_process_zero:
            logstr = f"[{state.global_step}/{state.max_steps}]"
            if logs:
                for k, v in logs.items():
                    if k == 'learning_rate':
                        k = 'lr'
                        if isinstance(v, (list, tuple)):
                            logstr = logstr + f" - {k}: [" + ", ".join([f"{val:.2E}" for val in v]) + "]" 
                        else:
                            logstr = logstr + f" - {k}: {v:.2E}"
                    else:
                        logstr = logstr + f" - {k}: {v:.3f}"

            logstr = logstr + f" - avg step time: {avg_step_time:.2f}"
            logstr = logstr + f" - ETA: {eta_str}"
            print(logstr)

        if dist_utils.is_main_process() and getattr(wandb, "run", None) is not None:  # global rank 0
            wandb.log({k: v for k, v in logs.items() if k in self.log_vars}, step=state.global_step)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    training_args.disable_tqdm = True
    should_save = dist_utils.get_local_rank() == 0 if training_args.save_on_each_node else dist_utils.is_main_process()

    if not osp.isabs(training_args.output_dir):
        assert "VIDEONET_MODELS_DIR" in os.environ
        training_args.output_dir = osp.join(Paths.saved_models_dir(), training_args.output_dir)
        training_args.logging_dir = osp.join(Paths.saved_models_dir(), training_args.logging_dir)
        dist_utils.print_once(f"Setting output directory to: {training_args.output_dir}")

    if model_args.pretrain_mm_mlp_adapter is not None:
        if not osp.isabs(model_args.pretrain_mm_mlp_adapter):
            model_args.pretrain_mm_mlp_adapter = osp.join(Paths.saved_models_dir(), model_args.pretrain_mm_mlp_adapter)
        assert osp.exists(model_args.pretrain_mm_mlp_adapter), f"Weights file not found: {model_args.pretrain_mm_mlp_adapter}"

    # clear output_dir if needed
    if training_args.clear_output_dir and osp.exists(training_args.output_dir) and should_save:
        shutil.rmtree(training_args.output_dir, ignore_errors=True)

    dist_utils.synchronize()

    if model_args.restore_weights:
        if not osp.isabs(model_args.restore_weights):
            assert "VIDEONET_MODELS_DIR" in os.environ
            model_args.restore_weights = osp.join(os.environ["VIDEONET_MODELS_DIR"], model_args.restore_weights)
        assert osp.exists(model_args.restore_weights), f"Restore weights path not found: {model_args.restore_weights}"

    log_level = logging.INFO if dist_utils.get_local_rank() == 0 else logging.WARN
    train_logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    train_logger.addHandler(handler)

    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    model_max_length_args = {}
    if 'llava-v1.6-8b' not in model_args.model_name_or_path:
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True)
        if config.max_position_embeddings < training_args.model_max_length:
            dist_utils.print_once(
                f'Set the max_position_embeddings from {config.max_position_embeddings} to {training_args.model_max_length}')
            model_max_length_args.update(
                {'max_position_embeddings': training_args.model_max_length})
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args,
                **model_max_length_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )

    if model_args.seg_image_size is None:
        model_args.seg_image_size = model_args.image_size

    model.config.use_cache = False
    model.config.use_text_prompt = data_args.use_text_prompt
    assert data_args.num_slow_frames <= data_args.num_frames
    model.config.num_frames = data_args.num_frames
    model.config.num_slow_frames = data_args.num_slow_frames
    model.config.max_seg_frames = data_args.max_seg_frames
    model.config.image_size = model_args.image_size
    model.config.seg_image_size = model_args.seg_image_size
    model.config.seg_pad_mode = data_args.seg_pad_mode

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, training_args.lora_qv_proj_only),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        dist_utils.print_once("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    model.config.conversation_template = "vicuna_v1"

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.pad_token is None:
            dist_utils.print_once("Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version]
            model.config.conversation_template = model_args.version
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    model_args.num_frames = data_args.num_frames
    model_args.num_slow_frames = data_args.num_slow_frames
    data_args.seg_image_size = model_args.seg_image_size

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor
        data_args.is_multimodal = True
        data_args.dataloader_num_workers = training_args.dataloader_num_workers

        if model.config.seg_head is not None:
            data_args.seg_frames_shared_encoder = not model.get_model().seg_head.has_image_encoder
        else:
            data_args.seg_frames_shared_encoder = None

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0]*base_size, g[1]*base_size] for g in grids]

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if model.config.seg_head is not None:
            for p in model.get_model().seg_head.parameters():
                p.requires_grad = True

        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        if training_args.unfreeze_mm_vision_tower:
            lr_of_vit = training_args.mm_vision_tower_lr if training_args.mm_vision_tower_lr is not None else training_args.learning_rate
            lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
            training_args.mm_projector_lr = lr_of_mlp
            unfreeze_vit(vision_tower)
            dist_utils.print_once(
                f'Tune the entire model! The LR of ViT is {lr_of_vit}. The LR of MLP is {lr_of_mlp}. The LR of LLM is {training_args.learning_rate}')

        if training_args.seg_head_encoder_lr is None:
            training_args.seg_head_encoder_lr = training_args.learning_rate
        if training_args.seg_head_decoder_lr is None:
            training_args.seg_head_decoder_lr = training_args.learning_rate
        if training_args.mm_vision_tower_lr is None:
            training_args.mm_vision_tower_lr = training_args.learning_rate

        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in model.get_model().parameters() if p.requires_grad)

        dist_utils.print_once(f"Total parameters: {format_bytes(total_params)}")
        dist_utils.print_once(f"Trainable parameters: {format_bytes(trainable_params)}")

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

        model.config.mm_use_sf_vid_separator_token = data_args.mm_use_sf_vid_separator_token = model_args.mm_use_sf_vid_separator_token
        training_args.use_sf_vid_separator_token = model_args.mm_use_sf_vid_separator_token
        
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.video_mode = data_args.video_mode
        if model_args.seg_head:
            model.config.seg_token_idx = tokenizer(DEFAULT_VID_SEG_TOKEN)['input_ids'][-1]

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args
    )

    # set number of epochs inside the dataset to avoid weird issues with FalconDataset
    if hasattr(data_module["train_dataset"], "set_num_epochs"):
        data_module["train_dataset"].set_num_epochs(training_args.num_train_epochs)
        training_args.num_train_epochs = 1

    # remove checkpoint directories that are not fully complete otherwise it will cause training to crash
    checkpoint_dirs = sorted(glob(osp.join(training_args.output_dir, "checkpoint-*")))
    for d in checkpoint_dirs:
        if not osp.exists(osp.join(d, LLaVATrainer.CHECKPOINT_COMPLETE_FLAG)):
            dist_utils.print_once(f"Deleting incomplete checkpoint dir: {d}")
            if should_save: # dist_utils.is_main_process():
                shutil.rmtree(d, ignore_errors=True)

    dist_utils.synchronize()

    resume_from_checkpoint = len(glob(osp.join(training_args.output_dir, "checkpoint-*"))) > 0
    if model_args.restore_weights and not resume_from_checkpoint:
        # load model weights from pretraining directory
        verify_config_consistency(config, model_args.restore_weights)
        load_model_weights(
            model, 
            model_args.restore_weights, 
            ignore_missing_seg_head_wts=True,
            ignore_vocab_size_mismatch=model_args.seg_head is not None
        )
        dist_utils.synchronize()

    # initialize wandb
    tb_enabled = False
    if training_args.report_to:
        tb_enabled = training_args.report_to[0] == "tensorboard"
        
    if dist_utils.is_main_process() and not tb_enabled and "dummy" not in osp.split(training_args.output_dir)[-1]:
        if not training_args.wandb_run_name:
            training_args.wandb_run_name = osp.split(training_args.output_dir)[-1]
        
        resume_wandb = False if training_args.clear_output_dir else "allow"
        dist_utils.print_once(f"Initializing wandb...")
        wandb.init(project="videonet", name=training_args.wandb_run_name, resume=resume_wandb, id=training_args.wandb_run_name, mode=training_args.wandb_mode)
        wandb.config.update(model.config.to_dict())

    dist_utils.synchronize()

    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(ETAEstimatorCallback)

    if resume_from_checkpoint: # list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if should_save: # dist_utils.is_main_process():
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

        # once the final model has been saved, delete checkpoint directories to save space
        checkpoint_dirs = sorted(glob(osp.join(training_args.output_dir, "checkpoint-*")))
        dist_utils.print_once(f"Deleting intermediate checkpoints: {checkpoint_dirs}")
        for d in checkpoint_dirs:
            if should_save:
                shutil.rmtree(d, ignore_errors=True)

    if should_save: 
        with open(osp.join(training_args.output_dir, "training_complete.flag"), 'w') as fh:
            pass

    # upload to HDFS if needed
    # if training_args.sync_checkpoints_with_hdfs and INTERNAL_IMPORTED:
    #     upload_model_dir_to_hdfs(training_args.output_dir)

    dist_utils.synchronize()


if __name__ == "__main__":
    train()
