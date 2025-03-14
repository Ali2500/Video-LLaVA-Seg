import math
import os
import os.path as osp
import sys
import torch
from torch import nn
import torch.nn as nn
import time
import shutil

from collections import defaultdict
from typing import Callable, List, Dict, Any, Tuple, Union
from dataloader import FalconReader
from torch.utils.data import Dataset, Sampler, DataLoader, get_worker_info

from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    PrinterCallback,
    seed_worker,
    is_datasets_available,
    is_torch_tpu_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME
from typing import List, Optional

from llava import distributed_utils as dist_utils

try:
    from llava.internal import init_possibly_internal_dataloader
    INTERNAL_IMPORTED = True
except ImportError as _:
    INTERNAL_IMPORTED = False

if is_datasets_available():
    from transformers.trainer import datasets


def dataloader_worker_init_fn(worker_id):
    seed_worker(worker_id)
    if INTERNAL_IMPORTED:
        init_possibly_internal_dataloader()


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    CHECKPOINT_COMPLETE_FLAG = "checkpoint_saved.flag"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # dict for accumulating other log variables
        self.log_vars_cache = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float32, device=self.args.device))
        self.log_var_keys = ["loss_text", "loss_mask", "loss_mask_ce", "loss_mask_dice", "mask_ious"]

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        def get_seg_head_embedding_params(model):
            embedding_params = []
            if model.model.seg_head is not None:
                for n, m in model.model.seg_head.named_modules():
                    if isinstance(m, nn.Embedding):
                        embedding_params.append("model.seg_head." + n + ".weight")

                for n, _ in model.model.seg_head.named_parameters(recurse=False):
                    if n.endswith("_embed") or n.endswith("_enc"):
                        embedding_params.append("model.seg_head." + n)

            return embedding_params

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # decay_parameters.extend(get_seg_head_embedding_params(opt_model))
            assert self.args.mm_projector_lr is not None
            assert self.args.mm_vision_tower_lr is not None
            assert self.args.seg_head_encoder_lr is not None
            assert self.args.seg_head_decoder_lr is not None

            def is_projector_param(name):
                return "mm_projector" in name

            def is_vision_tower_param(name):
                return "vision_tower" in name

            def is_seg_head_encoder_param(name):
                return name.startswith("model.seg_head.image_encoder")

            def is_seg_head_decoder_param(name):
                return name.startswith("model.seg_head") and not name.startswith("model.seg_head.image_encoder")

            def is_llm_param(name):
                return name.split(".")[1] not in ['mm_projector', 'seg_head', 'vision_tower']

            optimizer_grouped_parameters = [
                # LLM, decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n in decay_parameters and is_llm_param(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                # LLM, not decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n not in decay_parameters and is_llm_param(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                # vision tower, decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n in decay_parameters and is_vision_tower_param(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_vision_tower_lr,
                },
                # vision tower, not decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n not in decay_parameters and is_vision_tower_param(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_vision_tower_lr,
                },
                # projector, decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n in decay_parameters and is_projector_param(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                # projector, not decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n not in decay_parameters and is_projector_param(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
                # seg head encoder, decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n in decay_parameters and is_seg_head_encoder_param(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.seg_head_encoder_lr
                },
                # seg head encoder, not decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n not in decay_parameters and is_seg_head_encoder_param(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.seg_head_encoder_lr
                },
                # seg head decoder, decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n in decay_parameters and is_seg_head_decoder_param(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.seg_head_decoder_lr
                },
                # seg head decoder, not decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                        (n not in decay_parameters and is_seg_head_decoder_param(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.seg_head_decoder_lr
                },
            ]

            optimizer_grouped_parameters = [grp for grp in optimizer_grouped_parameters if len(grp) > 0]

            # sanity check: total length of optimizer param groups should equal length of all params with requires_grad = True
            sum_param_groups = sum([len(grp['params']) for grp in optimizer_grouped_parameters])
            sum_grad_params = len([p for p in opt_model.parameters() if p.requires_grad])
            assert sum_param_groups == sum_grad_params, f"Mismatch: {len(sum_param_groups)} =/= {len(sum_grad_params)}"

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        should_save = dist_utils.get_local_rank() == 0 if self.args.save_on_each_node else dist_utils.is_main_process()
        if should_save:
            # for some reason this is not saved via deepspeed checkpointing
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            with open(os.path.join(output_dir, self.CHECKPOINT_COMPLETE_FLAG), 'w'):
                pass

        dist_utils.synchronize()

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(LLaVATrainer, self)._save(output_dir, state_dict)
        # if getattr(self.args, 'tune_mm_mlp_adapter', False):
        #     pass
        # else:
        #     super(LLaVATrainer, self)._save(output_dir, state_dict)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if not self.is_deepspeed_enabled:
            super()._load_optimizer_and_scheduler(checkpoint)

        if checkpoint is None:
            return

        self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))

    def _get_learning_rate(self):
        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()  # support multiple parameter groups
                last_lr = sorted(list(set(last_lr)))
                if len(last_lr) == 1:
                    last_lr = last_lr[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[0]["lr"]
            else:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = dataloader_worker_init_fn

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            # reduce other logging variables
            num_objs = self.log_vars_cache.pop("num_object_tracks", None)
            if num_objs is not None:
                dist_utils.reduce(num_objs)

            reduce_list = []
            reduce_keys = []

            for key in sorted(self.log_vars_cache.keys()):
                if key == "loss_text":
                    v = self._nested_gather(self.log_vars_cache[key]).mean().item()
                    logs[key] = round(v / (self.state.global_step - self._globalstep_last_logged), 4)
                else:
                    metric_sum = self.log_vars_cache[key]
                    assert num_objs is not None, f"{key} is present but 'num_object_tracks' was not found"
                    reduce_list.append(metric_sum)
                    reduce_keys.append(key)

            if reduce_keys and num_objs > 0:
                reduce_list = torch.stack(reduce_list)
                dist_utils.reduce(reduce_list)
                for key, v in zip(reduce_keys, reduce_list):
                    v = (v / num_objs).item()
                    logs[key] = round(v / (self.state.global_step - self._globalstep_last_logged), 4)

            # BEFORE ---------------
            # # reduce other logging variables
            # num_objs = self.log_vars_cache.pop("num_object_tracks", None)
            # for key in sorted(self.log_vars_cache.keys()):
            #     if key == "mask_ious":
            #         iou_sum = self.log_vars_cache[key]
            #         assert num_objs is not None
            #         dist_utils.reduce(iou_sum)
            #         dist_utils.reduce(num_objs)
            #         if num_objs.item() == 0:
            #             continue
            #         v = (iou_sum / num_objs).item()
            #     else:
            #         v = self._nested_gather(self.log_vars_cache[key]).mean().item()

            #     logs[key] = round(v / (self.state.global_step - self._globalstep_last_logged), 4)
            # -------------

            self.log_vars_cache.clear()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, model_outputs = self.compute_loss(model, inputs, return_outputs=True)

        if "num_object_tracks" in model_outputs:
            num_object_tracks = model_outputs["num_object_tracks"].float().sum().detach()
            self.log_vars_cache["num_object_tracks"] += num_object_tracks
        else:
            num_object_tracks = None

        for key in self.log_var_keys:
            if key in model_outputs:
                if key == "loss_text":
                    self.log_vars_cache[key] += model_outputs[key].detach() / self.args.gradient_accumulation_steps
                else:
                    assert num_object_tracks is not None
                    if key == "mask_ious":
                        self.log_vars_cache[key] += model_outputs[key].float().sum().detach() # IoU is raw sum over objects i.e. no scaling needed
                    else:
                        self.log_vars_cache[key] += model_outputs[key].float().sum().detach() * num_object_tracks # unscale the loss terms so we can rescale them at the end over all sub-iterations and all processes

                # Before (buggy):
                # if key == "mask_ious":
                #     self.log_vars_cache[key] += model_outputs[key].float().sum().detach()
                #     self.log_vars_cache["num_object_tracks"] += model_outputs.pop("num_object_tracks").float().sum().detach()
                # else:
                #     self.log_vars_cache[key] += model_outputs[key].detach() / self.args.gradient_accumulation_steps

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
