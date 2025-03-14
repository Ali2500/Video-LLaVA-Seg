from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from llava.constants import DEFAULT_VICAS_VERSION
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_use_sf_vid_separator_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # ------------------------------
    image_size: Optional[int] = field(default=384)
    restore_weights: Optional[str] = field(default=None)
    seg_head: Optional[str] = field(default=None)
    seg_image_size: Optional[int] = field(default=1024)
    seg_num_queries: Optional[int] = field(default=1)
    seg_backbone: Optional[str] = field(default="facebook/sam2.1-hiera-small")


@dataclass
class DataArguments:
    data_path: str = field(default=None)  # unused
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)  # unused
    image_aspect_ratio: str = 'square'
    # -------------------------------------
    video_mode: Optional[bool] = field(default=True)
    num_frames: int = 32
    num_slow_frames: int = 8
    training_data_type: str = "video_caption"
    subsample_factor: float = 1.0
    data_version: Optional[str] = field(default="v1")
    use_text_prompt: Optional[bool] = field(default=True)
    use_reworded_captions: Optional[bool] = field(default=True)
    use_falcon_dataset: Optional[bool] = field(default=False)
    max_seg_frames: Optional[int] = field(default=8)
    vicas_version: Optional[str] = field(default=DEFAULT_VICAS_VERSION)
    seg_pad_mode: Optional[str] = "topleft"
    use_caption_only_annotations: Optional[bool] = field(default=False)
    exclude_seg: Optional[bool] = field(default=False)
    exclude_captions: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_qv_proj_only: bool = False
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    seg_head_encoder_lr: Optional[float] = None
    seg_head_decoder_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    # -------------------------------------
    clear_output_dir: bool = False
    wandb_run_name: Optional[str] = None
    wandb_mode: Optional[str] = "online"
    sync_checkpoints_with_hdfs: bool = False
