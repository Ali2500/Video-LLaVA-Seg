#!/usr/bin/env bash

set -x

# Set the following according to your setup. We use an overall batch size of 64. (2 nodes * 8 GPUs * 4 accumulation steps)
export GRADIENT_ACCU_STEPS=4
export BATCH_SIZE=1
export NUM_NODES=2
export RDZV_ENDPOINT="" # not needed in some cases depending on pytorch version and hardware infrastructure. Should be in format <IP address>:<port number>

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export BASE_LR=2e-5
export VIT_LR=2e-6

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "${NUM_NODES}" -gt "1" ]; then
    LAUNCHER="torchrun --nproc_per_node ${NUM_GPUS} --nnodes ${NUM_NODES} --rdzv_endpoint ${RDZV_ENDPOINT} --rdzv_backend c10d"
else
    LAUNCHER="torchrun --nproc_per_node ${NUM_GPUS} --master_port 2222"
fi

# If we have enough >4 GPUs with 80GB VRAM then offloading is not needed which speeds up training a lot
TOTAL_VRAM=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024**3)")  # total VRAM in GB
if [ "${TOTAL_VRAM}" -gt "50" ] && [ "${NUM_GPUS}" -gt "4" ]; then
    DEEPSPEED_CONFIG="scripts/deepspeed_configs/zero2.json"
else
    DEEPSPEED_CONFIG="scripts/deepspeed_configs/zero2_offload.json"
fi

${LAUNCHER} llava/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --training_data_type vicas \
    --version v1 \
    --vision_tower nvidia/RADIO \
    --mm_projector_type mlp2x_gelu \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr ${VIT_LR} \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 15 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5824 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True $@
