#!/usr/bin/env bash

set -x

# Set the following according to your setup. We use an overall batch size of 256. (4 nodes * 8 GPUs * 8 accumulation steps)
export BATCH_SIZE=1 # per GPU
export GRADIENT_ACCU_STEPS=8 # accumulation steps
export NUM_NODES=4
export RDZV_ENDPOINT="" # not needed in some cases depending on your infrastructure. Should be in format <IP address>:<port number>

export CUDA_DEVICE_MAX_CONNECTIONS=1
export BASE_LR=1e-3

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "${NUM_NODES}" -gt "1" ]; then
    LAUNCHER="torchrun --nproc_per_node ${NUM_GPUS} --nnodes ${NUM_NODES} --rdzv_endpoint ${RDZV_ENDPOINT} --rdzv_backend c10d"
else
    LAUNCHER="torchrun --nproc_per_node ${NUM_GPUS} --master_port 2222"
fi

${LAUNCHER} llava/train/train_mem.py \
    --deepspeed ./scripts/deepspeed_configs/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --vision_tower nvidia/RADIO \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --unfreeze_mm_vision_tower False \
    --image_aspect_ratio pad \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --mm_projector_lr ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5824 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --data_version v4_pt $@
