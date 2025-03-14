CHUNKS=$(python3 -c "import torch; print(torch.cuda.device_count())")

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${IDX} python3 llava/inference/main.py --num_chunks $CHUNKS --chunk_idx $IDX $@ &
    PROCESS_ID_LIST="${PROCESS_ID_LIST} $!"
done
trap "kill ${PROCESS_ID_LIST}; exit 1" INT
wait