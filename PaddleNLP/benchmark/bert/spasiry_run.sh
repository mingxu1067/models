#!/bin/bash

export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH

# TASK_NAMES=(qnli sst cola sts rte)
TASK_NAMES=(qnli)

# GPU_ID=0
# for name in "${TASK_NAMES[@]}";
# do
#     echo "Pre-training" $name "At" $GPU_ID
#     python3.6 -u ./run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name $name \
#     --max_seq_length 128 \
#     --select_device gpu:$GPU_ID \
#     --batch_size 64   \
#     --learning_rate 2e-5 \
#     --num_train_epochs 15 \
#     --sparsity true \
#     --nonprune true \
#     --load_dir ./glue_start_128/$name/ \
#     --output_dir ./glue_dss_128/$name/ &> dss_128_${name}.log &

#     GPU_ID=$((GPU_ID + 1))
# done

# GPU_ID=0
# for name in "${TASK_NAMES[@]}";
# do
#     echo "Pre-training" $name "At" $GPU_ID
#     python3.6 -u ./run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name $name \
#     --max_seq_length 128 \
#     --select_device gpu:$GPU_ID \
#     --batch_size 64   \
#     --learning_rate 2e-5 \
#     --num_train_epochs 15 \
#     --sparsity true \
#     --nonprune true \
#     --load_dir ./glue_start_512/$name/ \
#     --output_dir ./glue_dss_512/$name/ &> dss_512_${name}.log &

#     GPU_ID=$((GPU_ID + 1))
# done

# GPU_ID=2
# for name in "${TASK_NAMES[@]}";
# do
#     echo "Pre-training" $name "At" $GPU_ID
#     python3.6 -u ./run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name $name \
#     --max_seq_length 128 \
#     --select_device gpu:$GPU_ID \
#     --batch_size 64   \
#     --learning_rate 2e-5 \
#     --num_train_epochs 15 \
#     --nonprune true \
#     --load_dir ./glue_start_dense/$name/ \
#     --output_dir ./glue_dd/$name/ &> dd_${name}.log &

#     GPU_ID=$((GPU_ID + 1))
# done

# GPU_ID=2
# for name in "${TASK_NAMES[@]}";
# do
#     echo "Pre-training" $name "At" $GPU_ID
#     python3.6 -u ./run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name $name \
#     --max_seq_length 128 \
#     --select_device gpu:$GPU_ID \
#     --batch_size 64   \
#     --learning_rate 2e-5 \
#     --num_train_epochs 15 \
#     --sparsity true \
#     --load_dir ./glue_start_dense/$name/ \
#     --output_dir ./glue_ds/$name/ &> ds_${name}.log &

#     GPU_ID=$((GPU_ID + 1))
# done

GPU_ID=0
for name in "${TASK_NAMES[@]}";
do
    echo "Pre-training" $name "At" $GPU_ID
    python3.6 -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $name \
    --max_seq_length 128 \
    --select_device gpu:$GPU_ID \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 15 \
    --sparsity true \
    --load_dir ./glue_dd/$name/model_final/ \
    --output_dir ./glue_dds/$name/ &> dds_${name}.log &

    GPU_ID=$((GPU_ID + 1))
done