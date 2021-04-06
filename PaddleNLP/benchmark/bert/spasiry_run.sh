#!/bin/bash

export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH

# TASK_NAMES=(cola sst sts mnli qnli rte)
TASK_NAMES=(cola sst sts qnli rte)
# declare -A EPOCHS=([cola]=3 [sst]=3 [sts]=3 [mnli]=3 [qnli]=3 [rte]=3)
# declare -A SAVE_STEPS=([cola]=42 [sst]=315 [sts]=27 [mnli]=1840 [qnli]=491 [rte]=11)

for name in "${TASK_NAMES[@]}";
do
    echo "Pre-training" $name
    python3.6 -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $name \
    --max_seq_length 128 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ./glue_DD/$name/ | tee ./log/log_1D/$name/dd.log
done

# declare -A SPARSITY_EPOCHS=([cola]=30 [sst]=30 [sts]=30 [mnli]=30 [qnli]=30 [rte]=30)
# declare -A SPARSITYSAVE_STEPS=([cola]=252 [sst]=1890 [sts]=162 [mnli]=2000 [qnli]=2000 [rte]=66)

for name in "${TASK_NAMES[@]}";
do
    echo "Fine-tuning with Pretrain" $name
    python3.6 -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $name \
    --max_seq_length 128 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --output_dir ./glue_DDS_1D/$name/ \
    --load_dir ./glue_DD/$name/model_final \
    --sparsity true | tee ./log/log_1D/$name/dds.log
done

for name in "${TASK_NAMES[@]}";
do
    echo "Fine-tuning without Pretrain" $name

    python3.6 -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $name \
    --max_seq_length 128 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --num_train_epochs30 \
    --output_dir ./glue_DS_1D/$name/ \
    --sparsity true | tee ./log/log_1D/$name/ds.log
done
