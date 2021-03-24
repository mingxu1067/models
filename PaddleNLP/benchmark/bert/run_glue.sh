export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=sts

python3.6 -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./need_to_remove \
    --load_dir ./glue_start/model_start \
    --sparsity true

