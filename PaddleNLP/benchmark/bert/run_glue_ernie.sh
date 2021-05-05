export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=sts

python3.6 -u ./run_glue_test.py \
    --model_type ernie \
    --model_name_or_path ernie-2.0-large-en \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./need_to_remove \
    --sparsity true \
    --cusparselt true
