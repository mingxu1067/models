export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python3.6 -u ./run_squad.py \
        --model_type ernie \
        --model_name_or_path ernie-2.0-en \
        --max_seq_length 384 \
        --batch_size 12 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir need_to_remove
