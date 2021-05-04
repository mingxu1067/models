export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 12 \
        --learning_rate 3e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_128_b12_lr3e5 \
        --load_dir ./squad_start_128/ \
        --sparsity true \
        --nonprune true &> dss_pretrain_128_b12_lr3e5.log &

export CUDA_VISIBLE_DEVICES=1
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 12 \
        --learning_rate 3e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_512_b12_lr3e5 \
        --load_dir ./squad_start_512/ \
        --sparsity true \
        --nonprune true &> dss_pretrain_512_b12_lr3e5.log &

export CUDA_VISIBLE_DEVICES=2
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 12 \
        --learning_rate 3e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_128_b12_lr3e5_amp \
        --load_dir ./squad_start_128/ \
        --use_amp true \
        --sparsity true \
        --nonprune true &> dss_pretrain_128_b12_lr3e5_amp.log &

export CUDA_VISIBLE_DEVICES=3
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 12 \
        --learning_rate 3e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_512_b12_lr3e5_amp \
        --load_dir ./squad_start_512/ \
        --use_amp true \
        --sparsity true \
        --nonprune true &> dss_pretrain_512_b12_lr3e5_amp.log &

export CUDA_VISIBLE_DEVICES=4
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 8 \
        --learning_rate 4e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_128_b8_lr4e5 \
        --load_dir ./squad_start_128/ \
        --sparsity true \
        --nonprune true &> dss_pretrain_128_b8_lr4e5.log &

export CUDA_VISIBLE_DEVICES=5
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 8 \
        --learning_rate 4e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_512_b8_lr4e5 \
        --load_dir ./squad_start_512/ \
        --sparsity true \
        --nonprune true &> dss_pretrain_512_b8_lr4e5.log &

export CUDA_VISIBLE_DEVICES=6
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 8 \
        --learning_rate 4e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_128_b8_lr4e5_amp \
        --load_dir ./squad_start_128/ \
        --use_amp true \
        --sparsity true \
        --nonprune true &> dss_pretrain_128_b8_lr4e5_amp.log &

export CUDA_VISIBLE_DEVICES=7
python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_seq_length 384 \
        --batch_size 8 \
        --learning_rate 4e-5 \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir dss_pretrain_512_b8_lr4e5_amp \
        --load_dir ./squad_start_512/ \
        --use_amp true \
        --sparsity true \
        --nonprune true &> dss_pretrain_512_b8_lr4e5_amp.log &
