export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

DATA_DIR=/root/.paddlenlp/datasets/books_wiki_en_corpus/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus

python3.6 ./run_pretrain_single.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 80 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir $DATA_DIR \
    --load_dir ./pretrain_models_1D_128/model_1000000/ \
    --output_dir ./pretrain_models_1D_512/ \
    --logging_steps 1000 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --sparsity true \
    --nonprune true
