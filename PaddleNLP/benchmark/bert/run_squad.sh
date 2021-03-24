export PYTHONPATH=/paddle/sparsity/models/PaddleNLP:$PYTHONPATH

python3.6 -u ./run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --output_dir need_to_remove
