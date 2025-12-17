#!/usr/bin/bash

export TOKENIZERS_PARALLELISM="false"

deepspeed gigacheck/train/scripts/train_classification_model.py \
    --deepspeed ${ROOT_DIR}/gigacheck/deepspeed_configs/zero2.json \
    --pretrained_model_name "mistralai/Mistral-7B-v0.3" \
    --attn_implementation "flash_attention_2" \
    --train_data_path "/data/classification/bilingual/train.jsonl" \
    --eval_data_path "/data/classification/bilingual/valid.jsonl" \
    --max_sequence_length 1024 \
    --min_sequence_length 100 \
    --random_sequence_length True \
    --lora_enable True \
    --lora_r 8 \
    --bf16 True \
    --output_dir "train_logs/mistral_7b_bilingual_dataset" \
    --num_train_epochs 20 \
    --learning_rate 0.00003 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.5}' \
    --warmup_steps 20 \
    --optim "adamw_torch" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --metric_for_best_model "eval/mean_cls_accuracy" \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 81 \
    --eval_steps 81 \
    --save_total_limit 3 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --seed 8888 \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --gradient_checkpointing False \
    --torch_compile False \
    --load_best_model_at_end False \
    --full_determinism True
