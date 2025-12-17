
export TOKENIZERS_PARALLELISM="false"

deepspeed gigacheck/train/scripts/train_classification_model.py \
    --deepspeed gigacheck/deepspeed_configs/zero2.json \
    --pretrained_model_name "mistralai/Mistral-7B-v0.3" \
    --attn_implementation "flash_attention_2" \
    --train_data_path "/datasets/MAGE/train.jsonl" \
    --eval_data_path "/datasets/MAGE/valid.jsonl" \
    --max_sequence_length 1024 \
    --min_sequence_length 900 \
    --random_sequence_length True \
    --lora_enable True \
    --lora_r 8 \
    --bf16 True \
    --output_dir "train_logs/mistral_7b_mage" \
    --num_train_epochs 20 \
    --lr_scheduler_type "cosine" \
    --learning_rate 0.0003 \
    --warmup_steps 20 \
    --optim "adamw_torch" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --metric_for_best_model "eval/mean_cls_accuracy" \
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    --save_total_limit 3 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --seed 8888 \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --load_best_model_at_end True \
    --ce_weights 1 2

python3 gigacheck/train/scripts/merge_lora_weights.py \
--lora_ckpt_path "train_logs/mistral_7b_mage/checkpoint-3120" \
--config_path "train_logs/mistral_7b_mage/config.json" \
--output_path "train_logs/mistral_7b_mage/final_model"
