export TOKENIZERS_PARALLELISM="false"

# Hyperparameters are set for 1GPU training
torchrun --nnodes=1 --nproc_per_node=1 --master_port=12345 \
gigacheck/train/scripts/train_detr_model.py \
    --pretrained_model_name "mistralai/Mistral-7B-v0.3" \
    --train_data_path "/datasets/ROFT-GPT/train.jsonl" \
    --eval_data_path "/datasets/ROFT-GPT/test.jsonl" \
    --num_queries 1 \
    --dec_layers 3 \
    --enc_layers 3 \
    --dn_detr True \
    --aux_loss True \
    --model_dim 256 \
    --use_focal_loss True \
    --label_loss_coef 2.0 \
    --query_initialization_method "second_half" \
    --output_dir "train_logs/mistral_7b_detr_roft_gpt" \
    --num_train_epochs 75 \
    --lr_scheduler_type "cosine" \
    --learning_rate 0.0002 \
    --weight_decay 0.0001 \
    --optim "adamw_torch" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    --eval_accumulation_steps 1 \
    --metric_for_best_model "eval_mAP@0.5-0.95" \
    --save_total_limit 2 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --seed 8888 \
    --dataloader_num_workers 8 \
    --report_to tensorboard
