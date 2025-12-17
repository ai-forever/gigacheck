
torchrun --nnodes=1 --nproc_per_node=8 --master_port=12346 \
gigacheck/train/scripts/eval_detr_model.py \
--eval_data_path "/data/detection/bilingual/test.jsonl" \
--pretrained_model_path "train_logs/dn_detr_bilingual/checkpoint-15943"
