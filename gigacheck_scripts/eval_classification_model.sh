
torchrun --nnodes=1 --nproc_per_node=8 --master_port=12345 \
gigacheck/train/scripts/eval_classification_model.py \
--eval_data_path "/data/classification/bilingual/test.jsonl" \
--pretrained_model_path "train_logs/mistral_7b_bilingual_dataset/final_model_checkpoint_14616"
