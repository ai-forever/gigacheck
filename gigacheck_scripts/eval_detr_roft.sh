
torchrun --nnodes=1 --nproc_per_node=1 --master_port=12345 \
gigacheck/train/scripts/eval_detr_model.py \
--eval_data_path "/datasets/ROFT-GPT/test.jsonl" \
--pretrained_model_path "train_logs/mistral_7b_detr_roft_gpt/checkpoint-5307"
