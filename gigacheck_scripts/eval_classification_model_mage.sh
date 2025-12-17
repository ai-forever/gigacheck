
torchrun --nnodes=1 --nproc_per_node=4 --master_port=12345 \
gigacheck/train/scripts/eval_classification_model.py \
--eval_data_path "/datasets/MAGE/test.jsonl" \
--pretrained_model_path "train_logs/mistral_7b_mage/final_model"
