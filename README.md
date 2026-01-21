# GigaCheck

<p style="text-align: center;">
  <div align="center">
  </div>
  <p align="center">
  <a href="https://sweetdream779.github.io/LLMTrace-info"> 🌐 LLMTrace Website </a> | 
  <a href="http://arxiv.org/abs/2509.21269"> 📜 LLMTrace Paper on arXiv </a> | 
  <a href="https://arxiv.org/abs/2410.23728"> 📜 GigaChek Paper on arXiv </a> | 
  <a href="https://huggingface.co/datasets/iitolstykh/LLMTrace_detection"> 🤗 LLMTrace - Detection Dataset </a> | 
  <a href="https://huggingface.co/datasets/iitolstykh/LLMTrace_classification"> 🤗 LLMTrace - Classification Dataset </a> | 
  <a href="https://huggingface.co/iitolstykh/GigaCheck-Detector-Multi">🤗 GigaCheck detection model | </a>
  <a href="https://huggingface.co/iitolstykh/GigaCheck-Classifier-Multi">🤗 GigaCheck classification model | </a> 
</p>

### Install:

```bash
pip install -U setuptools
pip install -e . && pip install flash-attn==2.7.3 --no-build-isolation
```

### Train classification model

#### Dataset format

You need to have a dataset in '.jsonl' file. Each line in the following format: 
```
{
    "label": "human", 
    "model": "human", 
    "text": "...", 
    "data_type": "news"
}
```

#### Training

```bash
deepspeed gigacheck/train/scripts/train_classification_model.py \
    --deepspeed ${ROOT_DIR}/gigacheck/deepspeed_configs/zero2.json \
    --pretrained_model_name "mistralai/Mistral-7B-v0.3" \
    --attn_implementation "flash_attention_2" \
    --train_data_path "/data/classification/train.jsonl" \
    --eval_data_path "/data/classification/valid.jsonl" \
    --max_sequence_length 1024 \
    --min_sequence_length 100 \
    --random_sequence_length True \
    --lora_enable True \
    --lora_r 8 \
    --bf16 True \
    --output_dir "train_logs/mistral_7b_cls" \
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
```


#### Save the model with merged LoRA weights

```bash
python3 gigacheck/train/merge_lora_weights.py \
--lora_ckpt_path "train_logs/mistral_7b_cls/checkpoint-3120" \
--config_path "train_logs/mistral_7b_cls/config.json" \
--output_path "train_logs/mistral_7b_cls/final_model"
```


### Train detr on dataset with pre-trained Mistral 7b model

#### Dataset format

You need to have a dataset in '.jsonl' file. Each line in the following format: 
```
{
    "label": "mixed", 
    "model": "gpt-3.5-turbo", 
    "text": "...", 
    "data_type": "news",
    "ai_char_intervals": [[492, 1003]]
}
```

> **NOTE:** We provide the **CoAuthor** dataset converted into our specific format for training and validation. The original data (sourced from [minalee-research/coauthor-interface](https://github.com/minalee-research/coauthor-interface)) has been pre-processed to match the schema described below. You can download the ready-to-use files here: [link](https://drive.google.com/drive/folders/1jnEdqJh5mN-luh4VAy3G-8HEJuYfp9Vl).

#### Training

```bash
accelerate launch --num_processes 8 gigacheck/train/scripts/train_detr_model.py \
    --pretrained_model_name "mistralai/Mistral-7B-v0.3" \
    --train_data_path "/data/detection/train.jsonl" \
    --eval_data_path "/data/detection/valid.jsonl" \
    --extractor_dtype "bfloat16" \
    --max_sequence_length 1024 \
    --min_sequence_length 100 \
    --random_sequence_length True \
    --num_queries 45 \
    --dec_layers 3 \
    --enc_layers 3 \
    --dn_detr True \
    --aux_loss True \
    --model_dim 256 \
    --use_focal_loss True \
    --label_loss_coef 2.0 \
    --query_initialization_method "default" \
    --special_ref_points True \
    --output_dir "train_logs/mistral_7b_dn_detr" \
    --num_train_epochs 150 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.5}' \
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
    --gradient_checkpointing False \
    --report_to tensorboard
```


### Inference example

```bash
CUDA_VISIBLE_DEVICES="0" \
python3 gigacheck/inference/inference.py \
--text "${TEXT}" \
--model_path ${model_path}
```

### License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Source code, model weights and datasets are licensed under the **Apache License 2.0**. 

### Citation

If you use this repository, datasets or models in your research, please cite our papers:

```bibtex
@article{Layer2025LLMTrace,
  Title = {{LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text}},
  Author = {Irina Tolstykh and Aleksandra Tsybina and Sergey Yakubson and Maksim Kuprashevich},
  Year = {2025},
  Eprint = {arXiv:2509.21269}
}
@article{tolstykh2024gigacheck,
  title={{GigaCheck: Detecting LLM-generated Content}},
  author={Irina Tolstykh and Aleksandra Tsybina and Sergey Yakubson and Aleksandr Gordeev and Vladimir Dokholyan and Maksim Kuprashevich},
  journal={arXiv preprint arXiv:2410.23728},
  year={2024}
}
```
