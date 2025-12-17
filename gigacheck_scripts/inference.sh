
TEXT="""Where are we going? I asked. The old cab rattled stiffly through the icy Montclair night. Shut up, said the cabby. Shut the fuck up. He sounded stressed, scared even. He leaned over and tossed a bundle of cloth onto the backseat. Put it on. I held the fabric up in the moonlight. It was a long, beige coat, tattered in some places. I'd seen it before. A crumpled, black fedora fell down onto my lap. I picked it up, but withdrew my hand when I felt the damp inside. A passing headlight shown brightly against my palm. Blood. - If you'll play my game, keep going each post 100 words or less"""
# use one of models
classificator="train_logs/mistral_7b_mage/final_model"
detector="train_logs/mistral_7b_detr_roft_gpt/checkpoint-5307"

CUDA_VISIBLE_DEVICES="0" \
python3 gigacheck/inference/inference.py \
--text "${TEXT}" \
--model_path ${classificator}
