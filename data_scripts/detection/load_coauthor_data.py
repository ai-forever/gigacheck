# NOTE: some parts of the code are borrowed from here:
# colab.research.google.com/drive/1vbmZJlk90QWdGBWPu6yVz5ZAyww9wFzc?usp=sharing#scrollTo=Qnedud7nhHUB
import json
import os
from typing import Any, Dict, List, Tuple

import click
from gigacheck.train.src.data.data_format import create_sample_from_dict, save_samples_jsonl


def clean_text_and_adjust_intervals(text: str, ai_char_intervals: List[List[int]]):
    cleaned_text_chars = []
    position_diffs = {}  # Maps original positions to cumulative position differences
    cumulative_diff = 0

    i = 0  # Position in original text
    j = 0  # Position in cleaned text
    prev_char_in_cleaned_text = None

    while i < len(text):
        char = text[i]

        if char == "\n":
            char = " "

        if not re.match(r"[A-Za-z0-9 !\"$%&\'()\*+,-./:;?@^_`~]", char):
            cumulative_diff -= 1
            i += 1
            continue

        if char == " " and (prev_char_in_cleaned_text == " " or prev_char_in_cleaned_text is None):
            cumulative_diff -= 1
            i += 1
            continue

        cleaned_text_chars.append(char)
        position_diffs[i] = cumulative_diff
        prev_char_in_cleaned_text = char
        i += 1
        j += 1

    cleaned_text = "".join(cleaned_text_chars)

    adjusted_intervals = []
    for start, end in ai_char_intervals:
        while start < len(text) and start not in position_diffs:
            start += 1
        if start >= len(text):
            continue

        new_start = start + position_diffs.get(start, 0)

        temp_end = end - 1
        while temp_end >= 0 and temp_end not in position_diffs:
            temp_end -= 1
        if temp_end < 0:
            continue

        new_end = temp_end + 1 + position_diffs.get(temp_end, 0)

        if new_start >= new_end:
            continue

        adjusted_intervals.append([new_start, new_end])

    return cleaned_text.strip(), adjusted_intervals


def find_writing_sessions(dataset_dir: str) -> List[str]:
    paths = [os.path.join(dataset_dir, path) for path in os.listdir(dataset_dir) if path.endswith("jsonl")]
    return paths


def read_writing_session(path: str) -> List[Any]:
    events = []
    with open(path, "r") as f:
        for event in f:
            events.append(json.loads(event))
    return events


def apply_ops(doc: str, mask: str, ops: List[Dict], source: str) -> Tuple[str, str]:
    original_doc = doc
    original_mask = mask

    new_doc = ""
    new_mask = ""
    for i, op in enumerate(ops):

        # Handle retain operation
        if "retain" in op:
            num_char = op["retain"]

            retain_doc = original_doc[:num_char]
            retain_mask = original_mask[:num_char]

            original_doc = original_doc[num_char:]
            original_mask = original_mask[num_char:]

            new_doc = new_doc + retain_doc
            new_mask = new_mask + retain_mask

        # Handle insert operation
        elif "insert" in op:
            insert_doc = op["insert"]

            insert_mask = "U" * len(insert_doc)  # User
            if source == "api":
                insert_mask = "A" * len(insert_doc)  # API

            if isinstance(insert_doc, dict):
                if "image" in insert_doc:
                    print("Skipping invalid object insertion (image)")
                else:
                    print("Ignore invalid insertions:", op)
                    # Ignore other invalid insertions
                    # Debug if necessary
                    pass
            else:
                new_doc = new_doc + insert_doc
                new_mask = new_mask + insert_mask

        # Handle delete operation
        elif "delete" in op:
            num_char = op["delete"]

            if original_doc:
                original_doc = original_doc[num_char:]
                original_mask = original_mask[num_char:]
            else:
                new_doc = new_doc[:-num_char]
                new_mask = new_mask[:-num_char]

        else:
            # Ignore other operations
            # Debug if necessary
            print("Ignore other operations:", op)
            pass

    final_doc = new_doc + original_doc
    final_mask = new_mask + original_mask
    return final_doc, final_mask


def get_text_and_mask(events: Dict[Any, Any], event_id: int, remove_prompt: bool = True) -> Tuple[str, str]:
    prompt = events[0]["currentDoc"].strip()

    text = prompt
    mask = "P" * len(prompt)  # Prompt
    for event in events[:event_id]:
        if "ops" not in event["textDelta"]:
            continue
        ops = event["textDelta"]["ops"]
        source = event["eventSource"]
        text, mask = apply_ops(text, mask, ops, source)

    if remove_prompt:
        if "P" not in mask:
            print("=" * 80)
            print("Could not find the prompt in the final text")
            print("-" * 80)
            print("Prompt:", prompt)
            print("-" * 80)
            print("Final text:", text)
        else:
            end_index = mask.rindex("P")
            text = text[end_index + 1 :]
            mask = mask[end_index + 1 :]

    return text, mask


def get_label(mask: str) -> str:
    if "A" in mask and "U" in mask:
        return "mixed"
    elif "A" in mask:
        return "ai"
    elif "U" in mask:
        return "human"
    else:
        raise ValueError("No AI or User parts present in the text.")


def get_ai_char_intervals(mask: str) -> List[List[int]]:
    intervals = []
    start = None

    for i, char in enumerate(mask):
        if char == "A":
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append([start, i])
                start = None
    if start is not None:
        intervals.append([start, len(mask)])

    return intervals


def get_ai_word_intervals(text: str, mask: str) -> List[List[int]]:
    intervals = []
    current_word_start = None
    in_ai_word = False

    for i, char in enumerate(text):
        if char == " ":
            if in_ai_word:
                intervals.append([current_word_start, i - 1])
                in_ai_word = False
            current_word_start = None
        else:
            if mask[i] == "A":
                if not in_ai_word:
                    current_word_start = i if current_word_start is None else current_word_start
                in_ai_word = True
            elif in_ai_word and char != " ":
                intervals.append([current_word_start, i - 1])
                in_ai_word = False
                current_word_start = None

    if in_ai_word and current_word_start is not None:
        intervals.append([current_word_start, len(text) - 1])

    return intervals


def load_coauthor_data(dataset_dir: str, out_dir: str) -> None:
    paths = find_writing_sessions(dataset_dir)

    samples = []
    for path in paths:
        events = read_writing_session(path)
        text, mask = get_text_and_mask(events, len(events), remove_prompt=False)

        label = get_label(mask)
        ai_char_intervals = get_ai_char_intervals(mask) if label == "mixed" else []

        text, ai_char_intervals = clean_text_and_adjust_intervals(text, ai_char_intervals)

        sample_dict = {
            "label": label,
            "model": "human" if label == "human" else "gpt3",
            "text": text,
            "source": "CoAuthor",
            "source_dataset": "CoAuthor",
            "data_type": "story",
            "original_split": "all",
            "ai_char_intervals": None if len(ai_char_intervals) == 0 else ai_char_intervals,
        }

        samples.append(create_sample_from_dict(sample_dict))
    save_samples_jsonl(samples, "CoAuthor", out_dir)


@click.command()
@click.option("--dataset_dir", type=str, default="/data/data/en/coauthor/orig_from_website/coauthor-v1.0")
@click.option("--out_dir", type=str, default="/data/data/en/coauthor/jsonl_clean")
def main(dataset_dir, out_dir):
    """
    This data-loading logic is for loading from the CoAuthor dataset,
    but original split is undefined. To load data with original split refer to
    load_coauthor_data_from_excel.py
    """
    load_coauthor_data(dataset_dir, out_dir)


if __name__ == "__main__":
    main()
