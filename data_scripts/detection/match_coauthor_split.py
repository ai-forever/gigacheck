import json
import re

import click
from fuzzywuzzy import fuzz, process
from tqdm import tqdm


def clean_string(input_string):
    input_string = re.sub(r"\n", " ", input_string)
    input_string = re.sub(r"[^A-Za-z0-9 !\"$%&\'()\*+,-./:;?@^_`~]", "", input_string)
    input_string = re.sub(r"[ ]+", " ", input_string)
    input_string = input_string.strip()

    return input_string


def modify_original_split(coauthor_file, coauthor_split_file, output_path):
    with open(coauthor_split_file, "r", encoding="utf-8") as f2:
        data2 = [json.loads(line) for line in f2]

    with open(coauthor_file, "r", encoding="utf-8") as f1, open(output_path, "w", encoding="utf-8") as output_file:
        num_no_matches = 0
        total_lines = sum(1 for _ in f1)
        f1.seek(0)
        for line in tqdm(f1, total=total_lines, desc="Processing entries"):
            entry1 = json.loads(line)
            cleaned_text1 = clean_string(entry1["text"])

            cleaned_texts2 = [(clean_string(entry2["text"]), entry2) for entry2 in data2]

            # Use fuzzy matching to find the best match in data2 for the whole cleaned text
            # Using process.extractOne to get the top 1 match based on fuzzy ratio
            best_match = process.extractOne(cleaned_text1, [text for text, _ in cleaned_texts2], scorer=fuzz.ratio)

            if best_match:
                matched_text, score = best_match
                matched_entry = next(entry for text, entry in cleaned_texts2 if text == matched_text)

                entry1["original_split"] = matched_entry["original_split"]

            else:
                print(f"No matches found for: '{cleaned_text1}'")
                num_no_matches += 1

            output_file.write(json.dumps(entry1, ensure_ascii=False) + "\n")

    print(f"Number of texts with 0 matches: {num_no_matches}")


@click.command()
@click.option("--coauthor_file", type=str, default="/data/data/en/coauthor/jsonl_clean/CoAuthor.jsonl")
@click.option(
    "--coauthor_split_file", type=str, default="/data/data/en/coauthor/orig_split_by_text/CoAuthor_train.jsonl"
)
@click.option("--output_path", type=str, default="/data/data/en/coauthor/jsonl_clean/CoAuthorOrigSplit.jsonl")
def main(coauthor_file, coauthor_split_file, output_path):
    modify_original_split(coauthor_file, coauthor_split_file, output_path)


if __name__ == "__main__":
    main()
