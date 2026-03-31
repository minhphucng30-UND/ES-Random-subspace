from datasets import load_dataset
from typing import Dict, List
from transformers import AutoTokenizer

ds = load_dataset("GAIR/LIMO-v2", split="train")
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def _tokenize(batch: Dict[str, List]) -> Dict[str, List[List[int]]]:
    rows = [{k: batch[k][i] for k in batch.keys()} for i in range(len(next(iter(batch.values()))))]
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    instruction_following = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    for row in rows:
        input_ids = []
        labels = []
        prompt = row["question"]
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt + instruction_following}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
        response = row['solution']
        prompt_enc = tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=False,
            max_length=1024,
        )["input_ids"]
        input_ids.extend(prompt_enc)
        labels.extend([-100] * len(prompt_enc))

        response = tokenizer(
            response + tokenizer.eos_token,
            truncation=True,
            add_special_tokens=False,
            max_length=16384,
        )["input_ids"]
        input_ids.extend(response)
        labels.extend(response)
        attention_masks = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_masks)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels}

tokenized = ds.map(_tokenize, batched=True, remove_columns=ds.column_names, num_proc=8)
import os
os.makedirs("data/LIMO", exist_ok=True)
tokenized.save_to_disk("data/LIMO")