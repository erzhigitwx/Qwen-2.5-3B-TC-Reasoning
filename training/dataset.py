# dataset.py
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import json

class TCDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [{"role": "system", "content": item["system"]}]
        for msg in item["messages"]:
            if msg.get("content") is not None:
                messages.append({"role": msg["role"], "content": msg["content"]})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


def collate_fn(batch, tokenizer):
    input_ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}