import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, ConcatDataset
from training.dataset import SFTDataset, collate_fn

class Checkpoint:
    def __init__(self, model, path="../checkpoints"):
        self.model = model
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save(self, step, optimizer, loss):
        entries = sorted(
            [e for e in os.listdir(self.path) if e.startswith("ckpt_")],
            key=lambda x: int(x.split("_")[1])
        )

        if len(entries) >= 2:
            os.remove(os.path.join(self.path, entries[0]))
            print(f"Deleted old checkpoint: {entries[0]}")

        ckpt_path = os.path.join(self.path, f"ckpt_{step}")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "loss": loss.item(),
        }, ckpt_path)
        print(f"Saved checkpoint: ckpt_{step}")

    def load_latest(self):
        entries = sorted(
            [e for e in os.listdir(self.path) if e.startswith("ckpt_")],
            key=lambda x: int(x.split("_")[1])
        )
        if not entries:
            return None
        return torch.load(os.path.join(self.path, entries[-1]), map_location="cuda")

class Model:
    def __init__(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        model_name = "Qwen/Qwen2.5-3B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        self.checkpoint = Checkpoint(self.model)
        self.start_step = 0

        ckpt = self.checkpoint.load_latest()
        if ckpt:
            self.model.load_state_dict(ckpt["model"], strict=False)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_step = ckpt["step"]
            print(f"Resumed from step {self.start_step}")
        else:
            print("No checkpoint found, starting from scratch")

    def train(self):
        dataset_tc = SFTDataset("../data/processed/tool_calling.json", self.tokenizer)
        dataset_rs = SFTDataset("../data/processed/reasoning.json", self.tokenizer)
        combined = ConcatDataset([dataset_tc, dataset_rs])
        loader = DataLoader(combined, batch_size=4, shuffle=True, collate_fn=lambda batch: collate_fn(batch, self.tokenizer))
        GRAD_ACCUM = 4

        self.model.train()
        for step, batch in enumerate(loader):
            global_step = self.start_step + step

            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"Step {global_step} | Loss {loss.item():.4f}")
            if global_step % 50 == 0 and global_step > 0:
                self.checkpoint.save(global_step, self.optimizer, loss)

    def forward(self):
        return self.model

    def get_feature(self, *fields):
        result = {f: getattr(self, f) for f in fields}
        return result[fields[0]] if len(fields) == 1 else result

if __name__ == "__main__":
    model = Model()
    model.train()