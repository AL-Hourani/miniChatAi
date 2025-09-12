import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "flax-community/arabic-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        q = item["question"].strip()
                        a = item["answer"].strip()
                        if q and a:
                            self.data.append((q, a))
                    except:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        inputs = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            answer,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels_ids = labels.input_ids.squeeze()
        labels_ids[labels_ids == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels_ids}

# 3. تحميل البيانات
# -----------------------------
dataset = QADataset("traning_final_cleaned.jsonl", tokenizer)
print(f"📂 Loaded dataset with {len(dataset)} samples")


training_args = TrainingArguments(
    output_dir="./miniChat",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)


trainer.train()
trainer.save_model("./miniChat")
tokenizer.save_pretrained("./miniChat")
print("تم حفظ النموذج ")



