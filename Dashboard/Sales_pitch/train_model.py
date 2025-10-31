import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# ===== 1️⃣ Setup =====
MODEL_NAME = "google/flan-t5-small"  # ✅ public + lightweight
DATA_PATH = "synthetic_dataset.csv"

# ===== 2️⃣ Load dataset =====
df = pd.read_csv(DATA_PATH).fillna("")
dataset = Dataset.from_pandas(df)

# ===== 3️⃣ Tokenizer =====
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    input_text = (
        f"Product: {example['product_name']}\n"
        f"Description: {example['description']}\n"
        f"Features: {example['features']}\n"
        f"Generate sales recommendation:"
    )
    output_text = (
        f"Target audience: {example['target_audience']}. "
        f"Highlight: {example['highlight_features']}. "
        f"Sales strategy: {example['sales_strategy']}."
    )
    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    labels = tokenizer(
        output_text,
        truncation=True,
        padding="max_length",
        max_length=64
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=False)

# ===== 4️⃣ Load model =====
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Freeze most parameters for faster fine-tuning
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

# ===== 5️⃣ Training settings =====
args = TrainingArguments(
    output_dir="./sales_model",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=5,
    learning_rate=5e-4,
    disable_tqdm=True,
)

# ===== 6️⃣ Train =====
trainer = Trainer(model=model, args=args, train_dataset=tokenized)
print("🚀 Training started (should take ~2 minutes on CPU)...")
trainer.train()

# ===== 7️⃣ Save Model =====
model.save_pretrained("./sales_model")
tokenizer.save_pretrained("./sales_model")

print("✅ Training complete. Model saved to ./sales_model")
