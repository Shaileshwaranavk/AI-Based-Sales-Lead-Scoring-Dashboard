import pandas as pd
from Dashboard.models import Customer, CustomerReview
from Dashboard.model_pipeline import train_and_predict_with_ml
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

def retrain_models():
    """
    Retrain ML & LLM models using CRM customer data + reviews.
    """
    # 1️⃣ Export data from CRM
    customers = Customer.objects.all().values()
    reviews = CustomerReview.objects.all().values()

    if not customers:
        return "❌ No customer data to train."

    df_customers = pd.DataFrame(customers)
    df_reviews = pd.DataFrame(reviews)
    if not df_reviews.empty:
        df_reviews = df_reviews.groupby("customer_id")["review_text"].apply(lambda x: " ".join(x)).reset_index()
        df = pd.merge(df_customers, df_reviews, on="customer_id", how="left")
    else:
        df = df_customers.copy()

    df["review_text"] = df["review_text"].fillna("")

    # 2️⃣ Use reviews to adjust Conversion_Rate heuristically
    df["Conversion_Rate"] = df["conversion_rate"] + df["review_text"].apply(lambda x: 5 if "good" in x.lower() else 0)

    # 3️⃣ Save dataset snapshot for reproducibility
    df.to_csv("crm_training_data.csv", index=False)

    # 4️⃣ Retrain ML model
    print("🔁 Retraining ML model...")
    train_and_predict_with_ml(df, df, "Generic Product", "Auto-learned from CRM", "CRM data", top_n=5)

    # 5️⃣ Retrain LLM model (T5) for sales recommendations
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    dataset = Dataset.from_pandas(df[["product_interested", "review_text", "conversion_rate"]].fillna(""))

    def preprocess(example):
        input_text = f"Product: {example['product_interested']}\nReview: {example['review_text']}\nGenerate recommendation:"
        output_text = f"Predicted Conversion Rate: {example['conversion_rate']}"
        model_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
        labels = tokenizer(output_text, truncation=True, padding="max_length", max_length=64)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=False)
    args = TrainingArguments(output_dir="./crm_sales_model", num_train_epochs=1, per_device_train_batch_size=2, save_strategy="no")
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained("./crm_sales_model")
    tokenizer.save_pretrained("./crm_sales_model")
    print("✅ CRM-based models retrained successfully.")
    return "✅ CRM retraining complete."
