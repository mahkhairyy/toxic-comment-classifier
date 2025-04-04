# train.py - Fine-tune DistilBERT for Toxic Comment Classification
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
df = pd.read_csv("https://raw.githubusercontent.com/mahkhairy/datasets/main/jigsaw/train_sample.csv")
df["label"] = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].max(axis=1)

# Balance dataset
toxic = df[df["label"] == 1]
non_toxic = df[df["label"] == 0].sample(len(toxic), random_state=42)
df = pd.concat([toxic, non_toxic]).sample(frac=1).reset_index(drop=True)

# Prepare HuggingFace Datasets
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"])
train_ds = Dataset.from_pandas(train_df[["comment_text", "label"]])
val_ds = Dataset.from_pandas(val_df[["comment_text", "label"]])

# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize(example): return tokenizer(example["comment_text"], truncation=True, padding=True)
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Training setup
args = TrainingArguments(
    output_dir="toxic-distilbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"]
    }

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Save
model.save_pretrained("toxic-distilbert")
tokenizer.save_pretrained("toxic-distilbert")
print("✅ Model and tokenizer saved to /toxic-distilbert")
