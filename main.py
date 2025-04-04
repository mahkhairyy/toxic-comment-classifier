# Toxic Comment Classifier - AI in Trust & Safety Project

# 📌 Project Goal:
# Build a model that can classify comments as toxic or non-toxic using NLP techniques.

# 🧰 Required Librariesfrom transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import torch.nn.functional as F
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 📥 Load Dataset
df = pd.read_csv('train.csv')

# 📊 Quick Look at the Data
os.makedirs("results", exist_ok=True)
print(df.head())
print(df.columns)

# 🧹 Preprocessing
df['toxic_label'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].max(axis=1)
df['comment_text'] = df['comment_text'].astype(str)

# 🧪 Split Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic_label'], test_size=0.2, random_state=42)

# 🤖 Load Pretrained BERT Model
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 📈 Inference Function
def predict_toxicity(texts, threshold=0.5):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    probs = F.softmax(outputs.logits, dim=1)
    preds = (probs[:, 1] > threshold).int().tolist()
    return preds, probs[:, 1].tolist()

# Predict on test set
batch_size = 32
predictions = []
scores = []

for i in range(0, len(X_test), batch_size):
    batch_texts = X_test.iloc[i:i+batch_size].tolist()
    batch_preds, batch_scores = predict_toxicity(batch_texts)
    predictions.extend(batch_preds)
    scores.extend(batch_scores)

# 📈 Evaluation
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ✅ Sample Predictions
test_results = pd.DataFrame({
    'Comment': X_test.reset_index(drop=True),
    'Actual': y_test.reset_index(drop=True),
    'Predicted': predictions,
    'Toxicity_Score': scores
})
sample = test_results.head(100)
correct = (sample['Actual'] == sample['Predicted']).sum()
accuracy = (correct / len(sample)) * 100
print(f"\n✅ Sample Accuracy: {correct} out of 100 correct → {accuracy:.2f}%")

# Save sample predictions
test_results.head(100).to_csv("results/sample_predictions_bert.csv", index=False)
print("✅ File saved to: results/sample_predictions_bert.csv")

# 🔍 Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
