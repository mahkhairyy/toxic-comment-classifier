# Toxic Comment Classifier - AI in Trust & Safety Project

# 📌 Project Goal:
# Build a model that can classify comments as toxic or non-toxic using NLP techniques.

# 🧰 Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 📥 Load Dataset
# Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Place the CSV in your working directory as 'train.csv'
df = pd.read_csv('/train[1].csv')

# 📊 Quick Look at the Data
os.makedirs("results", exist_ok=True)  # ✅ Creates the folder if it doesn't exist
print(df.head())
print(df.columns)

# 🧹 Preprocessing
# Combine all toxicity labels into a single binary column
df['toxic_label'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].max(axis=1)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

df['clean_comment'] = df['comment_text'].astype(str).apply(clean_text)

# 🧪 Split Dataset
X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['toxic_label'], test_size=0.2, random_state=42)

# 🔠 TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🤖 Train Model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# 📈 Evaluation
y_pred = model.predict(X_test_vec)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ✅ Print 100 Sample Predictions
test_results = pd.DataFrame({
    'Comment': X_test.reset_index(drop=True),
    'Actual': y_test.reset_index(drop=True),
    'Predicted': y_pred
})
sample = test_results.head(100)
correct = (sample['Actual'] == sample['Predicted']).sum()
accuracy = (correct / len(sample)) * 100

print(f"\n✅ Sample Accuracy: {correct} out of 100 correct → {accuracy:.2f}%")
print(test_results.head(100))  # or .to_csv(...) to save

# (Optional) Save to CSV
os.makedirs("results", exist_ok=True)
test_results.head(100).to_csv("results/sample_predictions.csv", index=False)
print("✅ File saved to: results/sample_predictions.csv")



# 🔍 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
