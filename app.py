# app.py - Streamlit Toxic Comment Classifier using fine-tuned DistilBERT
import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model & tokenizer
MODEL_PATH = "toxic-distilbert"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0][1].item()

# Streamlit UI
st.set_page_config(page_title="DistilBERT Toxic Comment Classifier")
st.title("🛡️ Toxic Comment Classifier (DistilBERT)")

user_input = st.text_area("Enter a comment:")
if st.button("Classify"):
    label, score = predict(user_input)
    if label == 1:
        st.error(f"⚠️ Toxic Comment Detected (Score: {score:.2f})")
    else:
        st.success(f"✅ Clean Comment (Score: {score:.2f})")
