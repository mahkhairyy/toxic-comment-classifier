import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load BERT model
MODEL_NAME = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Prediction function
def predict_toxicity(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    toxic_score = probs[0][1].item()
    label = "Toxic" if toxic_score > threshold else "Not Toxic"
    return label, toxic_score

# Streamlit UI
st.title("🛡️ Toxic Comment Classifier (BERT)")
comment = st.text_area("Enter a comment:")

if st.button("Classify"):
    label, score = predict_toxicity(comment)
    color = "red" if label == "Toxic" else "green"
    st.markdown(f"### Prediction: **:{color}[{label}]**")
    st.caption(f"Toxicity Score: {score:.2f}")
