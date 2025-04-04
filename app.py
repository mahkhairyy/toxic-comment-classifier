import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd

# Load multi-label model (returns toxic, insult, threat, etc.)
MODEL_NAME = "SkolkovoInstitute/roberta_toxicity_classifier"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.set_page_config(page_title="Toxic Comment Classifier (Multi-label BERT)", layout="centered")

st.markdown("## 🛡️ Toxic Comment Classifier (Multi-label BERT)")
st.caption("Using `SkolkovoInstitute/roberta_toxicity_classifier` to detect multiple forms of toxicity.")

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Prediction function
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].tolist()  # Multi-label uses sigmoid
    return dict(zip(LABELS, probs))

# Streamlit UI
text_input = st.text_area("Enter a comment:")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter a comment first.")
    else:
        result = classify_text(text_input)
        df = pd.DataFrame.from_dict(result, orient="index", columns=["Score"])
        df["Toxic"] = df["Score"] > 0.5
        st.markdown("### 🔍 Prediction Scores:")
        st.dataframe(df.style.format({"Score": "{:.2f}"}).applymap(
            lambda v: "background-color: #ffcccc" if v else "", subset=["Toxic"]
        ))
        st.markdown("✅ **Detected as toxic**: " + ", ".join([label for label, val in result.items() if val > 0.5]) or "None")
