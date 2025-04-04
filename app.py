import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd

# Load multi-label model
MODEL_NAME = "SkolkovoInstitute/roberta_toxicity_classifier"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Label-specific thresholds
THRESHOLDS = {
    "toxic": 0.7,
    "severe_toxic": 0.8,
    "obscene": 0.6,
    "threat": 0.5,
    "insult": 0.6,
    "identity_hate": 0.6
}

SAFE_WORDS = {"hi", "hello", "yes", "no", "thank you", "okay", "ok", "cool"}

st.set_page_config(page_title="Toxic Comment Classifier (Multi-label BERT)", layout="centered")
st.markdown("## 🛡️ Toxic Comment Classifier (Multi-label BERT)")
st.caption("Using `SkolkovoInstitute/roberta_toxicity_classifier` to detect multiple forms of toxicity.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].tolist()
    return dict(zip(LABELS, probs))

text_input = st.text_area("Enter a comment:")

if st.button("Classify"):
    text = text_input.strip().lower()
    if not text:
        st.warning("Please enter a comment first.")
        st.stop()
    elif len(text) < 3:
        st.warning("Too short to analyze reliably.")
        st.stop()
    elif text in SAFE_WORDS:
        st.success("✅ Detected as non-toxic (manually filtered)")
        st.stop()

    result = classify_text(text_input)
    df = pd.DataFrame.from_dict(result, orient="index", columns=["Score"])
    df["Toxic"] = [score > THRESHOLDS.get(label, 0.5) for label, score in result.items()]
    st.markdown("### 🔍 Prediction Scores:")
    st.dataframe(df.style.format({"Score": "{:.2f}"}).applymap(
        lambda v: "background-color: #ffcccc" if v else "", subset=["Toxic"]
    ))

    toxic_labels = [label for label, score in result.items() if score > THRESHOLDS.get(label, 0.5)]
    if toxic_labels:
        st.markdown("✅ **Detected as toxic**: " + ", ".join(toxic_labels))
    else:
        st.markdown("🟢 **No toxicity detected.**")
