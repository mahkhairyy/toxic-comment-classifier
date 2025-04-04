import streamlit as st
import joblib  # or pickle
import pandas as pd

#load term mmmm
# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🛡️ Toxic Comment Classifier")

# 📝 Input box FIRST
user_input = st.text_area("Enter a comment:")

# 🖱️ Button to trigger prediction
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Preprocess and predict
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("❌ Toxic Comment")
        else:
            st.success("✅ Clean Comment")

