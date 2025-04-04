import streamlit as st
import joblib  # or pickle
import pandas as pd

# Load your trained model
model = joblib.load("model.pkl")  # Save your model with joblib.dump()

st.title("🛡️ Toxic Comment Classifier")

user_input = st.text_area("Enter a comment:")
if st.button("Classify"):
    pred = model.predict([user_input])[0]
    if pred == 1:
        st.error("⚠️ Toxic Comment Detected")
    else:
        st.success("✅ Clean Comment")
