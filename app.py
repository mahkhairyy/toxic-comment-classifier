import streamlit as st
import joblib  # or pickle
import pandas as pd

# Load both model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Inside your Streamlit button logic
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        vectorized = vectorizer.transform([user_input])
        pred = model.predict(vectorized)[0]

        if pred == 1:
            st.error("❌ Toxic Comment")
        else:
            st.success("✅ Clean Comment")
