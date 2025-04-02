# 🛡️ Toxic Comment Classifier

An NLP-based machine learning project that detects toxic comments to support safer online communities. This project is part of a career shift into AI in Trust & Safety, using real-world datasets and ethical considerations to simulate how platforms handle harmful content.

---

## 📌 Project Overview
This classifier predicts whether an online comment is toxic or non-toxic. It's built using TF-IDF vectorization and logistic regression on a simplified version of the Jigsaw dataset, with a deep learning model (DistilBERT) to be added soon.

---

## 🧠 Key Concepts
- Natural Language Processing (NLP)
- Content Moderation Automation
- Trust & Safety AI
- Text Classification
- Ethical AI and Bias Awareness

---

## 🛠️ Technologies Used
- Python
- pandas, NumPy
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Matplotlib & Seaborn (for visualization)
- (Coming Soon) HuggingFace Transformers (DistilBERT)
- (Optional) Streamlit for demo UI

---

## 🧪 Dataset
Mock dataset based on the Jigsaw Toxic Comment dataset:
- `comment_text` — the user comment
- `toxic`, `severe_toxic`, `obscene`, etc. — labels for classification

For full-scale development, you can use the [original Kaggle dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

---

## 🚀 How to Run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place `mock_toxic_comments.csv` or `train.csv` in the root directory
4. Run: `python main.py`

---

## 📈 Outputs
- Classification report (precision, recall, F1)
- Confusion matrix heatmap

---

## ⚖️ Ethical Considerations
This project is for educational purposes only. Machine learning models can produce biased or inaccurate results when applied to real-world moderation. Always include human oversight and test for bias and fairness when building AI for T&S.

---

## 🙌 Credits
Created by Mahmoud Abdelsalam — aspiring AI & Trust & Safety professional with experience at TikTok and Meta.

📫 Let's connect on [LinkedIn](https://www.linkedin.com/in/mahkhairy/)!
