# utils/inference.py

import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer

# === Modelo 1: TF-IDF + Logistic Regression ===
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_model = joblib.load("models/tfidf_logistic_model.pkl")

# === Modelo 2: Sentence Transformers + Clasificador ===
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_classifier = torch.load("models/classifier.pt")  # ← CORRECTO: es .pt y usa torch

# === Modelo 3: Hugging Face fine-tuned BERT ===
MODEL_ID = "AbyDatateo/FakeNewsClassifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# === Función principal de inferencia ===
def get_predictions(text1, text2=""):
    input_text = (text1.strip() + " " + text2.strip()).strip()

    # Modelo 1: TF-IDF
    X_tfidf = tfidf_vectorizer.transform([input_text])
    pred1 = tfidf_model.predict(X_tfidf)[0]
    score1 = max(tfidf_model.predict_proba(X_tfidf)[0])

    # Modelo 2: Sentence Transformers
    embedding = sentence_model.encode([input_text])
    pred2 = sentence_classifier.predict([embedding])[0]
    score2 = max(sentence_classifier.predict_proba([embedding])[0])

    # Modelo 3: Hugging Face BERT
    hf_result = classifier(input_text)[0]
    label3 = int(hf_result["label"].replace("LABEL_", ""))
    score3 = hf_result["score"]

    return {
        "TF-IDF + Logistic Regression": (pred1, score1),
        "Sentence Transformers": (pred2, score2),
        "Fine-tuned BERT (HuggingFace)": (label3, score3)
    }
