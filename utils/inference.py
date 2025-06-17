# utils/inference.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib

# === Modelo 1: TF-IDF + Logistic Regression ===
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_model = torch.load("models/tfidf_logistic_model.pt")

# === Modelo 2: Sentence Transformers + Clasificador ===
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_classifier = torch.load("models/classifier.pt")

# === Modelo 3: Hugging Face ===
MODEL_ID = "AbyDatateo/FakeNewsClassifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# === Función de inferencia unificada ===
def get_predictions(text1, text2=""):
    input_text = (text1.strip() + " " + text2.strip()).strip()

    # ---- Modelo 1: TF-IDF ----
    X_tfidf = tfidf_vectorizer.transform([input_text])
    pred1 = tfidf_model.predict(X_tfidf)[0]
    score1 = max(tfidf_model.predict_proba(X_tfidf)[0])

    # ---- Modelo 2: Sentence Transformers ----
    emb = sentence_model.encode([input_text])
    pred2 = sentence_classifier.predict(emb)[0]
    score2 = max(sentence_classifier.predict_proba(emb)[0])

    # ---- Modelo 3: Hugging Face (corregido) ----
    hf_result = classifier(input_text)[0]
    label3 = hf_result["label"]
    # Ajuste: convertir etiquetas "LABEL_0"/"LABEL_1" en enteros
    label3_int = int(label3.split("_")[-1]) if "LABEL" in label3 else label3
    score3 = hf_result["score"]

    return {
        "TF-IDF + Logistic Regression": (pred1, score1),
        "Sentence Transformers": (pred2, score2),
        "Fine-tuned BERT (HuggingFace)": (label3_int, score3)
    }
