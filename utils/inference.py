# utils/inference.py

import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer

# === Modelo 1: TF-IDF + Logistic Regression ===
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_model = joblib.load("models/tfidf_logistic_model.pkl")

# === Modelo 2: Sentence Transformers + Clasificador ===
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_classifier = torch.load("models/classifier.pt")  # requiere embeddings

# === Modelo 3: Hugging Face fine-tuned BERT ===
MODEL_ID = "AbyDatateo/FakeNewsClassifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def get_predictions(text1, text2=""):
    input_text = (text1.strip() + " " + text2.strip()).strip()
    results = {}

    # Modelo 1: TF-IDF
    try:
        X_tfidf = tfidf_vectorizer.transform([input_text])
        pred1 = tfidf_model.predict(X_tfidf)[0]
        score1 = max(tfidf_model.predict_proba(X_tfidf)[0])
        results["TF-IDF + Logistic Regression"] = (pred1, score1)
    except Exception as e:
        results["TF-IDF + Logistic Regression"] = f"❌ Error: {e}"

    # Modelo 2: Sentence Transformers
    try:
        emb = sentence_model.encode([input_text])
        pred2 = sentence_classifier.predict([emb])[0]
        score2 = max(sentence_classifier.predict_proba([emb])[0])
        results["Sentence Transformers"] = (pred2, score2)
    except Exception as e:
        results["Sentence Transformers"] = f"❌ Error: {e}"

    # Modelo 3: Hugging Face
    try:
        hf_result = classifier(input_text)[0]
        label3 = int(hf_result["label"].split("_")[-1])  # de LABEL_0 a 0
        score3 = hf_result["score"]
        results["Fine-tuned BERT (HuggingFace)"] = (label3, score3)
    except Exception as e:
        results["Fine-tuned BERT (HuggingFace)"] = f"❌ Error: {e}"

    return results
