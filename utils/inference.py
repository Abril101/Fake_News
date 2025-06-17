import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim * 2, 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)

def load_model_1(path="models/classifier.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    classifier = BinaryClassifier(encoder.get_sentence_embedding_dimension()).to(device)
    classifier.load_state_dict(torch.load(path, map_location=device))
    classifier.eval()

    def predict(text1, text2):
        with torch.no_grad():
            emb1 = encoder.encode([text1], convert_to_tensor=True, device=device)
            emb2 = encoder.encode([text2], convert_to_tensor=True, device=device)
            output = classifier(emb1, emb2)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            return pred, prob[0][pred].item()
    return predict

def load_model_2(path="models/tfidf_logistic_model.pkl"):
    model = joblib.load(path)
    def predict(text1, text2=None):
        combined_text = text1 if not text2 else text1 + " " + text2
        probas = model.predict_proba([combined_text])[0]
        pred = probas.argmax()
        return pred, probas[pred]
    return predict

def load_model_3(model_id="AbyDatateo/FakeNewsClassifier"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()

    def predict(text1, text2=None):
        input_text = text1 if text2 is None else f"{text1} {text2}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            return pred, confidence
    return predict

def get_predictions(text1, text2):
    model1 = load_model_1()
    model2 = load_model_2()
    model3 = load_model_3()
    return {
        "MiniLM + FFN": model1(text1, text2),
        "TF-IDF + Logistic": model2(text1, text2),
        "DistilBERT fine-tuned": model3(text1, text2)
    }

