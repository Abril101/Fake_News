# utils/inference.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ===============================
# Cargar modelo desde Hugging Face (una sola vez)
# ===============================

MODEL_ID = "AbyDatateo/FakeNewsClassifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# ===============================
# Función de predicción
# ===============================

def get_predictions(text1, text2=""):
    """
    Clasifica el texto como FAKE (0) o REAL (1).
    Devuelve un diccionario con el nombre del modelo y (label, confianza)
    """
    input_text = text1.strip()
    if text2.strip():
        input_text += " " + text2.strip()

    # Clasificación
    output = classifier(input_text)[0]
    label = output["label"]          # 'LABEL_0' o 'LABEL_1'
    score = output["score"]

    # Convertir a entero 0 o 1
    if "LABEL_" in label:
        label_int = int(label.replace("LABEL_", ""))
    else:
        label_int = label  # fallback por si el modelo usa nombres personalizados

    return {
        "HuggingFaceModel": (label_int, score)
    }
