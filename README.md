# 📰 Fake News Detection

Esta app está hecha para detectar si una noticia es **falsa (0)** o **real (1)** usando modelos de machine learning entrenados previamente.  
Todo el análisis se presenta en una interfaz sencilla hecha con Streamlit, donde puedes probar los modelos directamente con cualquier texto.

---

## 📌 ¿Qué incluye?

- **Interfaz de predicción** con tres modelos distintos
- **Análisis exploratorio** del dataset original
- **Visualización de métricas**: matriz de confusión, clasificación, etc.
- **Tuning de hiperparámetros** (por página separada)
- Dataset de [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 🧠 Modelos usados

- TF-IDF + Logistic Regression
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Fine-tuned BERT (`bert-base-uncased` con SimpleTransformers)

---

## 🚀 Cómo correrlo

Puedes probar la app directamente online desde Streamlit Cloud.  
Si quieres correrla local:

```bash
pip install -r requirements.txt
streamlit run app.py
```


## Modelo entrenado
El modelo fue subido a Hugging Face y puede encontrarse aquí:  
[https://huggingface.co/AbyDatateo/FakeNewsClassifier](https://huggingface.co/AbyDatateo/FakeNewsClassifier)

