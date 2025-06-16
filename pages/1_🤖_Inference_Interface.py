# pages/1_🤖_Inference_Interface.py

import streamlit as st
from utils.inference import get_predictions

# Título
st.title("🤖 Fake News Detection – Inference")

# Introducción
st.markdown("""
Esta herramienta te permite detectar si una noticia es **Falsa (0)** o **Verdadera (1)**  
usando tres modelos distintos entrenados previamente.  
Ingresa el título y/o cuerpo del artículo para obtener una predicción.
""")

# Inputs de texto
text1 = st.text_area("📝 Texto principal (por ejemplo, título o contenido)", height=150)
text2 = st.text_area("🧾 Texto secundario (opcional, por ejemplo, cuerpo de la noticia)", height=100)

# Botón de inferencia
if st.button("🔍 Predecir"):
    if not text1.strip():
        st.warning("⚠️ Por favor, introduce al menos el texto principal.")
    else:
        st.markdown("---")
        st.subheader("📊 Resultados de los modelos")

        try:
            results = get_predictions(text1, text2)

            for model_name, (label, confidence) in results.items():
                label_str = "✅ REAL" if label == 1 else "❌ FAKE" if label == 0 else str(label)
                st.markdown(f"**{model_name}** → **{label_str}** con confianza de `{confidence:.2%}`")
        except Exception as e:
            st.error(f"Error al hacer la predicción: {e}")
