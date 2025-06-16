import streamlit as st

st.title("📊 Visualización del Dataset")

st.markdown("Análisis exploratorio del dataset usado para entrenar los modelos.")

st.image("models/viz/distribucion_clase.png", caption="Distribución de Clases")
st.image("models/viz/longitud_texto.png", caption="Distribución de Longitud de Texto")
st.image("models/viz/wordcloud_fake.png", caption="WordCloud – Noticias Falsas")
st.image("models/viz/wordcloud_real.png", caption="WordCloud – Noticias Reales")
