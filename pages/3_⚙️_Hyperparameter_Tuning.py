import streamlit as st

st.set_page_config(layout="wide")
st.title("⚙️ Hyperparameter Tuning")

st.markdown("""
In this section, we demonstrate how hyperparameters were optimized for two of the models:

1. TF-IDF + Logistic Regression
2. MiniLM + Feedforward Neural Network (FFN)

We used two different tuning strategies: **GridSearchCV** for the logistic model and **Optuna** for the transformer-based FFN.
""")

# —————————————————————————————————————————
# 🔍 Logistic Regression + TF-IDF
# —————————————————————————————————————————
st.subheader("🔍 TF-IDF + Logistic Regression")

st.markdown("""
We performed a **grid search** over 162 hyperparameter combinations using 3-fold cross-validation.

### 🔧 Parameters Tuned:
- `C`: Regularization strength
- `solver`: Optimization algorithm (`liblinear`, `saga`, etc.)
- `tfidf__max_df`: Maximum document frequency (e.g., 0.5, 0.7, 1.0)
- `tfidf__min_df`: Minimum document frequency (e.g., 1, 3, 5)
- `tfidf__max_features`: Vocabulary size (e.g., 5000, 10000)

### 🏆 Best Configuration:
```python
{
  'clf__C': 10,
  'clf__solver': 'saga',
  'tfidf__max_df': 0.7,
  'tfidf__max_features': 10000,
  'tfidf__min_df': 5
}
Best Cross-Validation Accuracy: 0.9927

✅ Test Set Accuracy: 0.9931

This shows that even traditional models can achieve high accuracy when well-tuned.


🤖 MiniLM + FFN with Optuna
""")
st.subheader("🤖 MiniLM + FFN – Optuna Optimization")

st.markdown("""
We used Optuna, a powerful hyperparameter optimization framework, to tune the transformer-based FFN.

🔧 Parameters Tuned:
batch_size: Number of samples per gradient step

lr (learning rate): Step size for optimizer updates

We ran 10 optimization trials. Each trial trained the model for 2 epochs and evaluated on validation data.

🏆 Best Trial:
python
Copiar
Editar
{
  'batch_size': 16,
  'lr': 1.4114564760209243e-05
}
✅ Best Accuracy Achieved: 1.0
""")

st.image("models/viz/optuna_accuracy_plot.png", caption="Optuna Trial Performance (Accuracy)", use_column_width=True)

st.markdown("""
The plot above shows how validation accuracy changed across different trials.

📌 Observations:
Trials with small learning rates (1e-5 to 3e-5) achieved the best performance.

Some configurations underperformed (e.g., larger batch sizes or poorly tuned learning rates).

Model is sensitive to hyperparameter settings, especially due to small data size.

Optuna helped us efficiently find a performant configuration.


✅ Conclusion
st.subheader("✅ Summary")
            """)

st.markdown("""

Both models benefited from proper hyperparameter tuning.

Grid search was effective for the logistic baseline.

Optuna provided a flexible and automated search for the neural model.

These efforts directly improved model accuracy and reliability.
""")