import streamlit as st

st.set_page_config(layout="wide")
st.title("📈 Model Analysis and Justification")

# —————————————————————————————————————————
# 🔍 Model Motivation and Overview
# —————————————————————————————————————————
st.subheader("🧠 Model Motivation and Selection")

st.markdown("""
This project compares three different modeling strategies for detecting fake news:

1. **TF-IDF + Logistic Regression** – A fast, explainable traditional machine learning model.
2. **MiniLM + Feedforward Neural Network (FFN)** – Uses semantic sentence embeddings and a lightweight classifier.
3. **Fine-Tuned DistilBERT** – A transformer model trained end-to-end on our dataset.

Each approach was selected to balance accuracy, interpretability, and resource usage.
""")

# —————————————————————————————————————————
# 📋 TF-IDF + Logistic Regression
# —————————————————————————————————————————
st.subheader("📋 TF-IDF + Logistic Regression")

st.markdown("**Classification Report:**")
st.code("""
              precision    recall  f1-score   support

        Fake       0.99      0.98      0.99      4710
        True       0.98      0.99      0.99      4270

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980
""")

st.image("models/viz/confusion_matrix_t.png", caption="TF-IDF Confusion Matrix", use_column_width=True)

st.markdown("""
### 🔍 Error Analysis:
- **False Positives**:
  - "BREAKING: IRAN THROWS DOWN ULTIMATUM..."
  - "OBAMA USES LABOR DAY TO HARM PRIVATE SECTOR..."
- **False Negatives**:
  - "Trump promises tax relief..."
  - "Comey accuses Trump administration of defaming him"

- ✅ **False Positives:** 74  
- ✅ **False Negatives:** 49

**Conclusion:** This traditional model provides fast and robust predictions, with high accuracy even without deep learning.
""")


# —————————————————————————————————————————
# 🧪 MiniLM + FFN – Final Evaluation
# —————————————————————————————————————————
st.subheader("🧪 MiniLM + FFN – Final Evaluation (classifier.pt)")

st.markdown("""
We evaluated the final `classifier.pt` model using 100 MiniLM-encoded sentence pairs. The final classification report and confusion matrix are shown below:
""")

st.image("models/viz/distribucion_clases_classification.png", caption="Classification Report & Confusion Matrix – MiniLM Final", use_column_width=True)

st.markdown("""
### ✅ Results:
- Accuracy: `99.5%`
- Precision and recall: `1.00` for class 0, `0.99` for class 1
- Misclassifications: only 2 false negatives

### 🔍 Observations:
- The model handles balanced data very well.
- Misclassifications occurred on real news headlines with ambiguous or vague tone.
- Could benefit from further data augmentation or contrastive training.

**Conclusion:** A solid compromise between efficiency and accuracy.
""")

# —————————————————————————————————————————
# 🧠 Fine-Tuned DistilBERT
# —————————————————————————————————————————
st.subheader("🧠 Fine-Tuned DistilBERT")

st.markdown("""
DistilBERT was fine-tuned using Hugging Face's `Trainer` API. The model learns directly from raw text without precomputed embeddings.
""")

st.image("models/viz/report_distilbert.png", caption="Classification Report & Confusion Matrix – DistilBERT", use_column_width=True)

st.markdown("""
### ✅ Results:
- Accuracy: `1.00`
- Only 2 misclassifications out of 13,469 samples

### 🔍 Observations:
- Extremely strong performance — possibly overfitting.
- May require evaluation on external datasets to verify robustness.

**Conclusion:** Best performance among all models, but less explainable and more resource intensive.
""")

# —————————————————————————————————————————
# ✅ Conclusion
# —————————————————————————————————————————
st.subheader("✅ Overall Comparison")

st.markdown("""
| Model                  | Accuracy | Comments |
|------------------------|----------|--------------------------------------------------------|
| TF-IDF + Logistic      | 99%      | Fast, interpretable, performs extremely well           |
| MiniLM + FFN (CV)      | 89%      | High variance in folds, good tuning potential          |
| MiniLM + FFN (Final)   | 99.5%    | Very strong, low error, sensitive to hyperparameters   |
| DistilBERT Fine-Tuned  | 100%     | Best overall, may overfit, costly to train             |

---

**Final Thoughts:**
- TF-IDF still shines for simplicity and speed.
- DistilBERT is the most accurate but least efficient.
- A possible ensemble (TF-IDF + MiniLM) could blend speed and generalization.

This evaluation justifies the use of transformer models in NLP pipelines while also appreciating the efficiency of classical techniques.
""")
