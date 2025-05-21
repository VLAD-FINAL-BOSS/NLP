import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch


MODEL_PATH = "C:/Users/Msi/Python/NLP_Deploy/bert_toxic_classifier"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

st.title("🧠 Классификация токсичности текста (BERT)")

user_input = st.text_area("Введите текст для анализа:")

if st.button("Проверить токсичность"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        toxic_prob = probs[0][1].item()

    st.write(f"🧪 Вероятность токсичности: **{toxic_prob:.2%}**")
    if toxic_prob > 0.5:
        st.error("⚠️ Текст может быть токсичным!")
    else:
        st.success("✅ Текст выглядит безопасным.")


