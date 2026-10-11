import json
import nltk
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.append("src")
from model import SentimentNet

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


@st.cache_resource   # loads models once, not on every click
def load_models():
    with open("models/lstm_sentiment/vocab.json") as f:
        word2idx = json.load(f)
    lstm = SentimentNet(len(word2idx), 2, 128, 128, 2)
    lstm.load_state_dict(torch.load("models/lstm_sentiment/model.pt", map_location="cpu"))
    lstm.eval()
    path = "models/distilbert_sentiment"
    tok = AutoTokenizer.from_pretrained(path)
    bert = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return word2idx, lstm, tok, bert


def predict_lstm(text, word2idx, lstm, max_len=200):
    tokens = nltk.word_tokenize(text.lower())
    enc = [word2idx.get(w, 1) for w in tokens][:max_len]
    feats = np.zeros((1, max_len), dtype=int)
    if enc:
        feats[0, -len(enc):] = enc
    with torch.no_grad():
        return F.softmax(lstm(torch.from_numpy(feats)), dim=1)[0][1].item()


def predict_bert(text, tok, bert):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return F.softmax(bert(**inputs).logits, dim=1)[0][1].item()


st.title("Sentiment Analysis: LSTM vs DistilBERT")
st.write("Type a product review and compare both models.")

word2idx, lstm, tok, bert = load_models()
text = st.text_area("Review", "The coffee was great but the shipping took forever.")

if st.button("Analyze") and text.strip():
    col1, col2 = st.columns(2)
    for col, name, p in [(col1, "LSTM", predict_lstm(text, word2idx, lstm)),
                         (col2, "DistilBERT", predict_bert(text, tok, bert))]:
        label = "POSITIVE" if p >= 0.5 else "NEGATIVE"
        col.subheader(name)
        col.metric(label, f"{max(p, 1 - p):.1%} confident")
        