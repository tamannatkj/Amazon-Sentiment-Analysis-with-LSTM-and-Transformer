import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import nltk
from collections import Counter
from tqdm import tqdm
from evaluate import load

def load_and_preprocess_data(data_path, n_per_class=10000):
    """Loads Reviews.csv, converts star ratings to positive/negative labels,
    cleans the text, and returns a balanced sample."""
    data = pd.read_csv(data_path)
    data = data[["Text", "Score"]]
    data.dropna(inplace=True)

    # Remove neutral 3-star reviews
    data = data[data["Score"] != 3]

    # 1-2 stars = negative (0), 4-5 stars = positive (1)
    data["label"] = data["Score"].map({1: 0, 2: 0, 4: 1, 5: 1})

    data["cleaned_text"] = data["Text"].apply(clean_text)
    data.drop_duplicates(subset="cleaned_text", keep="first", inplace=True)

    # Dataset is mostly positive, so take equal numbers of each class
    data = data.groupby("label").sample(n=n_per_class, random_state=42)
    return data.reset_index(drop=True)

# Clean Text
def clean_text(text):
    text = str(text)

    text = text.lower()   # lowercase
    text = re.sub(r'<.*?>', ' ', text)   # remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces and tabs

    return text


# LSTM Padding
def pad_input(sentences, seq_len):
    features = np.zeros(
        (len(sentences), seq_len), dtype=int
    )  # instances * features (seq_len)
    for idx, review in enumerate(tqdm(sentences)):
        if len(review) != 0:
            # Pre-padding
            features[idx, -len(review) :] = np.array(review)[:seq_len]
    return features

# Building Vocab
def build_vocab(texts, min_freq=1):
    """根据训练文本构建词汇表"""
    words = Counter()
    for sentence in tqdm(texts, desc="Building Vocab"):
        for word in nltk.word_tokenize(str(sentence).lower()):
            words.update([word])

    words = {k: v for k, v in words.items() if v > min_freq}
    words = sorted(words, key=words.get, reverse=True)
    words = ["_PAD", "_UNK"] + words

    word2idx = {o: i for i, o in enumerate(words)}
    return word2idx


def encode_and_pad(texts, word2idx, seq_len):
    features = np.zeros((len(texts), seq_len), dtype=int)

    for idx, sentence in enumerate(tqdm(texts, desc="Encoding")):
        clean_sent = str(sentence).lower()
        tokens = nltk.word_tokenize(clean_sent)

        # 转换为索引，如果找不到则用 _UNK (1)
        encoded = [word2idx.get(word, 1) for word in tokens]

        if len(encoded) != 0:
            # 前向填充 (Pre-padding)，与 Notebook 保持一致
            features[idx, -len(encoded):] = np.array(encoded)[:seq_len]

    return features

def compute_metrics(p):

    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.argmax(preds, axis=1)

    metric_acc = load("accuracy")
    metric_f1 = load("f1")

    accuracy = metric_acc.compute(predictions=preds, references=p.label_ids)
    f1_macro = metric_f1.compute(predictions=preds, references=p.label_ids, average="macro")

    return {
        'accuracy': accuracy['accuracy'],
        'f1_macro': f1_macro['f1']
    }


# Visualize the Transformer's attention mechanism
def get_attention_map(text, model, tokenizer, layer_num):
    """
    layer_num: 0 means the first layer，-1 means the last layer
               For DistilBERT，layer: 1 - 5
    """
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions

    num_layers = len(attentions)
    if layer_num < -num_layers or layer_num >= num_layers:
        raise ValueError(
            f"The layer index is out of range. The model has {num_layers} layers，Please input an integer between - {num_layers} and {num_layers - 1}.")

    selected_layer_attention = attentions[layer_num]

    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    selected_layer_attention = selected_layer_attention.squeeze(0)
    attention_matrix = selected_layer_attention.mean(dim=0).cpu().numpy()

    return tokens, attention_matrix


def plot_attention_heatmap(tokens, attention_matrix, layer_num, save_path=None):
    plt.figure(figsize=(10, 8))

    # Plot heatmap using Seaborn
    # xticklabels and yticklabels correspond to tokens
    sns.heatmap(attention_matrix,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="Blues",
                square=True,
                cbar_kws={'label': 'Attention Weight'})

    layer_name = "Last Layer" if layer_num == -1 else f"Layer {layer_num}"

    plt.title(f"Self-Attention Map ({layer_name}, Avg over Heads)", fontsize=16)
    plt.xlabel("Key (Source Token)", fontsize=12)
    plt.ylabel("Query (Target Token)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f" Heatmap saved to {save_path}")

    plt.close()
