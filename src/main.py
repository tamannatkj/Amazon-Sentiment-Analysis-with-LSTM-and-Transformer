import matplotlib
matplotlib.use("Agg")   # save plots to files instead of opening windows
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from utils import (load_and_preprocess_data, build_vocab, encode_and_pad,
                   compute_metrics, get_attention_map, plot_attention_heatmap)
from model import SentimentNet, AmazonSentimentDataset, load_distilbert

# ---------- Settings ----------
DATA_PATH = "data/Reviews.csv"
RESULTS_DIR = "results"
N_PER_CLASS = 500      # START SMALL to test. Later change to 10000
SEQ_LEN = 200          # LSTM: max words per review
MAX_LEN = 128          # DistilBERT: max tokens per review
LSTM_EPOCHS = 5
BERT_EPOCHS = 2
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_lstm(train_texts, train_labels, test_texts, test_labels):
    # Build the vocabulary from TRAINING text only, so no test data leaks in
    word2idx = build_vocab(train_texts)
    X_train = torch.tensor(encode_and_pad(train_texts, word2idx, SEQ_LEN))
    X_test = torch.tensor(encode_and_pad(test_texts, word2idx, SEQ_LEN))
    y_train = torch.tensor(train_labels)

    model = SentimentNet(len(word2idx), 2, 128, 128, 2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(LSTM_EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)  # stops exploding gradients
            optimizer.step()
            total_loss += loss.item()
        print(f"LSTM epoch {epoch + 1}/{LSTM_EPOCHS}, loss: {total_loss / len(loader):.4f}")

    # Evaluate on the test set
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            out = model(X_test[i:i + 256].to(device))
            preds.extend(out.argmax(dim=1).cpu().tolist())

    os.makedirs("models/lstm_sentiment", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_sentiment/model.pt")
    with open("models/lstm_sentiment/vocab.json", "w") as f:
        json.dump(word2idx, f)

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    return acc, f1


def train_distilbert(train_texts, train_labels, test_texts, test_labels):
    tokenizer, model = load_distilbert()

    train_enc = tokenizer(train_texts, truncation=True, padding=True,
                          max_length=MAX_LEN, return_tensors="pt")
    test_enc = tokenizer(test_texts, truncation=True, padding=True,
                         max_length=MAX_LEN, return_tensors="pt")
    train_ds = AmazonSentimentDataset(train_enc, torch.tensor(train_labels))
    test_ds = AmazonSentimentDataset(test_enc, torch.tensor(test_labels))

    args = TrainingArguments(
        output_dir="bert_output",
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="no",      # don't save big checkpoint files
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=test_ds, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    return tokenizer, model, metrics["eval_accuracy"], metrics["eval_f1_macro"]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data and split 80% train / 20% test
    df = load_and_preprocess_data(DATA_PATH, n_per_class=N_PER_CLASS)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"])
    train_texts = train_df["cleaned_text"].tolist()
    test_texts = test_df["cleaned_text"].tolist()
    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)} | Device: {device}")

    # 2. Train and evaluate both models
    print("\n=== Training LSTM ===")
    lstm_acc, lstm_f1 = train_lstm(train_texts, train_labels, test_texts, test_labels)

    print("\n=== Training DistilBERT ===")
    tokenizer, bert, bert_acc, bert_f1 = train_distilbert(
        train_texts, train_labels, test_texts, test_labels)

    # 3. Save the comparison
    report = (f"Model        Accuracy   F1 (macro)\n"
              f"LSTM         {lstm_acc:.4f}     {lstm_f1:.4f}\n"
              f"DistilBERT   {bert_acc:.4f}     {bert_f1:.4f}\n")
    print("\n" + report)
    with open(f"{RESULTS_DIR}/evaluation_metrics.txt", "w") as f:
        f.write(report)

    bert.save_pretrained("models/distilbert_sentiment")
    tokenizer.save_pretrained("models/distilbert_sentiment")
    
    # 4. Attention heatmaps for one sample review (DistilBERT has layers 0-5)
    bert.config.output_attentions = True
    sample = "The delivery was late and the taste was terrible, but the staff were friendly."
    for layer in range(6):
        tokens, attn = get_attention_map(sample, bert, tokenizer, layer)
        plot_attention_heatmap(tokens, attn, layer,
                               save_path=f"{RESULTS_DIR}/attention_map_layer_{layer}.png")
    print("Done! Check the results/ folder.")


if __name__ == "__main__":
    main()
    