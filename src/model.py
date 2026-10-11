import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# LSTM model (learns language from scratch)
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim,
                 hidden_dim, n_layers, drop_prob=0.3):
        super().__init__()
        # Turns each word ID into a list of numbers; ID 0 is padding, so ignore it
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Reads the words in order and remembers context
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        # Randomly switches off neurons during training to prevent memorizing
        self.dropout = nn.Dropout(drop_prob)
        # Final layer: one score per class (negative, positive)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.long()
        embeds = self.embedding(x)            # (batch, words, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        last_step = lstm_out[:, -1, :]        # last position has read the whole review
        return self.fc(self.dropout(last_step))   # (batch, 2)


# Wraps tokenized text + labels so Hugging Face's Trainer can read them
class AmazonSentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# DistilBERT model (already pre-trained, we just fine-tune it)
def load_distilbert(model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
                attn_implementation="eager",   # lets us read attention weights later
    )
    return tokenizer, model
