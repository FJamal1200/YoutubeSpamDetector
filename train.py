import re
import torch
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from model import SpamClassifier


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text.split()


def build_vocab(text, max_words=10000):
    counter = Counter()

    for text in text:
        counter.update(clean_text(text))

    most_common= counter.most_common(max_words)
    vocab = {word: i for i, (word, count) in enumerate(most_common)}

    return vocab

def vectorize(text, vocab):
    vector = torch.zeros(len(vocab))

    for word in clean_text(text):
        if word in vocab:
            vector[vocab[word]] = 1

    return vector

df = pd.read_csv('data.csv')

vocab = build_vocab(df.text)

texts = df["text"].tolist()
labels = df["label"].tolist()

X = torch.stack([vectorize(text, vocab) for text in texts])
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SpamClassifier(vocab_size=len(vocab))

lossS_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    logits = model(x_train)
    loss = lossS_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

torch.save({
    "model_state": model.state_dict(),
    "vocab": vocab
}, "spam_model.pt")

print("Model saved as spam_model.pt")


