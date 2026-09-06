import json
import re
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

random.seed(42)
torch.manual_seed(42)

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load(name):
    return [json.loads(l) for l in open(f"{name}.jsonl")]


train_rows = load("train")
val_rows = load("val")
test_rows = load("test")

labels = sorted({r["genre"] for r in train_rows})
label2id = {l: i for i, l in enumerate(labels)}
print("labels:", labels)

# --- vocab ---
counter = Counter()
for r in train_rows:
    counter.update(tokenize(r["title"]))

VOCAB_SIZE = 6000
most_common = [w for w, _ in counter.most_common(VOCAB_SIZE - 2)]
word2id = {"<pad>": 0, "<unk>": 1}
for w in most_common:
    word2id[w] = len(word2id)
print("vocab size:", len(word2id))

MAX_LEN = 16


def encode(title):
    ids = [word2id.get(t, 1) for t in tokenize(title)][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


class GenreDataset(Dataset):
    def __init__(self, rows):
        self.x = torch.tensor([encode(r["title"]) for r in rows], dtype=torch.long)
        self.y = torch.tensor([label2id[r["genre"]] for r in rows], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds = GenreDataset(train_rows)
val_ds = GenreDataset(val_rows)
test_ds = GenreDataset(test_rows)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256)
test_dl = DataLoader(test_ds, batch_size=256)


class GenreClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, num_classes=len(labels)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        emb = self.emb(x) * mask
        pooled = emb.sum(1) / mask.sum(1).clamp(min=1)
        h = torch.relu(self.fc1(pooled))
        h = self.dropout(h)
        return self.fc2(h)


model = GenreClassifier(len(word2id))

# inverse-frequency class weights to counter the Hip-Hop / Reggae / Jazz imbalance
train_label_counts = Counter(r["genre"] for r in train_rows)
weights = torch.tensor(
    [1.0 / train_label_counts[l] for l in labels], dtype=torch.float32
)
weights = weights / weights.sum() * len(labels)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(dl):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            preds.extend(logits.argmax(1).tolist())
            targets.extend(y.tolist())
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    return acc, macro_f1, preds, targets


best_val_f1 = -1
best_state = None
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    train_loss = total_loss / len(train_ds)
    val_acc, val_f1, _, _ = evaluate(val_dl)
    print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)

test_acc, test_f1, preds, targets = evaluate(test_dl)
print(f"\nBEST MODEL (by val macro-F1={best_val_f1:.4f})")
print(f"test_acc={test_acc:.4f}  test_macro_f1={test_f1:.4f}\n")
print(classification_report(targets, preds, target_names=labels, digits=3))

cm = confusion_matrix(targets, preds)
print("confusion matrix (rows=true, cols=pred):")
print("labels:", labels)
print(cm)

# majority-class baseline, for an honest comparison
majority_label = Counter(r["genre"] for r in train_rows).most_common(1)[0][0]
maj_preds = [label2id[majority_label]] * len(targets)
maj_acc = accuracy_score(targets, maj_preds)
print(f"\nmajority-class baseline ('{majority_label}') test_acc={maj_acc:.4f}")

# --- export for lightweight (numpy-only) CPU inference ---
np.savez(
    "genre_model.npz",
    emb=model.emb.weight.detach().numpy(),
    fc1_w=model.fc1.weight.detach().numpy(),
    fc1_b=model.fc1.bias.detach().numpy(),
    fc2_w=model.fc2.weight.detach().numpy(),
    fc2_b=model.fc2.bias.detach().numpy(),
)
json.dump(
    {"word2id": word2id, "labels": labels, "max_len": MAX_LEN},
    open("genre_vocab.json", "w"),
)
print("\nexported genre_model.npz + genre_vocab.json")
