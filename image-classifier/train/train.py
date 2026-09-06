import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32->16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 16->8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 8->4
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


model = SmallCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.CrossEntropyLoss()

EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    scheduler.step()
    train_acc = correct / total
    print(f"epoch {epoch:2d}  loss={total_loss/total:.4f}  train_acc={train_acc:.4f}  ({time.time()-t0:.1f}s)")

model.eval()
preds, targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().tolist())
        targets.extend(y.tolist())

test_acc = accuracy_score(targets, preds)
print(f"\nTEST ACCURACY: {test_acc:.4f}")
report = classification_report(targets, preds, target_names=CLASSES, digits=3)
print(report)
cm = confusion_matrix(targets, preds)
print("confusion matrix:")
print(cm)

majority_acc = accuracy_score(targets, [0] * len(targets))
print(f"\nmajority-class baseline: {majority_acc:.4f}")

with open("eval_report.txt", "w") as f:
    f.write(f"TEST ACCURACY: {test_acc:.4f}\n\n")
    f.write(report)
    f.write(f"\nconfusion matrix (rows=true, cols=pred), classes={CLASSES}\n{cm}\n")
    f.write(f"\nmajority-class baseline: {majority_acc:.4f}\n")

# --- export to ONNX for lightweight CPU-only production inference ---
model.eval()
dummy = torch.randn(1, 3, 32, 32, device=device)
torch.onnx.export(
    model, dummy, "image_classifier.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13,
)
print("\nexported image_classifier.onnx")

# sanity-check ONNX gives the same predictions as the PyTorch model
import onnxruntime as ort
sess = ort.InferenceSession("image_classifier.onnx", providers=["CPUExecutionProvider"])
sample_x = next(iter(test_loader))[0][:8].numpy()
onnx_preds = sess.run(None, {"image": sample_x})[0].argmax(1)
with torch.no_grad():
    torch_preds = model(torch.tensor(sample_x, device=device)).argmax(1).cpu().numpy()
assert (onnx_preds == torch_preds).all(), "ONNX/PyTorch prediction mismatch!"
print("ONNX export verified: predictions match PyTorch exactly on a sample batch")

np.save("normalize_mean.npy", np.array([0.4914, 0.4822, 0.4465], dtype=np.float32))
np.save("normalize_std.npy", np.array([0.2470, 0.2435, 0.2616], dtype=np.float32))
with open("classes.txt", "w") as f:
    f.write("\n".join(CLASSES))
