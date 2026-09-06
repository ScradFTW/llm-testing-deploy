import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline


def load(name):
    rows = [json.loads(l) for l in open(f"{name}.jsonl")]
    return [r["title"] for r in rows], [r["genre"] for r in rows]


X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
])

# Final model trains on train+val (val was only used earlier to compare
# against the neural-net alternative); test stays untouched for reporting.
pipeline.fit(X_train + X_val, y_train + y_val)

preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
labels = sorted(set(y_train))

print(f"FINAL MODEL: test_acc={acc:.4f} test_macro_f1={f1:.4f}\n")
report = classification_report(y_test, preds, digits=3)
print(report)

cm = confusion_matrix(y_test, preds, labels=labels)
print("labels:", labels)
print(cm)

majority = max(set(y_train), key=y_train.count)
maj_acc = accuracy_score(y_test, [majority] * len(y_test))
print(f"\nmajority-class baseline ('{majority}'): test_acc={maj_acc:.4f}")

joblib.dump(pipeline, "genre_pipeline.joblib")

with open("eval_report.txt", "w") as f:
    f.write(f"FINAL MODEL: test_acc={acc:.4f} test_macro_f1={f1:.4f}\n\n")
    f.write(report)
    f.write(f"\nconfusion matrix labels: {labels}\n{cm}\n")
    f.write(f"\nmajority-class baseline ('{majority}'): test_acc={maj_acc:.4f}\n")

print("\nsaved genre_pipeline.joblib + eval_report.txt")
