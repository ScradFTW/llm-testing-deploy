import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score


def load(name):
    rows = [json.loads(l) for l in open(f"{name}.jsonl")]
    return [r["title"] for r in rows], [r["genre"] for r in rows]


X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
clf.fit(Xtr, y_train)

preds = clf.predict(Xte)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
print(f"TF-IDF(char 2-4) + LogisticRegression: test_acc={acc:.4f} test_macro_f1={f1:.4f}")
print(classification_report(y_test, preds, digits=3))
