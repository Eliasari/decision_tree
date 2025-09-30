import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

if len(sys.argv) < 2:
    print("Usage: python train_decision_tree.py <BlaBla.csv")
    sys.exit(0)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# Auto detect label column
candidates = [c for c in df.columns if c.lower() in
              ("target", "label", "kelas", "class", "outcome")]
label_col = candidates[0] if candidates else df.columns[-1]
print("Detected label column:", label_col)

X = pd.get_dummies(df.drop(columns=[label_col]), drop_first=True)
y_raw = df[label_col]
y = pd.factorize(y_raw)[0] if y_raw.dtype == object else y_raw

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))