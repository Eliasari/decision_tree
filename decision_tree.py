import os, shutil, zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def make_package(csv_path: str, out_zip: str = "DecisionTree.zip",
                 max_depth: int = 5, test_size: float = 0.3):
    out_dir = "decision_tree_submission"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # === Load dataset ===
    df = pd.read_csv(csv_path)
    # Auto detect label column
    candidates = [c for c in df.columns if c.lower() in
                  ("target", "label", "kelas", "class", "outcome")]
    label_col = candidates[0] if candidates else df.columns[-1]

    X = pd.get_dummies(df.drop(columns=[label_col]), drop_first=True)
    y_raw = df[label_col]
    y, y_classes = pd.factorize(y_raw) if y_raw.dtype == object else (y_raw.values, None)

    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=strat
    )

    # === Train model ===
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # === Save textual report ===
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Label column: {label_col}\nAccuracy: {acc:.4f}\n\n")
        if y_classes is not None:
            f.write("Label mapping:\n")
            for i, cls in enumerate(y_classes):
                f.write(f"{i} -> {cls}\n")
            f.write("\n")
        f.write(report)

    # === Dataset preview PNG ===
    fig, ax = plt.subplots(figsize=(12,4))   
    ax.axis('off')
    table = ax.table(
        cellText=df.head(10).values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)        
    table.scale(1.2, 1.2)        
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "dataset_preview.png"),
     dpi=300                  
    )
    plt.close(fig)


    # === Confusion matrix PNG ===
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha="center", va="center",
                color="white" if val > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close(fig)

    # === Decision tree PNG ===
    fig, ax = plt.subplots(figsize=(24,16))
    plot_tree(clf, feature_names=X.columns,
              class_names=(y_classes if y_classes is not None
                           else [str(c) for c in np.unique(y)]),
              filled=True, fontsize=10, ax=ax)
    plt.tight_layout()
    plt.savefig(
    os.path.join(out_dir, "decision_tree.png"),
    dpi=300         
    )
    plt.close(fig)

    # === Classification report PNG ===
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axis('off')
    ax.text(0, 1, report, fontsize=10, family='monospace', va='top')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "classification_report.png"))
    plt.close(fig)

        # === Streamlit placeholder ===
    with open(os.path.join(out_dir, "streamlit_app.py"), "w") as f:
        f.write("# Simple Streamlit app placeholder\n")
        f.write("import streamlit as st\nst.title('Decision Tree Demo')\n")

    # === Requirements ===
    with open(os.path.join(out_dir, "requirements.txt"), "w") as f:
        f.write("pandas\nscikit-learn\nmatplotlib\nstreamlit\n")

    # === Zip everything ===
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for file in files:
                full = os.path.join(root, file)
                arc = os.path.relpath(full, out_dir)
                zf.write(full, arc)
    print(f"Done. ZIP created: {out_zip}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python decision_tree.py <dataset.csv> [out_zip_name]")
    else:
        csv = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else "DecisionTree.zip"
        make_package(csv, out)
    