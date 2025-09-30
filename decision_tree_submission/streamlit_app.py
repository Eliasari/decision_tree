# Simple Streamlit app placeholder
import streamlit as st
st.title('Decision Tree Demo')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tempfile, joblib, os

st.set_page_config(page_title="Decision Tree Demo", layout="wide")
st.title("Decision Tree Demo")

st.markdown(
    "Upload CSV atau biarkan kosong untuk mencoba file lokal `BlaBla.csv` (jika ada). "
    "Pilih kolom label jika deteksi otomatis salah."
)

uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("CSV berhasil diunggah!")
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()
else:
    if os.path.exists("BlaBla.csv"):
        try:
            df = pd.read_csv("BlaBla.csv")
            st.info("Memuat `BlaBla.csv` dari working dir.")
        except Exception as e:
            st.error(f"Gagal membaca BlaBla.csv: {e}")
            st.stop()
    else:
        st.error("Tidak ada file. Upload CSV atau taruh `BlaBla.csv` di folder kerja.")
        st.stop()

st.subheader("Preview dataset (10 baris pertama)")
st.dataframe(df.head(10))

# Auto-detect label candidates but allow override
candidates = [c for c in df.columns if c.lower() in ("target","label","kelas","class","outcome","hasil","n")]
default_label = candidates[0] if candidates else df.columns[-1]
label_col = st.selectbox("Pilih kolom label/target", options=list(df.columns), index=list(df.columns).index(default_label))

st.markdown("**Model & split settings**")
test_size = st.slider("Proporsi test set", 0.1, 0.5, 0.3, 0.05)
max_depth = st.slider("Max depth (Decision Tree)", 1, 30, 5)
run = st.button("Train & Evaluate")

if run:
    # Basic checks
    if label_col not in df.columns:
        st.error("Kolom label tidak ditemukan di dataset.")
        st.stop()

    X = df.drop(columns=[label_col])
    y_raw = df[label_col]

    # Encode y if needed (keep mapping)
    if y_raw.dtype == object or y_raw.dtype.name == "category":
        y_codes, y_classes = pd.factorize(y_raw)
    else:
        y_codes = y_raw.values
        y_classes = None

    # One-hot encode X (safe, deterministic)
    X_enc = pd.get_dummies(X, drop_first=True)

    if X_enc.shape[1] == 0:
        st.error("Tidak ada fitur setelah One-Hot Encoding. Cek dataset.")
        st.stop()

    # stratify if possible
    strat = y_codes if len(np.unique(y_codes)) > 1 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y_codes, test_size=test_size, random_state=42, stratify=strat
        )
    except Exception as e:
        st.error(f"Gagal split data: {e}")
        st.stop()

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Metrics
    st.metric("Accuracy", f"{acc:.4f}")
    st.subheader("Classification report")
    st.code(report)

    # Confusion matrix
    st.subheader("Confusion matrix")
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    # Use class names if available
    if y_classes is not None:
        labels = list(y_classes)
        ax_cm.set_xticks(np.arange(len(labels)))
        ax_cm.set_xticklabels(labels, rotation=45, ha="right")
        ax_cm.set_yticks(np.arange(len(labels)))
        ax_cm.set_yticklabels(labels)
    for (i, j), val in np.ndenumerate(cm):
        ax_cm.text(j, i, val, ha="center", va="center", color="white" if val > cm.max()/2 else "black")
    fig_cm.colorbar(im)
    st.pyplot(fig_cm)

    # Decision tree visualization (try; fallback to text)
    st.subheader("Decision tree")
    try:
        fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
        class_names = list(y_classes) if y_classes is not None else [str(c) for c in np.unique(y_codes)]
        plot_tree(clf, feature_names=X_enc.columns, class_names=class_names, filled=True, fontsize=8, ax=ax_tree)
        st.pyplot(fig_tree)
    except Exception as e:
        st.warning("Gagal menggambar tree (mungkin terlalu besar). Menampilkan tree teks sebagai gantinya.")
        tree_text = export_text(clf, feature_names=list(X_enc.columns))
        st.text_area("Tree (text)", tree_text, height=400)

    # Save model to temporary file and provide download
    st.subheader("Download model")
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
        joblib.dump(clf, tmp.name)
        tmp.close()
        with open(tmp.name, "rb") as f:
            model_bytes = f.read()
        st.download_button(
            label="Download trained model (.joblib)",
            data=model_bytes,
            file_name="decision_tree.joblib",
            mime="application/octet-stream"
        )
        os.remove(tmp.name)
    except Exception as e:
        st.error(f"Gagal menyiapkan file model untuk diunduh: {e}")



