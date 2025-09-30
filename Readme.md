Cara menjalankan Tugas Decision Tree

Proyek ini berisi kode **Decision Tree Classifier** menggunakan Python dan Streamlit
untuk keperluan praktikum Machine Learning.

  ### 1️⃣ Clone / Download Project
  - **Jika pakai Git**
    ```bash
    git clone https://github.com/<username>/<nama-repo>.git
    cd <nama-repo>
    ```
  - **Atau download ZIP**
    Klik tombol **Code → Download ZIP** di GitHub, lalu ekstrak.

  ## 2️⃣ Buat Virtual Environment (karena saya menggunakan virtual env)
  Buka terminal (CMD/PowerShell) di folder project, jalankan:
    ```bash
    python -m venv venv
    ```
  Aktifkan environment:
  - Windows:
    ```bash
    venv\Scripts\activate
    ```
  - Mac/Linux:
    ```bash
    source venv/bin/activate

  ### 3️⃣ Install Library yang Dibutuhkan
    ```bash
    pip install -r requirements.txt
    ```

  ### 4️⃣ Jalankan Aplikasi Streamlit
    ```bash
    streamlit run streamlit_app.py 

  Browser akan otomatis terbuka ke: http://localhost:8501

  ### 5️⃣ Gunakan Aplikasi
  - Upload file CSV dataset
  - Pilih kolom target
  - Atur hyperparameter model
  - Klik "Train & Evaluate"
  - Lihat hasil evaluasi dan visualisasi
      - **Accuracy**
      - **Classification Report**
      - **Confusion Matrix**
      - **Visualisasi Decision Tree**
  - Klik **Download CSV (prediksi)** untuk mengunduh hasil prediksi.

  ## Script Lain
  Selain Streamlit, ada dua script pendukung:
  - `train_decision_tree.py`: Melatih dan mengevaluasi model langsung dari terminal
  - `decision_tree.py`: Menghasilkan paket ZIP (`decision_tree_submission`) berisi report, gambar, dan file pendukung untuk pengumpulan tugas

  Contoh menjalankan:
  ```bash
  python train_decision_tree.py BlaBla.csv
  python decision_tree.py BlaBla.csv DecisionTree_Tugas.zip
  ```
