# 🏠 SmartHomePrice-Machine-Learning-untuk-Prediksi-Harga-Rumah

📌 Konsep Proyek
Proyek ini mengimplementasikan dua model machine learning (regresi dan klasifikasi) dalam satu aplikasi desktop untuk memprediksi harga rumah berdasarkan berbagai fitur properti. Sistem ini memberikan prediksi harga numerik (regresi) sekaligus kategorisasi harga (klasifikasi) dalam satu pipeline terintegrasi.

🎯 Alur Machine Learning
📊 1. Persiapan Data
python
# Dataset: 1000 sampel rumah dengan 6 fitur
Fitur: area, bedrooms, bathrooms, age, location_score, garage
Target: price (regresi), price_category (klasifikasi: Low/Medium/High)

# Preprocessing:
- Splitting data (80% training, 20% testing)
- StandardScaler untuk normalisasi
- LabelEncoder untuk target kategorikal
🤖 2. Model Regression (Prediksi Harga)
Algoritma yang dibandingkan:

Linear Regression - Model linear sederhana

Random Forest Regressor - Ensemble dengan 100 trees

Metrik Evaluasi:

RMSE (Root Mean Squared Error): Mengukur error dalam satuan harga

R² Score: Mengukur seberapa baik model menjelaskan variasi data

Hasil: Random Forest memberikan performa lebih baik dengan R² score ~98%

🏷️ 3. Model Classification (Kategori Harga)
Algoritma: Random Forest Classifier
Target: 3 kelas (Low/Medium/High)

Metrik Evaluasi:

Accuracy: ~95%

Classification Report (precision, recall, F1-score)

💾 4. Model Deployment
Model disimpan menggunakan joblib:

regression_model.pkl - Model prediksi harga

classification_model.pkl - Model kategorisasi

scaler_*.pkl - Normalizer untuk masing-masing model

label_encoder.pkl - Encoder untuk kategori harga

🚀 Arsitektur Sistem
text
Data Input → Preprocessing → 
├── Regression Model → Prediksi Harga Numerik
└── Classification Model → Kategori Harga
Penanganan Data Penting:
Feature Engineering:

Semua fitur numerik

Tidak ada missing values

Skala fitur dinormalisasi

Stratified Splitting:

Untuk klasifikasi, menggunakan stratify agar distribusi kelas tetap

Feature Importance:

Area dan location_score menjadi fitur paling berpengaruh

🖥️ Aplikasi Desktop (Tkinter)
Fitur Interface:

Input data rumah dengan 6 parameter

Prediksi real-time untuk kedua model

Tampilan hasil dengan format yang jelas

Interpretasi kategori harga

Reset dan validasi input

Struktur Folder:

text
ML_Project_Praktikum/
├── data/              # Dataset house_prices.csv
├── models/           # Model yang disimpan (.pkl)
├── images/           # Visualisasi hasil EDA
├── notebook/         # Jupyter Notebook development
└── app.py           # Aplikasi desktop utama
📈 Hasil dan Performa
Model Regression:
Best Model: Random Forest Regressor

R² Score: 0.98+ (98% variansi data dapat dijelaskan)

RMSE: ~Rp 45,000 (error rata-rata)

Model Classification:
Accuracy: 95%+

Confusion Matrix: Minimal misklasifikasi antar kategori

🔧 Teknologi yang Digunakan
python
# Core ML Libraries:
pandas, numpy, scikit-learn, matplotlib, seaborn

# Model Algorithms:
LinearRegression, RandomForestRegressor, RandomForestClassifier

# Evaluation Metrics:
mean_squared_error, r2_score, accuracy_score, classification_report

# Deployment:
joblib (model persistence), tkinter (GUI), PIL (image handling)
🎨 Visualisasi yang Dihasilkan
EDA Visualization: Distribusi data, scatter plot, heatmap korelasi

Regression Results: Actual vs Predicted, residual plot, feature importance

Classification Results: Confusion matrix, feature importance

💡 Kelebihan Sistem
Dual Prediction: Satu input menghasilkan dua jenis prediksi

Interpretable: Feature importance membantu memahami faktor penentu harga

User-Friendly: Interface desktop yang intuitif

Scalable: Mudah ditambahkan model atau fitur baru

Reproducible: Random state diatur untuk konsistensi

📋 Cara Penggunaan
Isi form input data rumah

Klik "PREDIKSI HARGA"

Sistem akan menampilkan:

Prediksi harga dalam Rupiah

Kategori harga (Low/Medium/High)

Interpretasi hasil
