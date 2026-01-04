import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
import sys

# Cek dan import library
try:
    import joblib
    print("✅ joblib imported successfully")
except ImportError:
    print("❌ joblib tidak ditemukan. Install dengan: pip install joblib")
    sys.exit(1)

class HousePricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 House Price Predictor - Machine Learning")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f4f8')
        
        print("🚀 Aplikasi dimulai...")
        print(f"📁 Working directory: {os.getcwd()}")
        
        # Cek folder models
        if not os.path.exists('models'):
            print("❌ Folder 'models' tidak ditemukan!")
            messagebox.showerror(
                "Error", 
                "Folder 'models' tidak ditemukan!\n\n"
                "Silakan jalankan Jupyter Notebook terlebih dahulu untuk:\n"
                "1. Generate dataset\n"
                "2. Train models\n"
                "3. Save models ke folder 'models/'"
            )
            self.models_loaded = False
        else:
            print("✅ Folder 'models' ditemukan")
            self.load_models()
        
        self.setup_ui()
        print("✅ UI setup selesai")
    
    def load_models(self):
        """Load semua models dengan error handling"""
        self.models_loaded = True
        errors = []
        
        model_files = {
            'regression_model': 'models/regression_model.pkl',
            'classification_model': 'models/classification_model.pkl',
            'scaler_reg': 'models/scaler_regression.pkl',
            'scaler_class': 'models/scaler_classification.pkl',
            'label_encoder': 'models/label_encoder.pkl'
        }
        
        for name, path in model_files.items():
            try:
                if not os.path.exists(path):
                    errors.append(f"❌ File tidak ditemukan: {path}")
                    self.models_loaded = False
                else:
                    setattr(self, name, joblib.load(path))
                    print(f"✅ {name} loaded successfully")
            except Exception as e:
                errors.append(f"❌ Error loading {name}: {str(e)}")
                self.models_loaded = False
        
        if errors:
            error_msg = "\n".join(errors)
            error_msg += "\n\n⚠️  Silakan jalankan Jupyter Notebook untuk train dan save models!"
            messagebox.showwarning("Warning", error_msg)
            print(error_msg)
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🏠 House Price Predictor",
            font=('Helvetica', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f4f8')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Input
        left_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        input_title = tk.Label(
            left_panel,
            text="📝 Input Data Rumah",
            font=('Helvetica', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        input_title.pack(pady=15)
        
        # Input fields
        input_frame = tk.Frame(left_panel, bg='white')
        input_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.entries = {}
        fields = [
            ("🏘️ Luas Area (m²)", "area", 500, 5000, 2500),
            ("🛏️ Jumlah Kamar Tidur", "bedrooms", 1, 5, 3),
            ("🚿 Jumlah Kamar Mandi", "bathrooms", 1, 4, 2),
            ("📅 Umur Bangunan (tahun)", "age", 0, 50, 10),
            ("📍 Skor Lokasi (1-10)", "location_score", 1, 10, 7),
            ("🚗 Jumlah Garasi", "garage", 0, 3, 1)
        ]
        
        for i, (label, key, min_val, max_val, default) in enumerate(fields):
            frame = tk.Frame(input_frame, bg='white')
            frame.pack(fill=tk.X, pady=8)
            
            lbl = tk.Label(
                frame,
                text=label,
                font=('Helvetica', 11),
                bg='white',
                fg='#34495e',
                anchor='w'
            )
            lbl.pack(fill=tk.X)
            
            entry = ttk.Entry(frame, font=('Helvetica', 11))
            entry.pack(fill=tk.X, pady=(5, 0))
            entry.insert(0, str(default))
            self.entries[key] = entry
        
        # Buttons
        button_frame = tk.Frame(left_panel, bg='white')
        button_frame.pack(pady=20)
        
        predict_btn = tk.Button(
            button_frame,
            text="🎯 PREDIKSI HARGA",
            command=self.predict_price,
            font=('Helvetica', 12, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            cursor='hand2',
            width=20,
            height=2
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🔄 RESET",
            command=self.clear_inputs,
            font=('Helvetica', 12, 'bold'),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            cursor='hand2',
            width=15,
            height=2
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        result_title = tk.Label(
            right_panel,
            text="📊 Hasil Prediksi",
            font=('Helvetica', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        result_title.pack(pady=15)
        
        # Result display
        self.result_frame = tk.Frame(right_panel, bg='white')
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.result_text = tk.Text(
            self.result_frame,
            font=('Courier', 11),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief=tk.FLAT,
            wrap=tk.WORD,
            height=15
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Instructions if models not loaded
        if not self.models_loaded:
            instructions = """
╔════════════════════════════════════════════════╗
         ⚠️  MODELS BELUM TERSEDIA  ⚠️
╚════════════════════════════════════════════════╝

📝 LANGKAH UNTUK MEMULAI:

1. Buka Jupyter Notebook:
   py -m notebook

2. Buka file: ML_Project_Development.ipynb

3. Jalankan semua cell (Cell → Run All)

4. Tunggu sampai models tersimpan di folder
   'models/'

5. Close aplikasi ini dan jalankan lagi

╚════════════════════════════════════════════════╝
"""
            self.result_text.insert('1.0', instructions)
        else:
            welcome = """
╔════════════════════════════════════════════════╗
         ✅ APLIKASI SIAP DIGUNAKAN!
╚════════════════════════════════════════════════╝

Silakan masukkan data rumah di panel kiri,
lalu klik tombol "PREDIKSI HARGA" untuk melihat
hasil prediksi.

Model yang digunakan:
  • Random Forest Regressor (Harga)
  • Random Forest Classifier (Kategori)

╚════════════════════════════════════════════════╝
"""
            self.result_text.insert('1.0', welcome)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        status_text = "✅ Models loaded - Ready to predict" if self.models_loaded else "❌ Models not found - Run Jupyter Notebook first"
        self.status_label = tk.Label(
            status_frame,
            text=status_text,
            font=('Helvetica', 9),
            bg='#34495e',
            fg='white'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def get_input_values(self):
        try:
            values = []
            for key in ['area', 'bedrooms', 'bathrooms', 'age', 'location_score', 'garage']:
                value = float(self.entries[key].get())
                values.append(value)
            return np.array(values).reshape(1, -1)
        except ValueError:
            messagebox.showerror("Error", "Mohon masukkan nilai numerik yang valid!")
            return None
    
    def predict_price(self):
        if not self.models_loaded:
            messagebox.showerror(
                "Error", 
                "Models belum tersedia!\n\n"
                "Silakan jalankan Jupyter Notebook terlebih dahulu."
            )
            return
        
        input_data = self.get_input_values()
        if input_data is None:
            return
        
        try:
            print(f"📊 Input data: {input_data}")
            
            # Regression prediction
            input_scaled_reg = self.scaler_reg.transform(input_data)
            predicted_price = self.regression_model.predict(input_scaled_reg)[0]
            print(f"💰 Predicted price: {predicted_price:,.0f}")
            
            # Classification prediction
            input_scaled_class = self.scaler_class.transform(input_data)
            predicted_class_encoded = self.classification_model.predict(input_scaled_class)[0]
            predicted_category = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            print(f"🏷️  Predicted category: {predicted_category}")
            
            # Display results
            self.display_results(input_data[0], predicted_price, predicted_category)
            
            self.status_label.config(text=f"✅ Prediksi berhasil! Harga: Rp {predicted_price:,.0f}")
            
        except Exception as e:
            error_msg = f"Terjadi kesalahan saat prediksi:\n{str(e)}"
            messagebox.showerror("Error", error_msg)
            print(f"❌ {error_msg}")
    
    def display_results(self, input_data, price, category):
        self.result_text.delete('1.0', tk.END)
        
        # Format price
        price_formatted = f"Rp {price:,.0f}"
        
        # Category color
        category_colors = {
            'Low': '🟢',
            'Medium': '🟡',
            'High': '🔴'
        }
        category_icon = category_colors.get(category, '⚪')
        
        result = f"""
╔═══════════════════════════════════════════════╗
              HASIL PREDIKSI                    
╚═══════════════════════════════════════════════╝

💰 PREDIKSI HARGA (REGRESSION)
   {price_formatted}

🏷️  KATEGORI HARGA (CLASSIFICATION)
   {category_icon} {category}

───────────────────────────────────────────────

📋 DATA INPUT:
   
   🏘️  Luas Area      : {input_data[0]:.0f} m²
   🛏️  Kamar Tidur    : {input_data[1]:.0f}
   🚿 Kamar Mandi    : {input_data[2]:.0f}
   📅 Umur Bangunan  : {input_data[3]:.0f} tahun
   📍 Skor Lokasi    : {input_data[4]:.0f}/10
   🚗 Garasi         : {input_data[5]:.0f}

───────────────────────────────────────────────

💡 INTERPRETASI:
"""
        
        if category == 'Low':
            result += "   Rumah ini termasuk kategori harga rendah.\n"
            result += "   Cocok untuk pembeli dengan budget terbatas.\n"
        elif category == 'Medium':
            result += "   Rumah ini termasuk kategori harga menengah.\n"
            result += "   Ideal untuk keluarga kecil hingga menengah.\n"
        else:
            result += "   Rumah ini termasuk kategori harga tinggi.\n"
            result += "   Properti premium dengan fasilitas lengkap.\n"
        
        result += "\n╚═══════════════════════════════════════════════╝"
        
        self.result_text.insert('1.0', result)
    
    def clear_inputs(self):
        defaults = {
            'area': 2500,
            'bedrooms': 3,
            'bathrooms': 2,
            'age': 10,
            'location_score': 7,
            'garage': 1
        }
        
        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(defaults[key]))
        
        welcome = """
╔════════════════════════════════════════════════╗
         ✅ INPUT DIRESET
╚════════════════════════════════════════════════╝

Silakan masukkan data rumah baru dan klik
tombol "PREDIKSI HARGA" untuk melihat hasil.

╚════════════════════════════════════════════════╝
"""
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', welcome)
        self.status_label.config(text="✅ Input direset")
        print("🔄 Input direset")

def main():
    print("="*60)
    print("🏠 HOUSE PRICE PREDICTOR - MACHINE LEARNING APP")
    print("="*60)
    
    root = tk.Tk()
    app = HousePricePredictorApp(root)
    
    print("✅ Aplikasi berjalan...")
    print("📌 Tutup window untuk exit\n")
    
    root.mainloop()
    print("\n👋 Aplikasi ditutup")

if __name__ == "__main__":
    main()