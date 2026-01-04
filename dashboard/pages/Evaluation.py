import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import json
import os

st.set_page_config(page_title="System Evaluation", layout="wide", page_icon="✅")

def load_real_metrics():
    """
    Fungsi untuk membaca file JSON hasil output dari evaluation_engine.py
    """
    # Mengatur path relatif agar bisa menemukan folder 'output'
    # Asumsi struktur folder:
    # root/
    #   dashboard/pages/Evaluation.py
    #   output/system_metrics.json
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Naik 2 level dari dashboard/pages ke root, lalu masuk ke output
    file_path = os.path.join(current_dir, "..", "..", "output", "system_metrics.json")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"Gagal membaca file metrics: {e}")
            return None
    else:
        return None

def main():
    st.title("✅ Evaluasi Sistem (Real-Time Metrics)")
    
    # --- 1. LOAD DATA DARI FILE JSON (Bukan Mockup Lagi) ---
    data = load_real_metrics()
    
    # Default values jika file belum ada atau error (Fallback)
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    labels = ['Rendah', 'Sedang', 'Tinggi'] # Default label
    
    if data:
        st.success(f"Data Evaluasi Terakhir: {data.get('timestamp', 'Unknown')}")
        
        # Ambil metrics dari JSON
        metrics = data.get("metrics", {})
        
        # Mapping data JSON ke variabel dashboard
        # Menggunakan 'accuracy' langsung
        accuracy = metrics.get("accuracy", 0.0)
        
        # Menggunakan 'weighted avg' atau 'macro avg' untuk precision/recall
        avg_metrics = metrics.get("weighted avg", {})
        precision = avg_metrics.get("precision", 0.0)
        recall = avg_metrics.get("recall", 0.0)
        
        # Ambil Confusion Matrix
        cm_data = data.get("confusion_matrix", [])
        if cm_data:
            z = cm_data
        
        # Ambil Labels jika ada
        if "labels" in data:
            labels = data["labels"]
    else:
        st.warning("⚠️ File 'output/system_metrics.json' belum ditemukan. Jalankan backend/engine terlebih dahulu.")

    # Siapkan Label untuk Visualisasi
    x_labels = [f'Prediksi {l}' for l in labels]
    y_labels = [f'Aktual {l}' for l in labels]

    # --- 2. KPI UTAMA ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Akurasi Model", f"{accuracy*100:.1f}%", help="Akurasi Real-time dari Engine")
    c2.metric("Presisi (Weighted)", f"{precision*100:.1f}%")
    c3.metric("Recall (Weighted)", f"{recall*100:.1f}%")

    st.markdown("---")

    # --- 3. VISUALISASI CONFUSION MATRIX ---
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("🧩 Confusion Matrix")
        
        # Validasi dimensi Z agar sesuai dengan label
        if len(z) == len(x_labels):
            fig = ff.create_annotated_heatmap(
                z=z, x=x_labels, y=y_labels, 
                colorscale='Viridis',
                showscale=True
            )
            fig.update_layout(title_text="Perbandingan Prediksi vs Aktual")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Dimensi Confusion Matrix tidak sesuai dengan jumlah label.")
            
        st.caption("Data diambil langsung dari hasil training terakhir.")

    with col_r:
        st.subheader("📈 Performa Per Kelas")
        
        # Membuat DataFrame dari metrics JSON untuk grafik batang
        if data and "metrics" in data:
            # Filter hanya key yang merupakan nama kelas (bukan accuracy/macro avg/weighted avg)
            class_metrics = []
            for k, v in data["metrics"].items():
                if k in labels:  # Pastikan k adalah nama kelas (RINGAN, SEDANG, PARAH)
                    class_metrics.append({
                        "Kelas": k,
                        "F1-Score": v.get("f1-score", 0)
                    })
            
            if class_metrics:
                df_metrics = pd.DataFrame(class_metrics)
                fig2 = px.bar(df_metrics, x='Kelas', y='F1-Score', color='Kelas', 
                              title="F1-Score per Kelas", range_y=[0, 1])
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Menunggu data klasifikasi per kelas...")
        else:
             # Dummy visual jika data kosong
            dummy_prob = pd.DataFrame({
                'Kelas': labels,
                'Probabilitas': [0.0] * len(labels)
            })
            fig2 = px.bar(dummy_prob, x='Kelas', y='Probabilitas', title="Data Belum Tersedia")
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()