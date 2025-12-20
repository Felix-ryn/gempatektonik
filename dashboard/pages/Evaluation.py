import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np

st.set_page_config(page_title="System Evaluation", layout="wide", page_icon="✅")

def main():
    st.title("✅ Evaluasi Sistem (Naive Bayes)")
    
    # --- 1. MOCKUP DATA NAIVE BAYES (Sesuai output main.py nanti) ---
    # Di real case, load ini dari file CSV/JSON hasil evaluasi
    accuracy = 0.65  # Contoh: "Meskipun akurasi dikit gapapa"
    precision = 0.68
    recall = 0.62
    
    # Matriks Konfusi (Contoh 3 Kelas: Rendah, Sedang, Tinggi)
    z = [[15, 5, 2], 
         [4, 20, 6], 
         [3, 8, 12]]
    x = ['Prediksi Rendah', 'Prediksi Sedang', 'Prediksi Tinggi']
    y = ['Aktual Rendah', 'Aktual Sedang', 'Aktual Tinggi']

    # --- 2. KPI UTAMA ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Akurasi Model", f"{accuracy*100:.1f}%", help="Akurasi Naive Bayes Evaluator")
    c2.metric("Presisi", f"{precision*100:.1f}%")
    c3.metric("Recall", f"{recall*100:.1f}%")

    st.markdown("---")

    # --- 3. VISUALISASI CONFUSION MATRIX ---
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("🧩 Confusion Matrix")
        # Menggunakan Heatmap agar terlihat 'Data Science banget'
        fig = ff.create_annotated_heatmap(
            z=z, x=x, y=y, 
            colorscale='Viridis',
            showscale=True
        )
        fig.update_layout(title_text="Perbandingan Prediksi vs Aktual")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Confusion Matrix menunjukkan di mana model sering melakukan kesalahan klasifikasi.")

    with col_r:
        st.subheader("📈 Distribusi Probabilitas")
        # Visualisasi distribusi error atau probabilitas
        dummy_prob = pd.DataFrame({
            'Kelas': ['Rendah', 'Sedang', 'Tinggi'],
            'Probabilitas': [0.3, 0.5, 0.2]
        })
        fig2 = px.bar(dummy_prob, x='Kelas', y='Probabilitas', color='Kelas', title="Distribusi Probabilitas Kelas (Sampel)")
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()