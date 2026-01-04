import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CNN Prediction Dashboard",
    layout="wide",
    page_icon="🧭"
)

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Mengarah ke folder output/cnn_results
CSV_PATH = os.path.join(CURRENT_DIR, "../../output/cnn_results/cnn_next_earthquake_prediction.csv")

# --- FUNGSI LOAD DATA ---
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# --- FUNGSI CEK KONSISTENSI (LOGIC BARU) ---
def check_consistency(arah_label, sudut):
    """
    Mengecek apakah Label Arah (Teks) sinkron dengan Sudut Derajat (Angka).
    Sistem Navigasi: 0=Utara, 90=Timur, 180=Selatan, 270=Barat.
    """
    # Normalisasi sudut 0-360
    s = sudut % 360
    
    # Tentukan Zona berdasarkan Sudut
    if (s >= 315 or s < 45): 
        zona_sudut = "Utara"
    elif (s >= 45 and s < 135): 
        zona_sudut = "Timur"
    elif (s >= 135 and s < 225): 
        zona_sudut = "Selatan"
    else: 
        zona_sudut = "Barat"
        
    # Bandingkan dengan Label Prediksi
    if arah_label == "Unknown":
        return True, zona_sudut
        
    # Jika Label sama dengan Zona Sudut, berarti Sinkron
    if arah_label == zona_sudut:
        return True, zona_sudut
    else:
        return False, zona_sudut

# --- VISUALISASI KOMPAS YANG BENAR ---
def plot_compass_fixed(angle, label, is_consistent, real_zone):
    fig = go.Figure()

    # Warna jarum: Hijau jika konsisten, Oranye jika konflik
    needle_color = "green" if is_consistent else "orange"
    
    # Jarum Penunjuk
    fig.add_trace(go.Scatterpolar(
        r=[0, 1],
        theta=[0, angle],
        mode='lines+markers',
        line=dict(color=needle_color, width=5),
        marker=dict(size=14, symbol='arrow-bar-up', color=needle_color),
        name='Prediksi Model'
    ))

    # Layout Kompas Navigasi (0 di Atas, Putar Kanan/Clockwise)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 90, 180, 270],
                ticktext=['U (Utara)', 'T (Timur)', 'S (Selatan)', 'B (Barat)'],
                direction='clockwise', # Wajib Clockwise untuk Kompas
                rotation=90, # 0 Derajat di Atas (Utara)
                tickfont=dict(size=12, color="black")
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        title=dict(
            text=f"Arah Jarum: {angle:.1f}° ({real_zone})",
            y=0.95
        )
    )
    return fig

# --- MAIN APP ---
def main():
    st.title("CNN: Analisis Arah Perambatan")
    st.markdown("""
    **Status:** Simple CNN (Adaptive) | **Output:** Arah (Klasifikasi) & Sudut (Regresi)
    """)

    if st.button("🔄 Segarkan Data Real-Time"):
        st.rerun()

    df = load_data()

    if df is not None and not df.empty:
        # Ambil Data Terakhir
        last_row = df.iloc[-1]
        
        # Ekstrak Variabel
        ts = last_row.get('timestamp', '-')
        arah_lbl = last_row.get('arah_prediksi', 'Unknown')
        sudut = float(last_row.get('arah_derajat', 0.0))
        
        # [FIX CONFIDENCE] Mengubah 0.28 jadi 28.0
        conf_raw = float(last_row.get('confidence_scalar', 0.0))
        if conf_raw <= 1.0: 
            conf_pct = conf_raw * 100 # Jika desimal (0.28), kali 100
        else:
            conf_pct = conf_raw # Jika sudah ratusan, biarkan

        risk_arr = last_row.get('risk_k_array', '[]')

        if conf_pct > 99.9:
            conf_pct = 99.9
        
        # Cek Konsistensi Model (Label vs Sudut)
        is_consistent, zone_real = check_consistency(arah_lbl, sudut)

        # --- LAYOUT METRICS ---
        m1, m2, m3 = st.columns(3)
        m1.metric("📍 Label Arah (Teks)", arah_lbl, delta="Hasil Klasifikasi", delta_color="off")
        m2.metric("📐 Sudut (Angka)", f"{sudut:.2f}°", delta=f"Zona {zone_real}")
        m3.metric("🛡️ Confidence", f"{conf_pct:.2f}%", help="Probabilitas prediksi (Max 100%)")

        st.divider()

        # --- VISUALISASI ---
        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.info(f"🕒 **Waktu Prediksi:**\n{ts}")
            
            st.markdown("### 🔍 Analisis Konsistensi")
            if is_consistent:
                st.success(f"**SINKRON:** Model yakin arahnya **{arah_lbl}** dan sudutnya **{sudut:.1f}°** (keduanya menunjuk arah yang sama).")
            else:
                st.warning(f"**KONFLIK DATA:**")
                st.markdown(f"""
                - Model Klasifikasi memilih teks: **"{arah_lbl}"**
                - Model Regresi menghitung sudut: **{sudut:.1f}° ({zone_real})**
                
                *Penyebab: Model masih belajar menyeimbangkan kedua output ini.*
                """)

            with st.expander("Lihat Array Risiko (Input GA)"):
                st.code(risk_arr, language="json")

        with c_right:
            # Plot Kompas yang sudah diperbaiki rotasinya
            fig = plot_compass_fixed(sudut, arah_lbl, is_consistent, zone_real)
            st.plotly_chart(fig, use_container_width=True)

        # --- TABEL DATA ---
        st.markdown("### 📜 Riwayat Data")
        st.dataframe(df[['timestamp', 'arah_prediksi', 'arah_derajat', 'confidence_scalar']].sort_index(ascending=False), use_container_width=True)

    else:
        st.warning("Data CSV belum tersedia. Jalankan `TectonicOrchestrator` dulu.")

if __name__ == "__main__":
    main()