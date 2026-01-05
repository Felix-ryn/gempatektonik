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
# 1. Path Output Prediksi CNN
CSV_PRED_PATH = os.path.join(CURRENT_DIR, "../../output/cnn_results/cnn_next_earthquake_prediction.csv")
# 2. Path Input Data Mentah (Untuk visualisasi H-1 ke H)
INPUT_DATA_PATH = os.path.join(CURRENT_DIR, "../../output/lstm_results/lstm_data_for_cnn.xlsx")

# --- FUNGSI LOAD DATA ---
def load_prediction_data():
    if not os.path.exists(CSV_PRED_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PRED_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading Prediction CSV: {e}")
        return None

def load_input_data():
    """Load data mentah untuk melihat posisi H dan H-1"""
    if not os.path.exists(INPUT_DATA_PATH):
        # Fallback coba baca CSV jika excel ga ada
        csv_fallback = INPUT_DATA_PATH.replace(".xlsx", ".csv")
        if os.path.exists(csv_fallback):
            return pd.read_csv(csv_fallback)
        return None
    try:
        # Coba baca excel, jika error coba csv
        return pd.read_excel(INPUT_DATA_PATH)
    except:
        return pd.read_csv(INPUT_DATA_PATH.replace(".xlsx", ".csv"))

# --- FUNGSI CEK KONSISTENSI ---
def check_consistency(arah_label, sudut):
    s = sudut % 360
    # Mapping sederhana
    if (s >= 315 or s < 45): zone = "Utara"
    elif (s >= 45 and s < 135): zone = "Timur"
    elif (s >= 135 and s < 225): zone = "Selatan"
    else: zone = "Barat"
    return (zone == arah_label), zone

# --- FUNGSI PLOT KOMPAS (FIXED) ---
def plot_compass_fixed(sudut_derajat, arah_label, is_consistent, zone_real):
    sudut_vis = (450 - sudut_derajat) % 360 # Konversi rotasi matematika ke kompas
    color = "green" if is_consistent else "orange"
    
    fig = go.Figure()

    # 1. Lingkaran Kompas
    fig.add_trace(go.Scatterpolar(
        r=[1]*360, theta=list(range(360)),
        mode='lines', line=dict(color='black', width=1),
        hoverinfo='none', showlegend=False
    ))

    # 2. Jarum Penunjuk
    fig.add_trace(go.Scatterpolar(
        r=[0, 0.9], theta=[0, sudut_derajat],
        mode='lines+markers',
        line=dict(color=color, width=4),
        marker=dict(symbol="arrow-bar-up", size=20, color=color),
        name=f"Prediksi: {sudut_derajat:.1f}°"
    ))

    # Layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 90, 180, 270],
                ticktext=['U', 'T', 'S', 'B'],
                rotation=90, direction="clockwise",
                tickfont=dict(size=14, color="black")
            )
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        title=dict(
            text=f"Arah Dominan: {arah_label}<br><sub>(Real Angle: {sudut_derajat:.1f}°)</sub>",
            x=0.5
        ),
        height=400
    )
    return fig

# --- FUNGSI BARU: PLOT PERGESERAN VEKTOR (SPATIO-TEMPORAL) ---
def plot_movement_vector(df_input):
    if df_input is None or len(df_input) < 2:
        return None, "Data tidak cukup untuk analisis vektor."

    # Ambil 2 data terakhir
    cols = df_input.columns.tolist()
    
    col_x, col_y = None, None
    
    # Cari Latitude (Y)
    for c in ['Lintang', 'latitude', 'lat', 'ACO_center_x', 'center_x']: 
        if c in cols: col_y = c; break 
        
    # Cari Longitude (X)
    for c in ['Bujur', 'longitude', 'lon', 'ACO_center_y', 'center_y']: 
        if c in cols: col_x = c; break 

    if not col_x or not col_y:
        return None, "Kolom koordinat tidak ditemukan."

    # Data H (Sekarang) dan H-1 (Lalu)
    curr = df_input.iloc[-1]
    prev = df_input.iloc[-2]

    y1, x1 = prev[col_y], prev[col_x] # H-1
    y2, x2 = curr[col_y], curr[col_x] # H

    # Hitung Jarak
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # Plot Scatter Geo
    fig = go.Figure()

    # Garis Penghubung (Jejak Pergeseran)
    fig.add_trace(go.Scattermapbox(
        mode="lines+markers",
        lon=[x1, x2], lat=[y1, y2],
        marker={'size': 12, 'color': ["gray", "red"]},
        line={'width': 4, 'color': "red"},
        text=["H-1 (Sebelumnya)", "H (Terkini)"],
        name="Pergeseran Episentrum"
    ))

    # Layout Peta
    center_lat = (y1 + y2) / 2
    center_lon = (x1 + x2) / 2
    zoom_level = 8 if dist < 0.5 else 6

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
        title="Visualisasi Pergeseran (H-1 ke H)",
        height=350
    )
    
    # Buat Narasi
    arah_gerak = ""
    if y2 > y1: arah_gerak += "Utara"
    else: arah_gerak += "Selatan"
    if x2 > x1: arah_gerak += "-Timur"
    else: arah_gerak += "-Barat"

    narasi = f"""
    **Analisis Pergerakan:**
    Gempa bergeser dari **{prev.get('Lokasi', 'Area A')}** ke **{curr.get('Lokasi', 'Area B')}**.
    Secara vektor, terjadi pergeseran koordinat ke arah **{arah_gerak}**.
    Inilah yang menjadi basis input **Channel 3 & 4** pada CNN Spatio-Temporal.
    """
    
    return fig, narasi


# --- FUNGSI UTAMA (MAIN) ---
def main():
    st.title("🧭 CNN Spatio-Temporal Prediction")
    st.markdown("Dashboard ini menampilkan hasil prediksi arah gempa berikutnya berdasarkan pola **Citra ACO (Saat Ini)** dan **Riwayat Pergeseran (Masa Lalu)**.")

    df = load_prediction_data()
    df_input = load_input_data()

    if df is not None and not df.empty:
        # Ambil data prediksi terakhir
        last_pred = df.iloc[-1]
        
        ts = last_pred['timestamp']
        arah_lbl = last_pred['arah_prediksi']
        sudut = float(last_pred['arah_derajat'])
        risk_arr = last_pred.get('risk_k_array', '[0.0]')
        
        # Cek konsistensi
        is_consistent, zone_real = check_consistency(arah_lbl, sudut)

        # --- LAYOUT ATAS: METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Arah Prediksi", f"{arah_lbl}", delta=f"{sudut:.1f}°")
        col2.metric("Confidence Level", f"{last_pred.get('confidence_scalar', 0.0):.2%}")
        col3.metric("Status Model", "Spatio-Temporal (v3.3)", "Active")

        st.divider()

        # --- LAYOUT TENGAH: VISUALISASI UTAMA ---
        c_left, c_right = st.columns([1.5, 1])

        with c_left:
            st.subheader("1. Analisis Pergeseran (H-1 ⮕ H)")
            if df_input is not None:
                fig_map, narasi_map = plot_movement_vector(df_input)
                if fig_map:
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.info(narasi_map)
                else:
                    st.warning(narasi_map)
            else:
                st.warning("Data input mentah tidak ditemukan. Visualisasi vektor tidak tersedia.")

        with c_right:
            st.subheader("2. Hasil Prediksi Arah")
            # Plot Kompas
            fig_compass = plot_compass_fixed(sudut, arah_lbl, is_consistent, zone_real)
            st.plotly_chart(fig_compass, use_container_width=True)
            
            with st.expander("Lihat Detail Probabilitas"):
                st.write(f"**Raw Risk Array:** `{risk_arr}`")
                if is_consistent:
                    st.success("✅ Output Klasifikasi & Regresi Sinkron.")
                else:
                    st.warning("⚠️ Terdapat deviasi antara label & sudut.")

        # --- TABEL DATA ---
        st.divider()
        st.subheader("📜 Riwayat Prediksi Terbaru")
        st.dataframe(df.tail(10)[['timestamp', 'arah_prediksi', 'arah_derajat', 'confidence_scalar']].sort_values('timestamp', ascending=False))

    else:
        st.warning("Belum ada data prediksi CNN (File csv output belum tersedia). Jalankan `cnn_engine.py` terlebih dahulu.")

# --- EKSEKUSI PROGRAM ---
if __name__ == "__main__":
    main()