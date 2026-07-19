import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import math
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
# 2. Path Output Validasi Januari 2025
VALIDATION_PATH = os.path.join(
    CURRENT_DIR,
    "../../output/cnn_results/validation_2025.csv"
)
# 3. Path Input Data Mentah (Untuk visualisasi H-1 ke H)
INPUT_DATA_PATH = os.path.join(CURRENT_DIR, "../../output/lstm_results/lstm_data_for_cnn.xlsx")

JATIM_COORDINATES = {
    # ==========================
    # KABUPATEN (29)
    # ==========================
    "Pacitan": (-8.1944, 111.1055),
    "Ponorogo": (-7.8717, 111.4620),
    "Trenggalek": (-8.0500, 111.7167),
    "Tulungagung": (-8.0657, 111.9025),
    "Blitar": (-8.0982, 112.1681),
    "Kediri": (-7.8480, 112.0178),
    "Malang": (-8.0069, 112.6293),
    "Lumajang": (-8.1335, 113.2248),
    "Jember": (-8.1724, 113.7000),
    "Banyuwangi": (-8.2192, 114.3691),
    "Bondowoso": (-7.9135, 113.8214),
    "Situbondo": (-7.7062, 114.0098),
    "Probolinggo": (-7.7543, 113.2159),
    "Pasuruan": (-7.6450, 112.9070),
    "Sidoarjo": (-7.4467, 112.7183),
    "Mojokerto": (-7.5361, 112.4255),
    "Jombang": (-7.5459, 112.2338),
    "Nganjuk": (-7.6051, 111.9035),
    "Madiun": (-7.6298, 111.5239),
    "Magetan": (-7.6536, 111.3270),
    "Ngawi": (-7.4039, 111.4467),
    "Bojonegoro": (-7.1502, 111.8817),
    "Tuban": (-6.8976, 112.0649),
    "Lamongan": (-7.1197, 112.4171),
    "Gresik": (-7.1567, 112.6555),
    "Bangkalan": (-7.0455, 112.7351),
    "Sampang": (-7.1872, 113.2394),
    "Pamekasan": (-7.1603, 113.4821),
    "Sumenep": (-7.0045, 113.8592),

    # ==========================
    # KOTA (9)
    # ==========================
    "Kota Kediri": (-7.8167, 112.0167),
    "Kota Blitar": (-8.0956, 112.1608),
    "Kota Malang": (-7.9819, 112.6265),
    "Kota Probolinggo": (-7.7549, 113.2152),
    "Kota Pasuruan": (-7.6449, 112.9061),
    "Kota Mojokerto": (-7.4722, 112.4336),
    "Kota Madiun": (-7.6298, 111.5239),
    "Kota Surabaya": (-7.2575, 112.7521),
    "Kota Batu": (-7.8671, 112.5239),
}

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

def load_validation_data():
    """Load hasil validasi prediksi terhadap data aktual Januari 2025"""

    if not os.path.exists(VALIDATION_PATH):
        return None

    try:
        return pd.read_csv(VALIDATION_PATH)

    except Exception as e:
        st.error(f"Error loading Validation CSV : {e}")
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

# --- KONVERSI SUDUT MENJADI 4 ARAH MATA ANGIN ---
def angle_to_direction(angle):
    angle = angle % 360

    if angle >= 315 or angle < 45:
        return "Utara"
    elif angle >= 45 and angle < 135:
        return "Timur"
    elif angle >= 135 and angle < 225:
        return "Selatan"
    else:
        return "Barat"

def nearest_city(lat, lon):

    nearest = None
    min_distance = float("inf")

    for city, (city_lat, city_lon) in JATIM_COORDINATES.items():

        distance = math.sqrt(
            (lat - city_lat) ** 2 +
            (lon - city_lon) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            nearest = city

    return nearest

# --- FUNGSI PLOT KOMPAS (FIXED) ---
def plot_compass_fixed(sudut_derajat, arah_label, is_consistent, zone_real):
    # Visualisasi Kompas: 0 derajat di Utara (Atas)
    # Kita tidak perlu konversi 450 - x jika input sudah azimut kompas.
    # Asumsi: Model output adalah Azimuth (0=U, 90=T, 180=S, 270=B)
    
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

# --- FUNGSI PLOT PERGESERAN VEKTOR (SPATIO-TEMPORAL) ---
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
    validation_df = load_validation_data()
    df_input = load_input_data()

    if df is not None and not df.empty:
        # Ambil data prediksi terakhir
        last_pred = df.iloc[-1]
        
        ts = last_pred['timestamp']
        arah_lbl = last_pred['arah_prediksi']
        sudut = float(last_pred['arah_derajat'])
        risk_arr = last_pred.get('risk_k_array', '[0.0]')
        
        # [FIX] Handle nama kolom confidence (bisa 'confidence' atau 'confidence_scalar')
        conf = last_pred.get('confidence', last_pred.get('confidence_scalar', 0.0))
        
        # Cek konsistensi
        is_consistent, zone_real = check_consistency(arah_lbl, sudut)

        # --- LAYOUT ATAS: METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Arah Prediksi", f"{arah_lbl}", delta=f"{sudut:.1f}°")
        col2.metric("Confidence Level", f"{conf:.2%}")
        col3.metric("Status Model", "Simple CNN (v3.3)", "Active")

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

        # ==========================================================
        # VALIDASI PREDIKSI TERHADAP DATA AKTUAL JANUARI 2025
        # ==========================================================

        st.divider()

        st.subheader("📍 Prediction Validation")

        if validation_df is not None and not validation_df.empty:

            val = validation_df.iloc[-1]

            angle_pred = float(val["pred_angle"])
            angle_actual = float(val["best_match_angle"])
            pred_direction = angle_to_direction(angle_pred)
            actual_direction = angle_to_direction(angle_actual)
            angle_diff = float(val["angle_diff_deg"])
            distance = float(val["best_match_distance_km"])
            match_flag = str(val["match_flag"]).strip().lower()

            status = "✅ VALID" if match_flag == "true" else "❌ MENYIMPANG"

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Prediksi",
                pred_direction,
                f"{angle_pred:.2f}°"
            )

            c2.metric(
                "Aktual",
                actual_direction,
                f"{angle_actual:.2f}°"
            )

            c3.metric(
                "Angle Difference",
                f"{angle_diff:.2f}°"
            )

            c4.metric(
                "Shift Distance",
                f"{distance:.2f} km"
            )

            st.success(f"Validation Status : {status}")

        else:

            st.warning("Validation data belum tersedia.")

        if validation_df is not None and not validation_df.empty:

            pred = df.iloc[-1]
            val = validation_df.iloc[-1]

            pred_lat = pred["proj_target_lat"]
            pred_lon = pred["proj_target_lon"]

            actual_lat = val["actual_event_lat"]
            actual_lon = val["actual_event_lon"]
            pred_city = nearest_city(pred_lat, pred_lon)
            actual_city = nearest_city(actual_lat, actual_lon)
            st.write("Pred City :", pred_city)
            st.write("Actual City :", actual_city)

            fig_validation = go.Figure()

            fig_validation.add_trace(
                go.Scattermapbox(
                    mode="markers",
                    lat=[pred_lat],
                    lon=[pred_lon],
                    marker=dict(
                        size=16,
                        color="red"
                    ),
                    name="Prediction",

                    hoverlabel=dict(
                        bgcolor="#1E293B",     # biru gelap
                        bordercolor="#38BDF8", # biru muda
                        font=dict(
                            color="white",
                            size=14
                        )
                    ),

                    hovertemplate=(
                        "<b>Prediction</b><br>"
                        f"Daerah : {pred_city}<br>"
                        f"Arah : {pred_direction}<br>"
                        f"Latitude : {pred_lat:.5f}<br>"
                        f"Longitude : {pred_lon:.5f}"
                        "<extra></extra>"
                    )
                )
            )

            fig_validation.add_trace(
                go.Scattermapbox(
                    mode="markers",
                    lat=[actual_lat],
                    lon=[actual_lon],
                    marker=dict(
                        size=16,
                        color="green"
                    ),
                    name="Actual",

                    hoverlabel=dict(
                        bgcolor="#1E293B",     # biru gelap
                        bordercolor="#38BDF8", # biru muda
                        font=dict(
                            color="white",
                            size=14
                        )
                    ),

                    hovertemplate=(
                        "<b>Actual Event</b><br>"
                        f"Daerah : {actual_city}<br>"
                        f"Arah : {actual_direction}<br>"
                        f"Latitude : {actual_lat:.5f}<br>"
                        f"Longitude : {actual_lon:.5f}"
                        "<extra></extra>"
                    )
                )
            )

            fig_validation.add_trace(
                go.Scattermapbox(
                    mode="lines",
                    lat=[pred_lat, actual_lat],
                    lon=[pred_lon, actual_lon],
                    line=dict(
                        width=3,
                        color="orange"
                    ),
                    hoverinfo="skip",      # <<< TAMBAHKAN
                    showlegend=True,
                    name="Prediction Error"
                )
            )

            fig_validation.update_layout(

                mapbox_style="open-street-map",

                mapbox=dict(

                    center=dict(
                        lat=(pred_lat + actual_lat)/2,
                        lon=(pred_lon + actual_lon)/2
                    ),

                    zoom=6

                ),

                height=500,

                margin=dict(
                    l=0,
                    r=0,
                    t=40,
                    b=0
                ),

                title="Prediction vs Actual Event (January 2025)"

            )
            

            st.plotly_chart(
                fig_validation,
                use_container_width=True
            )

            st.info(f"""
            ### Ringkasan Validasi

            - **Prediksi CNN** mengarah ke **{pred_direction}** ({angle_pred:.2f}°)

            - **Data Aktual Januari 2025** berada di arah **{actual_direction}** ({angle_actual:.2f}°)

            - **Selisih Sudut** sebesar **{angle_diff:.2f}°**

            - **Pergeseran Lokasi** sebesar **{distance:.2f} km**

            - **Status Validasi:** **{status}**
            """)

        # --- TABEL DATA ---
        st.divider()
        st.subheader("📜 Riwayat Prediksi Terbaru")
        
        # [FIX] Pastikan kolom confidence ditampilkan dengan benar
        cols_to_show = ['timestamp', 'arah_prediksi', 'arah_derajat']
        if 'confidence' in df.columns:
            cols_to_show.append('confidence')
        elif 'confidence_scalar' in df.columns:
            cols_to_show.append('confidence_scalar')
            
        st.dataframe(df.tail(10)[cols_to_show].sort_values('timestamp', ascending=False))

    else:
        st.warning("Belum ada data prediksi CNN (File csv output belum tersedia). Jalankan `cnn_engine.py` terlebih dahulu.")

# --- EKSEKUSI PROGRAM ---
if __name__ == "__main__":
    main()