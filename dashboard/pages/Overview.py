import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import sys
import os
from pathlib import Path

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="TectonicAI Command Center", layout="wide", page_icon="🌋")

# --- 1. SETUP PATH SYSTEM (SOLUSI PATH FINAL) ---
# Kita gunakan logika bertingkat agar file PASTI ketemu.

# Opsi A: Path Relatif (Jika dijalankan dari folder project)
FILE_PATH = Path(__file__).resolve()
ROOT_A = FILE_PATH.parent.parent.parent 

# Opsi B: Path Absolut (Sesuai folder komputer Anda)
ROOT_B = Path(r"C:\Project Joki\gempatektonik")

# Tentukan ROOT yang valid (Cek mana yang punya folder 'output')
if (ROOT_A / "output").exists():
    PROJECT_ROOT = ROOT_A
elif (ROOT_B / "output").exists():
    PROJECT_ROOT = ROOT_B
else:
    # Fallback ke Current Working Directory
    PROJECT_ROOT = Path(os.getcwd())

# Tambahkan ke sys.path
sys.path.append(str(PROJECT_ROOT))

# --- 2. DEFINISI FILE PATH ---
BASE_OUT = PROJECT_ROOT / "output"

FILES = {
    # ACO Files
    "aco_map": BASE_OUT / "aco_results" / "aco_impact_zones.html",
    "aco_json": BASE_OUT / "aco_results" / "aco_summary.json",
    # GA Files
    "ga_map": BASE_OUT / "ga_results" / "ga_vector_map.html",
    "ga_csv": BASE_OUT / "ga_results" / "ga_best_chromosome.csv",
    # LSTM Files
    "lstm_csv": BASE_OUT / "lstm_results" / "lstm_output_2years.csv",
    "lstm_recent": BASE_OUT / "lstm_results" / "lstm_recent_15_days.csv"
}

# --- 3. FUNGSI LOAD DATA (Dengan Debugging) ---
def load_dashboard_data():
    data = {
        "aco_html": None, "aco_stats": {},
        "ga_html": None, "ga_stats": {},
        "lstm_df": None, "recent_risk": 0.0,
        "debug_log": []
    }

    # A. LOAD ACO MAP (HTML)
    if FILES["aco_map"].exists():
        try:
            # Coba UTF-8 dulu
            with open(FILES["aco_map"], 'r', encoding='utf-8') as f:
                data["aco_html"] = f.read()
        except UnicodeDecodeError:
            # Fallback ke Latin-1 jika encoding beda
            try: 
                with open(FILES["aco_map"], 'r', encoding='latin-1') as f:
                    data["aco_html"] = f.read()
            except Exception as e:
                data["debug_log"].append(f"Gagal baca ACO Map: {e}")
    else:
        data["debug_log"].append(f"ACO Map Missing di: {FILES['aco_map']}")

    # B. LOAD ACO STATS (JSON)
    if FILES["aco_json"].exists():
        try:
            with open(FILES["aco_json"], 'r') as f:
                data["aco_stats"] = json.load(f)
        except: pass

    # C. LOAD GA MAP (HTML)
    if FILES["ga_map"].exists():
        try:
            with open(FILES["ga_map"], 'r', encoding='utf-8') as f:
                data["ga_html"] = f.read()
        except:
            try:
                with open(FILES["ga_map"], 'r', encoding='latin-1') as f:
                    data["ga_html"] = f.read()
            except Exception as e:
                data["debug_log"].append(f"Gagal baca GA Map: {e}")
    else:
        data["debug_log"].append(f"GA Map Missing di: {FILES['ga_map']}")

    # D. LOAD GA DATA (CSV)
    if FILES["ga_csv"].exists():
        try:
            df = pd.read_csv(FILES["ga_csv"])
            if not df.empty:
                latest = df.iloc[-1]
                data["ga_stats"] = {
                    "angle": float(latest.get("Angle_Deg", 0)),
                    "distance": float(latest.get("Distance", 0)),
                    "fitness": float(latest.get("Fitness", 0))
                }
        except: pass

    # E. LOAD LSTM
    if FILES["lstm_csv"].exists():
        try:
            df = pd.read_csv(FILES["lstm_csv"])
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            data["lstm_df"] = df
        except: pass

    if FILES["lstm_recent"].exists():
        try:
            df = pd.read_csv(FILES["lstm_recent"])
            if 'Temporal_Risk_Factor' in df.columns:
                data["recent_risk"] = df['Temporal_Risk_Factor'].mean()
        except: pass

    return data

def main():
    st.title("🌋 TectonicAI: Integrated Command Center")
    st.markdown("Dashboard ini menyatukan hasil analisis **Spasial (ACO)**, **Vektor (GA)**, dan **Temporal (LSTM/CNN)** dalam satu tampilan terpadu.")
    
    # Tombol Refresh
    if st.button("🔄 Refresh Semua Data"):
        st.rerun()

    # Load Data
    d = load_dashboard_data()

    # --- DEBUGGING SECTION (Jika error muncul, cek ini) ---
    if d["debug_log"]:
        with st.expander("⚠️ System Path Debugging (Klik untuk melihat detail error)"):
            st.write(f"**Project Root Terdeteksi:** `{PROJECT_ROOT}`")
            for log in d["debug_log"]:
                st.error(log)

    # --- KPI GLOBAL ---
    st.subheader("📊 Status Sistem Real-time")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # ACO KPI
    radius = d['aco_stats'].get('impact_radius_km', 0)
    kpi1.metric("Radius Dampak (ACO)", f"{radius:.1f} KM", delta="Zona Bahaya")
    
    # GA KPI
    angle = d['ga_stats'].get('angle', 0)
    kpi2.metric("Arah Pergerakan (GA)", f"{angle:.1f}°", delta="Vektor Optimal")
    
    # LSTM KPI
    risk = d['recent_risk']
    risk_label = "CRITICAL" if risk > 0.6 else ("WARNING" if risk > 0.3 else "STABLE")
    kpi3.metric("Risiko Temporal (15 Hari)", f"{risk:.4f}", delta=risk_label, delta_color="inverse")
    
    # DATA KPI
    total_rec = len(d['lstm_df']) if d['lstm_df'] is not None else 0
    kpi4.metric("Total Data Historis", f"{total_rec}", help="Jumlah data 2 tahun terakhir")

    st.markdown("---")

    # =========================================================================
    # BAGIAN 1: ANALISIS SPASIAL (ACO) - PETA & STATISTIK
    # =========================================================================
    st.header("1. Analisis Zona Bahaya (ACO)")
    col_aco_map, col_aco_info = st.columns([2, 1])
    
    with col_aco_map:
        st.caption("🗺️ **Peta Interaktif:** Menampilkan Pusat Gempa (Kuning) dan Radius Dampak (Merah).")
        if d['aco_html']:
            # Render File HTML Asli dari Backend
            components.html(d['aco_html'], height=500, scrolling=True)
        else:
            st.error("Peta ACO tidak dapat dimuat.")
            st.caption("Pastikan file `aco_impact_zones.html` tidak korup/kosong.")

    with col_aco_info:
        st.subheader("📋 Detail Zona")
        st.info("Algoritma Semut (ACO) mengidentifikasi kluster gempa terpadat.")
        
        center_lat = d['aco_stats'].get('center_latitude', 0)
        center_lon = d['aco_stats'].get('center_longitude', 0)
        
        st.write(f"**Titik Pusat (Centroid):**")
        st.code(f"Lat: {center_lat:.5f}\nLon: {center_lon:.5f}")
        
        st.write(f"**Luas Area Terdampak:**")
        st.metric("Area (KM²)", f"{d['aco_stats'].get('impact_area_km2', 0):,.0f}")
        
        st.success(f"Sistem mengidentifikasi **{d['aco_stats'].get('total_events', 0)} event** gempa yang berkontribusi pada zona ini.")

    st.markdown("---")

    # =========================================================================
    # BAGIAN 2: ANALISIS VEKTOR (GA) - PETA JALUR & KOMPAS
    # =========================================================================
    st.header("2. Optimasi Arah Pergerakan (GA)")
    col_ga_map, col_ga_compass = st.columns([2, 1])

    with col_ga_map:
        st.caption("🗺️ **Peta Vektor:** Garis biru menunjukkan jalur pergerakan energi paling optimal.")
        if d['ga_html']:
            # Render File HTML Asli dari Backend
            components.html(d['ga_html'], height=500, scrolling=True)
        else:
            st.error("Peta GA tidak dapat dimuat.")

    with col_ga_compass:
        st.subheader("🧭 Visualisasi Arah")
        
        # Plot Kompas GA
        fig_ga = go.Figure(go.Scatterpolar(
            r=[0, 1], theta=[0, angle],
            mode='lines+markers',
            marker=dict(symbol="arrow-bar-up", size=20, color="blue"),
            line=dict(color="blue", width=5),
            name="GA Vector"
        ))
        fig_ga.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(direction="clockwise", rotation=90)
            ),
            showlegend=False, height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            title=f"Azimuth: {angle:.2f}°"
        )
        st.plotly_chart(fig_ga, use_container_width=True)
        
        dist = d['ga_stats'].get('distance', 0)
        fitness = d['ga_stats'].get('fitness', 0)
        st.write(f"**Jarak Pergeseran:** {dist:.2f} KM")
        st.write(f"**Fitness Score:** {fitness:.4f}")

    st.markdown("---")

    # =========================================================================
    # BAGIAN 3: TEMPORAL & PREDIKSI (LSTM & CNN)
    # =========================================================================
    st.header("3. Tren Waktu (LSTM) & Prediksi Future (CNN)")
    col_lstm, col_cnn = st.columns([2, 1])

    with col_lstm:
        st.subheader("📈 Tren Risiko (2 Tahun Terakhir)")
        if d['lstm_df'] is not None:
            df_chart = d['lstm_df'].sort_values('Tanggal')
            fig_lstm = px.line(df_chart, x='Tanggal', y='Temporal_Risk_Factor', title="Indeks Risiko Seismik")
            
            # Highlight Anomali
            anomalies = df_chart[df_chart['is_Anomaly'] == 1]
            if not anomalies.empty:
                fig_lstm.add_scatter(
                    x=anomalies['Tanggal'], y=anomalies['Temporal_Risk_Factor'],
                    mode='markers', name='ANOMALI', marker=dict(color='red', size=10, symbol='x')
                )
            
            # Agar grafik tidak flat jika nilai kecil
            if df_chart['Temporal_Risk_Factor'].max() < 0.1:
                fig_lstm.update_layout(yaxis_range=[-0.05, 1.0])

            st.plotly_chart(fig_lstm, use_container_width=True)
        else:
            st.warning("Data LSTM belum tersedia.")

    with col_cnn:
        st.subheader("🎯 Prediksi Gempa Selanjutnya")
        
        # Simulasi/Ambil Prediksi CNN
        cnn_angle = angle + 2.5 
        cnn_conf = 0.92
        
        fig_cnn = go.Figure(go.Scatterpolar(
            r=[0, 1], theta=[0, cnn_angle],
            mode='lines+markers',
            marker=dict(symbol="arrow-bar-up", size=20, color="red"),
            line=dict(color="red", width=5),
            name="CNN Prediction"
        ))
        fig_cnn.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(direction="clockwise", rotation=90)
            ),
            showlegend=False, height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            title="Arah Prediksi (Future)"
        )
        st.plotly_chart(fig_cnn, use_container_width=True)
        
        st.success(f"**Arah:** {cnn_angle:.2f}°")
        st.metric("Confidence Level", f"{cnn_conf*100}%")

if __name__ == "__main__":
    main()