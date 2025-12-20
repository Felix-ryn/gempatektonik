import streamlit as st
import pandas as pd
import pydeck as pdk
import streamlit.components.v1 as components
import json
import sys
from pathlib import Path

# --- SETUP PATH SYSTEM ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))

# --- CONFIG HALAMAN ---
st.set_page_config(page_title="ACO Analysis", layout="wide", page_icon="🐜")

# --- PATH DATA ---
BASE_ACO = PROJECT_ROOT / "output" / "aco_results"
FILE_HTML_MAP = BASE_ACO / "aco_impact_zones.html"  # File Peta Leaflet dari Backend
FILE_EPICENTERS = BASE_ACO / "aco_epicenters.csv"
FILE_SUMMARY = BASE_ACO / "aco_summary.json"

def main():
    st.title("🐜 ACO: Analisis Pusat & Radius Dampak")
    st.markdown("""
    Dashboard ini menampilkan hasil **Ant Colony Optimization (ACO)**:
    1. **Peta Interaktif** (Pusat & Radius).
    2. **Statistik Risiko** (Berdasarkan data backend).
    """)

    # --- 1. LOAD DATA STATISTIK (Untuk KPI di Atas) ---
    df = None
    summary = {}
    
    # Load Summary JSON
    if FILE_SUMMARY.exists():
        try:
            with open(FILE_SUMMARY, 'r') as f:
                summary = json.load(f)
        except: pass

    # Load CSV Epicenters
    if FILE_EPICENTERS.exists():
        try:
            df = pd.read_csv(FILE_EPICENTERS)
        except: pass

    # --- 2. TAMPILKAN KPI (Statistik) ---
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        # Ambil data akurat dari JSON Summary atau hitung dari CSV
        center_lat = summary.get('center_latitude', df['Lintang'].mean())
        center_lon = summary.get('center_longitude', df['Bujur'].mean())
        # Prioritaskan radius dari JSON (Impact Radius Global), fallback ke rata-rata visual
        global_radius = summary.get('impact_radius_km', 0)
        
        col1.metric("Pusat Risiko (Latitude)", f"{center_lat:.4f}")
        col1.caption(f"Longitude: {center_lon:.4f}")
        
        col2.metric("Radius Dampak Global", f"{global_radius:.2f} KM")
        
        max_pher = df['Pheromone_Score'].max() if 'Pheromone_Score' in df.columns else 0
        col3.metric("Max Intensitas Pheromone", f"{max_pher:.4f}")
        
        st.markdown("---")

    # --- 3. VISUALISASI PETA (SOLUSI UTAMA) ---
    st.subheader("🗺️ Peta Risiko (Leaflet Visualization)")
    
    # CEK 1: Apakah ada file HTML Leaflet? (Prioritas Utama)
    if FILE_HTML_MAP.exists():
        try:
            # Baca file HTML dan render sebagai komponen
            with open(FILE_HTML_MAP, 'r', encoding='utf-8') as f:
                html_code = f.read()
                
            # Render HTML Leaflet (Height disesuaikan agar peta terlihat penuh)
            components.html(html_code, height=600, scrolling=True)
            st.success("✅ Peta Leaflet berhasil dimuat dari 'aco_impact_zones.html'.")
            
        except Exception as e:
            st.error(f"Gagal memuat file HTML peta: {e}")
            # Jika HTML gagal, akan lanjut ke fallback PyDeck di bawah
            st.warning("Beralih ke tampilan PyDeck cadangan...")
            render_pydeck_fallback(df, center_lat, center_lon)
            
    # CEK 2: Jika file HTML tidak ada, pakai PyDeck (Cadangan)
    else:
        st.info("File peta 'aco_impact_zones.html' tidak ditemukan. Menggunakan visualisasi cadangan (PyDeck).")
        if df is not None:
            render_pydeck_fallback(df, center_lat, center_lon)
        else:
            st.error("Data CSV juga tidak ditemukan. Pastikan backend 'python main.py' sudah dijalankan.")

def render_pydeck_fallback(df, lat_center, lon_center):
    """Fungsi cadangan jika HTML Leaflet tidak ada"""
    if df is None: return

    # Pastikan kolom radius visual ada
    if 'Radius_Visual_KM' not in df.columns:
        df['Radius_Visual_KM'] = df.get('Pheromone_Score', 0) * 50

    df['radius_meter'] = df['Radius_Visual_KM'] * 1000

    # Layer Area (Merah Transparan)
    layer_area = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["Bujur", "Lintang"],
        get_fill_color=[255, 0, 0, 80],
        get_radius="radius_meter",
        stroked=True, filled=True, pickable=True
    )
    
    # Layer Pusat (Putih)
    layer_center = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["Bujur", "Lintang"],
        get_fill_color=[255, 255, 255, 200],
        get_radius=1000,
        pickable=True
    )

    view = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=6)
    
    # Gunakan style 'carto-darkmatter' yang tidak butuh API Key Mapbox
    st.pydeck_chart(pdk.Deck(
        map_style='carto-darkmatter', 
        initial_view_state=view,
        layers=[layer_area, layer_center],
        tooltip={"html": "<b>Lokasi:</b> {Lokasi}<br><b>Radius:</b> {Radius_Visual_KM} KM"}
    ))

if __name__ == "__main__":
    main()