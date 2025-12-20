import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import sys
from pathlib import Path

# --- SETUP PATH SYSTEM (Anti-Error Path) ---
FILE_PATH = Path(__file__).resolve()
# Root = C:\Project Joki\gempatektonik
PROJECT_ROOT = FILE_PATH.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))

# --- CONFIG ---
st.set_page_config(page_title="GA Optimization", layout="wide", page_icon="🧬")

# --- DEFINISI PATH DATA ---
BASE_GA = PROJECT_ROOT / "output" / "ga_results"
FILE_HTML_MAP = BASE_GA / "ga_vector_map.html"
FILE_SUMMARY = BASE_GA / "ga_summary.json"
FILE_BEST_CHROM = BASE_GA / "ga_best_chromosome.csv"

def get_cardinal_direction(degree):
    """Konversi derajat ke arah mata angin."""
    try:
        dirs = ["Utara (N)", "Timur Laut (NE)", "Timur (E)", "Tenggara (SE)", 
                "Selatan (S)", "Barat Daya (SW)", "Barat (W)", "Barat Laut (NW)"]
        ix = round(degree / (360. / len(dirs)))
        return dirs[ix % len(dirs)]
    except:
        return "-"

def load_ga_data():
    """
    Fungsi ini dipaksa membaca JSON secara langsung tanpa caching
    agar data selalu update sesuai backend.
    """
    data = {
        "angle": 0.0, "distance": 0.0, "fitness": 0.0,
        "start_lat": 0.0, "start_lon": 0.0,
        "json_status": "Missing",
        "html_exists": False,
        "df_hist": None
    }

    # 1. BACA JSON (Prioritas Utama)
    if FILE_SUMMARY.exists():
        try:
            with open(FILE_SUMMARY, 'r') as f:
                summary = json.load(f)
                # Mapping kunci sesuai file Anda
                data["angle"] = float(summary.get('angle_deg', 0))
                data["distance"] = float(summary.get('distance_km', 0))
                data["fitness"] = float(summary.get('fitness', 0))
                data["start_lat"] = float(summary.get('start_lat', 0))
                data["start_lon"] = float(summary.get('start_lon', 0))
                data["json_status"] = "Success"
        except Exception as e:
            data["json_status"] = f"Error: {e}"
    
    # 2. Cek Peta HTML
    if FILE_HTML_MAP.exists():
        data['html_exists'] = True

    # 3. Load CSV untuk Tabel (Pelengkap)
    if FILE_BEST_CHROM.exists():
        try:
            data['df_hist'] = pd.read_csv(FILE_BEST_CHROM)
        except: pass

    return data

def main():
    st.title("🧬 Genetic Algorithm: Optimasi Vektor")
    
    # Tombol Refresh Manual (Penting jika backend baru saja selesai)
    if st.button("🔄 Refresh Data JSON"):
        st.rerun()

    # --- LOAD DATA ---
    loaded = load_ga_data()

    # --- DIAGNOSIS (Jika masih error, ini akan memberitahu alasannya) ---
    if loaded["json_status"] != "Success":
        st.error(f"❌ Gagal memuat JSON. Status: {loaded['json_status']}")
        st.caption(f"Target Path: {FILE_SUMMARY}")
        
        # Fallback ke CSV jika JSON gagal, supaya dashboard tidak kosong melompong
        if loaded['df_hist'] is not None:
             st.warning("⚠️ Menampilkan data cadangan dari CSV (JSON bermasalah).")
             latest = loaded['df_hist'].iloc[-1]
             loaded["angle"] = latest.get("Angle_Deg", 0)
             loaded["distance"] = latest.get("Distance", 0)
             loaded["fitness"] = latest.get("Fitness", 0)
        else:
             return # Stop jika tidak ada data sama sekali

    # Ambil variable bersih
    angle = loaded["angle"]
    distance = loaded["distance"]
    fitness = loaded["fitness"]
    direction_str = get_cardinal_direction(angle)

    # --- VISUALISASI KPI ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Arah Dominan", direction_str, delta="GA Optimized")
    col2.metric("Sudut (Azimuth)", f"{angle:.2f}°")
    col3.metric("Jarak Pergeseran", f"{distance:.2f} KM")
    col4.metric("Fitness Score", f"{fitness:.4f}")
    
    st.markdown("---")

    # --- LAYOUT VISUALISASI ---
    c_left, c_right = st.columns([1.5, 1])

    # 1. PETA (Kiri)
    with c_left:
        st.subheader("🗺️ Peta Vektor Pergerakan")
        if loaded['html_exists']:
            # Render HTML Peta Leaflet
            with open(FILE_HTML_MAP, 'r', encoding='utf-8') as f:
                html_map = f.read()
            components.html(html_map, height=500, scrolling=True)
            st.caption(f"📍 Koordinat Awal: {loaded['start_lat']:.5f}, {loaded['start_lon']:.5f}")
        else:
            st.warning("File peta HTML belum tersedia.")

    # 2. KOMPAS (Kanan)
    with c_right:
        st.subheader("🧭 Arah Mata Angin")
        fig = go.Figure()
        
        # Gambar Panah
        fig.add_trace(go.Scatterpolar(
            r=[0, 1], theta=[0, angle],
            mode='lines+markers',
            marker=dict(symbol="arrow-bar-up", size=20, color="blue"),
            line=dict(color="blue", width=5),
            name="Vektor GA"
        ))
        
        # Setting Kompas
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                    direction="clockwise", rotation=90
                )
            ),
            showlegend=False, height=400,
            margin=dict(l=30, r=30, t=30, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- DEBUG SECTION (Opsional: Untuk membuktikan JSON terbaca) ---
    with st.expander("🔍 Cek Isi File JSON (Debug)"):
        if FILE_SUMMARY.exists():
            with open(FILE_SUMMARY, 'r') as f:
                st.json(json.load(f))
        else:
            st.write("File tidak ditemukan.")

if __name__ == "__main__":
    main()