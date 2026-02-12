import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="System Evaluation", layout="wide", page_icon="✅")

# 1. LOAD METRICS JSON (General System Health / Training Result)
def load_real_metrics():
    """Membaca file JSON hasil output training (metrics, confusion matrix)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "..", "output", "system_metrics.json")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

# =========================================================
# 2. LOAD VALIDASI CNN (Logika Engine Baru - Spatial)
# =========================================================
def load_cnn_validation():
    """Membaca CSV hasil prediksi CNN Engine (Titanium Safe Edition)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        current_dir,
        "..", "..", "output", "cnn_results",
        "cnn_next_earthquake_prediction.csv"
    )

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Gagal membaca CSV validasi CNN: {e}")
            return None
    return None

# =========================================================
# MAIN DASHBOARD
# =========================================================
def main():
    st.title("✅ Evaluasi Sistem & Validasi Model")
    
    # Load Data
    data_metrics = load_real_metrics()
    df_val = load_cnn_validation()

    # =====================================================
    # BAGIAN 1: PERFORMA TRAINING (METRICS & CONFUSION MATRIX)
    # Relevan dengan Screenshot Anda
    # =====================================================
    st.header("1. Performa Model (Training & Testing)")
    
    accuracy, precision, recall = 0.0, 0.0, 0.0
    labels = ['Rendah', 'Sedang', 'Tinggi'] # Default
    cm_data = []
    
    if data_metrics:
        metrics = data_metrics.get("metrics", {})
        accuracy = metrics.get("accuracy", 0.0)
        avg = metrics.get("weighted avg", {})
        precision = avg.get("precision", 0.0)
        recall = avg.get("recall", 0.0)
        
        cm_data = data_metrics.get("confusion_matrix", [])
        if "labels" in data_metrics:
            labels = data_metrics["labels"]

    # --- KPI CARDS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Akurasi Model", f"{accuracy*100:.1f}%")
    c2.metric("Presisi (Weighted)", f"{precision*100:.1f}%")
    c3.metric("Recall (Weighted)", f"{recall*100:.1f}%")

    st.markdown("---")

    # --- CHARTS ROW (Confusion Matrix & F1-Score) ---
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("🧩 Confusion Matrix")
        if cm_data and len(cm_data) == len(labels):
            x_labels = [f'Pred {l}' for l in labels]
            y_labels = [f'Aktual {l}' for l in labels]
            
            fig_cm = ff.create_annotated_heatmap(
                z=cm_data,
                x=x_labels,
                y=y_labels,
                colorscale='Viridis',
                showscale=True
            )
            fig_cm.update_layout(height=400, margin=dict(t=50, l=0, r=0, b=0))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Data Confusion Matrix tidak tersedia.")

    with col_r:
        st.subheader("📊 Performa Per Kelas (F1-Score)")
        if data_metrics and "metrics" in data_metrics:
            class_metrics = []
            for k, v in data_metrics["metrics"].items():
                if k in labels:
                    class_metrics.append({
                        "Kelas": k,
                        "F1-Score": v.get("f1-score", 0)
                    })
            
            if class_metrics:
                df_metrics = pd.DataFrame(class_metrics)
                fig_bar = px.bar(
                    df_metrics,
                    x='Kelas',
                    y='F1-Score',
                    color='Kelas',
                    range_y=[0, 1.05],
                    text_auto='.2f'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Data F1-Score per kelas tidak ditemukan.")
        else:
            st.info("Menunggu data metrics...")

    st.markdown("---")
    st.markdown("---")

    # =====================================================
    # BAGIAN 2: VALIDASI SPASIAL (ENGINE TERBARU)
    # Logika Baru: Peta, Sudut, Status Validasi
    # =====================================================
    st.header("2. Validasi Prediksi Spasial (Real-World)")
    st.caption("Evaluasi berdasarkan engine v3.3: Membandingkan arah pergerakan gempa prediksi vs aktual.")

    if df_val is None or df_val.empty:
        st.warning("⚠️ Belum ada output prediksi dari CNN Engine.")
        return

    # Ambil data terbaru
    latest = df_val.iloc[-1]

    # --- STATUS BANNER ---
    status = latest.get("status_validasi", "PENDING")
    note = latest.get("validasi_note", "-")
    
    status_color = "blue"
    if status == "VALID": status_color = "green"
    elif status == "MENYIMPANG": status_color = "red"
    elif status == "PENDING": status_color = "orange"

    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border-left: 6px solid {status_color};">
            <h3 style="margin:0; color:{status_color};">STATUS: {status}</h3>
            <p style="margin:5px 0 0 0;"><b>Catatan Engine:</b> {note}</p>
        </div>
        <br>
        """, 
        unsafe_allow_html=True
    )

    # --- METRIK SPASIAL ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Arah Prediksi", latest.get("arah_prediksi", "-"))
    k2.metric("Sudut (Azimuth)", f"{latest.get('arah_derajat', 0):.1f}°")
    
    selisih = latest.get("selisih_sudut", -1)
    selisih_str = f"{selisih:.1f}°" if selisih != -1 else "Menunggu Data"
    k3.metric("Selisih vs Aktual", selisih_str)
    
    conf = float(latest.get("confidence_scalar", 0.0)) * 100
    k4.metric("Confidence", f"{conf:.1f}%")

    # --- PETA VISUALISASI (Spatial Projection) ---
    st.subheader("📍 Peta Proyeksi Pergerakan")
    
    # Persiapan Data Peta
    map_data = []
    
    # 1. Pusat Gempa (Basis Data / H-1)
    lat_center = latest.get("ACO_Center_Lat")
    lon_center = latest.get("ACO_Center_Lon")
    if pd.notna(lat_center) and pd.notna(lon_center):
        map_data.append({
            "lat": lat_center, "lon": lon_center, 
            "label": "Pusat Gempa (Basis)", "color": "blue", "size": 10
        })

    # 2. Titik Prediksi (Hasil Proyeksi Sudut & Jarak)
    lat_proj = latest.get("proj_target_lat")
    lon_proj = latest.get("proj_target_lon")
    dist_proj = latest.get("proj_distance_km", 0)
    
    if pd.notna(lat_proj) and pd.notna(lon_proj):
        map_data.append({
            "lat": lat_proj, "lon": lon_proj, 
            "label": f"Prediksi (Est. {dist_proj}km)", "color": "orange", "size": 12
        })

    if map_data:
        df_map = pd.DataFrame(map_data)
        fig_map = px.scatter_mapbox(
            df_map, lat="lat", lon="lon", color="label", size="size",
            color_discrete_map={"Pusat Gempa (Basis)": "blue", f"Prediksi (Est. {dist_proj}km)": "orange"},
            zoom=6, mapbox_style="open-street-map",
            title="Visualisasi Arah Pergerakan (Basis -> Prediksi)"
        )
        
        # Garis Imajiner
        if len(df_map) >= 2:
            fig_map.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[lon_center, lon_proj], lat=[lat_center, lat_proj],
                line=dict(width=2, color='orange', dash='dot'),
                name="Arah Azimuth"
            ))
            
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Koordinat visualisasi tidak tersedia.")

    # --- DETAIL TEKNIS ---
    with st.expander("🔍 Detail Sampling & Data Mentah"):
        c_tech1, c_tech2 = st.columns(2)
        with c_tech1:
            st.markdown("**Sampling Kandidat (Search Engine):**")
            st.write(f"- Alt Sampling 1 (Angle): {latest.get('alt_sampling_1_angle', '-')}")
            st.write(f"- Alt Sampling 1 (Diff): {latest.get('alt_sampling_1_diff', '-')}")
        with c_tech2:
             st.markdown("**Parameter Validasi:**")
             st.write("- Threshold Sudut: 60°")
             st.write("- Threshold Jarak: 50 km")

if __name__ == "__main__":
    main()