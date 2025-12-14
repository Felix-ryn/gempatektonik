# ============================================================
# TectonicAI Historical Context Analyzer
# File: TectonicAI/reporting/historical_analyzer.py
# PERAN:
#   - Memproses DataFrame LENGKAP (Statis + Historys).
#   - Menampilkan Visualisasi Data Historis, ACO Zoning, dan GA Paths.
# ============================================================

import logging
import pandas as pd
import folium
import os
import json
from typing import Dict, Any, Tuple
from folium.plugins import MarkerCluster, HeatMap, MiniMap, MousePosition
from jinja2 import Template # Asumsi library Jinja2 digunakan untuk rendering HTML

logger = logging.getLogger("HistoricalAnalyzer")

def generate_historical_dashboard(
    df_dynamic: pd.DataFrame,  # SELURUH DATA: Statis (2090) + Historys (Live Buffer)
    config: dict, 
    summary: dict,
    eval_results: dict # Hasil metrik akurasi/F1-Score dari EvaluationEngine
):
    logger.info("=== GENERATING HISTORICAL CONTEXT DASHBOARD ===")
    
    if df_dynamic.empty:
        logger.error("DataFrame input historis kosong. Abort dashboard.")
        return

    # 1. INITIALIZE MAP 
    lat_col = 'Lintang' 
    lon_col = 'Bujur' 
    
    center_lat = df_dynamic[lat_col].mean()
    center_lon = df_dynamic[lon_col].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='cartodbdark_matter')

    # Plugins
    MiniMap(toggle_display=True).add_to(m)
    MousePosition().add_to(m)

    # ---------------------------------------------------------------------
    # 2. LAYER 1: SEISMIC HEATMAP (Density)
    # ---------------------------------------------------------------------
    heat_data = df_dynamic[[lat_col, lon_col, 'Magnitudo']].values.tolist()
    HeatMap(heat_data, radius=12, blur=10, name="Historical Heatmap (Weighted by Mag)").add_to(m)


    # ---------------------------------------------------------------------
    # 3. LAYER 2: ACO ZONING CLUSTERS (Area Risiko Semi-Permanen)
    # ---------------------------------------------------------------------
    aco_group = folium.FeatureGroup(name="ACO Risk Zones").add_to(m)
    
    df_risk_zones = df_dynamic[df_dynamic.get('Status_Zona') == 'Terdampak'].copy()
    
    if not df_risk_zones.empty:
        for _, row in df_risk_zones.iterrows():
            # Radius dapat diambil dari Pheromone Score atau Context_Impact_Radius
            risk_score = row.get('Pheromone_Score', 0.5) 

            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=risk_score * 8 + 3,  # Ukuran berdasarkan skor pheromone
                color='#e74c3c',         
                fill=True,
                fill_opacity=risk_score * 0.4,
                tooltip=f"ACO Zone: {row.get('Status_Zona')} (Pheromone: {risk_score:.2f})"
            ).add_to(aco_group)
    
    # ---------------------------------------------------------------------
    # 4. LAYER 3: GA OPTIMAL RISK PATH (Visualisasi Historis)
    # ---------------------------------------------------------------------
    ga_path_group = folium.FeatureGroup(name="GA Optimal Path").add_to(m)
    
    # Ambil titik (Pred_Lat, Pred_Lon) dari data Historis yang memiliki hasil GA
    if 'Pred_Lat' in df_dynamic.columns and 'Pred_Lon' in df_dynamic.columns:
        # Gunakan semua titik prediksi GA untuk melihat jalur historis
        path_points = df_dynamic.dropna(subset=['Pred_Lat', 'Pred_Lon'])[['Pred_Lat', 'Pred_Lon']].values.tolist()

        if len(path_points) > 1:
            folium.PolyLine(
                path_points,
                color="#f1c40f", 
                weight=3,
                opacity=0.7,
                tooltip="GA Optimal Risk Path (Historical Trend)"
            ).add_to(ga_path_group)
            
            # Marker Posisi Terakhir di Path
            folium.Marker(path_points[-1], icon=folium.Icon(color='orange', icon='arrow-up', prefix='fa'), tooltip="Historical Path End").add_to(ga_path_group)


    # ---------------------------------------------------------------------
    # 5. ELEMEN VALIDASI & METRIK EVALUASI
    # ---------------------------------------------------------------------
    
    accuracy = eval_results.get('accuracy', 0.0) * 100
    f1_score = eval_results.get('f1_score', 0.0) * 100
    lstm_mean_anomaly = summary.get('lstm_anomaly_mean', 0.0)
    
    validation_text = f"""
    <div style='position: fixed; top: 10px; left: 10px; z-index: 9999; background: white; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>
        <h4>VALIDASI MODEL KESELURUHAN</h4>
        <hr style='margin: 5px 0;'>
        <b>Acc. Ensemble (Train):</b> <span style='color:#2980b9;'>{accuracy:.2f}%</span><br>
        <b>F1-Score (Test):</b> <span style='color:#2980b9;'>{f1_score:.2f}%</span><br>
        <b>Rata-rata Anomali LSTM:</b> {lstm_mean_anomaly:.4f}<br>
        <hr style='margin: 5px 0;'>
        <small>Data historis terakhir diproses: {df_dynamic['Tanggal'].max().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(validation_text))


    # 6. FINAL RENDER
    folium.LayerControl().add_to(m)
    
    # Pastikan kunci 'dashboard_historical' ada di CONFIG
    output_path = config['output_paths']['dashboard_historical'] 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    
    logger.info(f"✔ HISTORICAL CONTEXT DASHBOARD generated at: {output_path}")