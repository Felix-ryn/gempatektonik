# ============================================================
# TectonicAI Live Monitor Generator
# File: TectonicAI/reporting/live_monitor_generator.py
# PERAN:
#   - Menerima HANYA 1 BARIS DATA LIVE dari Orchestrator.
#   - Memvisualisasikan Titik Gempa, Garis Vektor GA, dan Circle Risiko.
# ============================================================

import logging
import pandas as pd
import folium
import os
import math
import json
from typing import Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger("LiveMonitor_Generator")

# --- KERNEL GEODESIK (Asumsi GeodesicKernel sudah tersedia) ---
# Definisi ini dipindahkan dari Orchestrator/utility agar generator dapat berdiri sendiri
class GeodesicKernel:
    R_EARTH_KM = 6371.0
    
    @staticmethod
    def project_destination(lat: float, lon: float, bearing: float, distance_km: float) -> Tuple[float, float]:
        # Logika proyeksi Geodesic Haversine
        lat_rad, lon_rad, bear_rad = map(math.radians, [lat, lon, bearing])
        angular_dist = distance_km / GeodesicKernel.R_EARTH_KM
        
        lat_dest = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist) +
            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bear_rad)
        )
        lon_dest = lon_rad + math.atan2(
            math.sin(bear_rad) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(lat_dest)
        )
        return math.degrees(lat_dest), math.degrees(lon_dest)


def generate_live_monitor(
    df_live: pd.DataFrame,  # HARUS 1 BARIS DATA FINAL LIVE
    config: dict, 
    summary: dict, 
):
    logger.info("=== GENERATING LIVE MONITOR DASHBOARD (1-Record Focus) ===")
    
    if df_live.empty or len(df_live) != 1:
        logger.error("Input harus 1 baris data LIVE. Melewati pembuatan peta live.")
        return

    # 1. AMBIL DATA TERBARU DAN SKOR
    row = df_live.iloc[0] 
    
    final_prob = row.get('Final_Prob_Impact', 0) * 100
    is_anomaly = row.get('is_Anomaly', False)
    aco_status = row.get('Status_Zona', 'UNKNOWN')
    cnn_impact = row.get('Proporsi_Grid_Terdampak_CNN', 0) * 100
    
    risk_color = '#c0392b' if final_prob >= 70 or is_anomaly else (
                 '#f39c12' if final_prob >= 40 else '#27ae60')

    start_lat = row.get('Lintang', -7.0)
    start_lon = row.get('Bujur', 110.0)
    map_center = [start_lat, start_lon]
    
    m = folium.Map(location=map_center, zoom_start=9, tiles='CartoDB dark_matter')
    
    # ---------------------------------------------
    # 2. VISUALISASI TITIK KEJADIAN TERBARU (INPUT)
    # ---------------------------------------------
    popup_html = f"""
    <b>LIVE EVENT:</b> {row.get('Lokasi','-')}<br>
    M {row.get('Magnitudo',0):.2f} ({row.get('Kedalaman_km',0):.1f} km)<br>
    ---
    <b>RISIKO FINAL:</b> <span style="color:{risk_color};"><b>{final_prob:.1f}%</b></span><br>
    <b>LSTM:</b> { 'ANOMALI' if is_anomaly else 'NORMAL' }
    """
    
    folium.Marker(
        location=[start_lat, start_lon],
        icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa'), 
        tooltip="TITIK KEJADIAN TERBARU (INPUT)",
        popup=popup_html
    ).add_to(m)

    # ---------------------------------------------
    # 3. VISUALISASI AREA RISIKO PREDIKSI (ACO/CNN/Ensemble)
    # ---------------------------------------------
    # Asumsi: Context_Impact_Radius adalah output FE/ACO yang menentukan radius minimum dampak
    radius_km = row.get('Context_Impact_Radius', 15.0) # Default 15 km
    radius_m = max(1000, radius_km * 1000) 
    
    # Lingkaran Area Risiko Final
    folium.Circle(
        location=[start_lat, start_lon],
        radius=radius_m,
        color=risk_color,
        fill=True,
        fill_opacity=0.4, 
        weight=2,
        tooltip=f"AREA RISIKO (Radius {radius_km:.1f} km)"
    ).add_to(m)

    # ---------------------------------------------
    # 4. VISUALISASI VEKTOR PREDIKSI GA (Garis Proyeksi)
    # ---------------------------------------------
    ga_meta = summary.get("ga_meta", {})
    ga_vector = ga_meta.get("vector_data") 
    
    if ga_vector and 'Angle' in ga_vector and 'Distance' in ga_vector:
        angle = ga_vector['Angle']
        distance = ga_vector['Distance']
        
        # Proyeksikan titik akhir vektor
        end_lat, end_lon = GeodesicKernel.project_destination(start_lat, start_lon, angle, distance)

        # Garis Vektor Prediksi (Dari Titik Gempa ke Titik Prediksi)
        folium.PolyLine(
            [[start_lat, start_lon], [end_lat, end_lon]],
            color="#00ffff", # Cyan/Biru Muda
            weight=4,
            dash_array="5, 5",
            tooltip=f"Vektor Prediksi GA: {distance:.1f} km @ {angle:.1f}°"
        ).add_to(m)

        # Titik Prediksi (Ujung Vektor)
        folium.CircleMarker(
            location=[end_lat, end_lon],
            radius=6,
            color='#00ffff',
            fill=True,
            tooltip="TITIK PREDIKSI BERIKUTNYA (GA)"
        ).add_to(m)
        
        logger.info("[LIVE MONITOR] Vektor Prediksi GA divisualisasikan.")

    # ---------------------------------------------
    # 5. ELEMEN VALIDASI (HTML/Teks di Peta)
    # ---------------------------------------------
    
    validation_text = f"""
    <div style='position: fixed; top: 10px; right: 10px; z-index: 9999; background: #272727; color: white; padding: 10px; border-radius: 5px; opacity: 0.9;'>
        <h4>RISIKO STRATEGIS LIVE</h4>
        <hr style='margin: 5px 0; border-color: #555;'>
        <b>Final Prob:</b> <span style='color:{risk_color};'><b>{final_prob:.1f}%</b></span><br>
        <b>LSTM Anomaly:</b> { 'ANOMALI' if is_anomaly else 'NORMAL' }<br>
        <b>CNN Impact:</b> {cnn_impact:.2f}%<br>
        <b>Zonasi ACO:</b> {aco_status}
    </div>
    """
    m.get_root().html.add_child(folium.Element(validation_text))


    # 6. FINAL RENDER
    output_path = config['output_paths']['dashboard_live_monitor'] 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    m.save(output_path)
    logger.info(f"✔ LIVE MONITOR DASHBOARD generated at: {output_path}")