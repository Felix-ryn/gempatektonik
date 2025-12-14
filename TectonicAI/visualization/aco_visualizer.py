import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import os
import logging

# Setup Logger
logger = logging.getLogger("ACO_Visualizer")
logging.basicConfig(level=logging.INFO)

class ACOVisualizer:
    def __init__(self, config):
        self.config = config
        # Path ke file Excel hasil perhitungan ACO
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output'))
        self.aco_data_path = os.path.join(base_path, 'aco_results/aco_zoning_data_for_lstm.xlsx')
        self.map_output_path = os.path.join(base_path, 'aco_results/tectonic_risk_map.html')

    def generate_map(self):
        logger.info("Memulai generasi Peta Visualisasi Tektonik ACO...")
        
        # 1. Load Data Hasil Perhitungan ACO
        if not os.path.exists(self.aco_data_path):
            logger.error(f"File data ACO tidak ditemukan: {self.aco_data_path}. Jalankan ACO Engine dulu.")
            return
        
        df = pd.read_excel(self.aco_data_path)
        logger.info(f"Data dimuat: {len(df)} titik gempa.")

        # 2. Inisialisasi Peta Dasar (Gunakan peta gelap agar warna 'panas' terlihat jelas)
        avg_lat = df['Lintang'].mean()
        avg_lon = df['Bujur'].mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, tiles='CartoDB dark_matter')

        # --- LAYER 1: TECTONIC STRESS HEATMAP ---
        # Heatmap menunjukkan akumulasi tegangan berdasarkan jejak pheromone
        heatmap_data = df[['Lintang', 'Bujur', 'Pheromone_Score']].values.tolist()
        
        HeatMap(
            heatmap_data,
            name='Tectonic Stress Heatmap',
            radius=25,     # Radius blur perpindahan panas
            blur=15,       # Tingkat kehalusan
            max_zoom=10,
            # Gradien warna dari dingin (biru) ke panas (merah menyala)
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)

        # --- LAYER 2 & 3: ZONED MARKERS & DYNAMIC RADIUS ---
        # Kita gunakan FeatureGroup agar bisa di-toggle on/off
        risk_layer = folium.FeatureGroup(name="Titik & Zona Risiko Terdampak")
        safe_layer = folium.FeatureGroup(name="Titik Aman (Historis)", show=False) # Default hidden

        for _, row in df.iterrows():
            # Siapkan Popup Konten
            popup_html = f"""
            <div style="font-family: Arial; width: 180px;">
                <h4>{row['Lokasi']}</h4>
                <b>Status: {row['Status_Zona']}</b><br>
                Magnitudo: {row['Magnitudo']}<br>
                Kedalaman: {row['Kedalaman']} km<br>
                <b>Skor Pheromone: {row['Pheromone_Score']:.4f}</b><br>
                Radius Dampak Visual: {row.get('Radius_Dampak_Visual_KM', 0):.1f} km
            </div>
            """
            popup = folium.Popup(popup_html, max_width=250)

            if row['Status_Zona'] == 'Terdampak':
                # --- Visualisasi Zona MERAH (Bahaya) ---
                
                # A. Marker Titik Pusat
                folium.CircleMarker(
                    location=[row['Lintang'], row['Bujur']],
                    radius=5 + (row['Magnitudo'] * 1.5), # Ukuran marker berdasarkan Mag
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.8,
                    popup=popup
                ).add_to(risk_layer)
                
                # B. Lingkaran Radius Dampak Dinamis (Hasil Perhitungan ACO)
                radius_km = row.get('Radius_Dampak_Visual_KM', 0)
                if radius_km > 0:
                    folium.Circle(
                        location=[row['Lintang'], row['Bujur']],
                        radius=radius_km * 1000, # Folium butuh meter
                        color='#ff3333', # Merah agak transparan
                        weight=1,
                        fill=True,
                        fill_opacity=0.2, # Transparan agar peta di bawahnya terlihat
                        tooltip=f"Zona Dampak Estimasi: {radius_km:.1f} km"
                    ).add_to(risk_layer)
                    
            else:
                # --- Visualisasi Zona HIJAU (Aman) ---
                # Marker lebih kecil dan warna tidak mencolok
                folium.CircleMarker(
                    location=[row['Lintang'], row['Bujur']],
                    radius=3, 
                    color='green',
                    fill=True,
                    fill_opacity=0.5,
                    popup=popup
                ).add_to(safe_layer)

        # Tambahkan layer ke peta
        risk_layer.add_to(m)
        safe_layer.add_to(m)

        # Tambahkan kontrol layer agar user bisa memilih tampilan
        folium.LayerControl(collapsed=False).add_to(m)

        # Simpan Peta
        os.makedirs(os.path.dirname(self.map_output_path), exist_ok=True)
        m.save(self.map_output_path)
        logger.info(f"Visualisasi Peta Tektonik berhasil disimpan: {self.map_output_path}")
        print(f"Peta siap! Buka file ini di browser: {self.map_output_path}")

# --- CARA PENGGUNAAN (Contoh untuk ditest terpisah) ---
if __name__ == "__main__":
    # Config dummy, path sudah di-hardcode di dalam class agar mandiri
    dummy_config = {} 
    viz = ACOVisualizer(dummy_config)
    viz.generate_map()