"""
TectonicAI - ACO Engine (Titanium Edition).
Module: Advanced Ant Colony Optimization for Seismic Risk Zoning.

FIXES APPLIED:
- Added `ensure_consistent_size` in key lifecycle stages.
- CRITICAL FIX: Explicitly clamp `node_importance` array size in `_finalize_results` 
  to match DataFrame size (N_DF) to prevent the "operands could not be broadcast 
  together with shapes (2091,) (2090,)" error.
"""

import numpy as np
import pandas as pd
import logging
import os
import time
import math
import json
import pickle
import folium
from folium.plugins import HeatMap
from typing import List, Dict, Any, Tuple, Optional, Union

# --- KONSTANTA ---
R_EARTH_KM = 6371.0
EPSILON = 1e-12
DEFAULT_PHEROMONE = 0.1
MAX_PHEROMONE = 10.0
MIN_PHEROMONE = 0.01

# ==========================================
# 1. GEO UTILITIES
# ==========================================

class GeoMath:
    @staticmethod
    def haversine_vectorized(lat_array, lon_array):
        lat_rad = np.radians(lat_array)
        lon_rad = np.radians(lon_array)
        
        dlat = lat_rad[:, np.newaxis] - lat_rad
        dlon = lon_rad[:, np.newaxis] - lon_rad
        
        a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad) * np.sin(dlon / 2.0)**2
        a = np.clip(a, 0.0, 1.0)
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist_matrix = R_EARTH_KM * c
        
        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix

# ==========================================
# 2. ANT AGENT
# ==========================================

class AntAgent:
    def __init__(self, ant_id: int, start_node: int, alpha: float, beta: float, role: str = "Worker"):
        self.id = ant_id
        self.start_node = start_node
        self.current_node = start_node
        self.role = role
        self.alpha = alpha
        self.beta = beta
        self.path = [start_node]
        self.visited_mask = None
        self.accumulated_risk = 0.0
        
    def reset(self, new_start_node: int, n_nodes: int):
        self.start_node = new_start_node
        self.current_node = new_start_node
        self.path = [new_start_node]
        self.visited_mask = np.zeros(n_nodes, dtype=bool)
        self.visited_mask[new_start_node] = True
        self.accumulated_risk = 0.0

    def move_to(self, next_node: int, risk_val: float):
        self.path.append(next_node)
        self.visited_mask[next_node] = True
        self.current_node = next_node
        self.accumulated_risk += risk_val

# ==========================================
# 3. ENVIRONMENT MANAGER
# ==========================================

class EnvironmentManager:
    def __init__(self, df: pd.DataFrame, logger):
        self.df = df
        self.logger = logger
        self.n_nodes = len(df)
        
        self.dist_matrix = None
        self.heuristic_matrix = None
        self.pheromone_matrix = None
        
        self._build_distance_matrix()
        self._build_heuristic_matrix()
        self._init_pheromone_matrix()

    def _build_distance_matrix(self):
        lats = self.df['Lintang'].values
        lons = self.df['Bujur'].values
        self.dist_matrix = GeoMath.haversine_vectorized(lats, lons)
        self.dist_matrix += EPSILON

    def _build_heuristic_matrix(self):
        mag_col = 'Magnitudo_Original' if 'Magnitudo_Original' in self.df.columns else 'Magnitudo'
        mags = self.df[mag_col].values
        energy = np.power(mags, 2.5)
        
        depth_col = 'Kedalaman_Original' if 'Kedalaman_Original' in self.df.columns else 'Kedalaman_km'
        depths = self.df[depth_col].values
        
        depth_factor = np.log(depths + np.e)
        attractiveness = energy / depth_factor
        
        attr_matrix = np.tile(attractiveness, (self.n_nodes, 1))
        self.heuristic_matrix = attr_matrix / self.dist_matrix
        
        np.fill_diagonal(self.heuristic_matrix, 0.0)
        max_val = np.max(self.heuristic_matrix)
        if max_val > 0:
            self.heuristic_matrix /= max_val

    def _init_pheromone_matrix(self):
        self.pheromone_matrix = np.full((self.n_nodes, self.n_nodes), DEFAULT_PHEROMONE)

    def get_transition_probabilities(self, current_node: int, ant: AntAgent):
        tau = self.pheromone_matrix[current_node]
        eta = self.heuristic_matrix[current_node]
        prob = np.power(tau, ant.alpha) * np.power(eta, ant.beta)
        if ant.visited_mask is not None: # Tambahan safety check untuk mask
             prob[ant.visited_mask] = 0.0
        return prob

    def apply_global_update(self, evaporation_rate: float, deposit_matrix: np.ndarray):
        self.pheromone_matrix *= (1 - evaporation_rate)
        self.pheromone_matrix += deposit_matrix
        self.pheromone_matrix = np.clip(self.pheromone_matrix, MIN_PHEROMONE, MAX_PHEROMONE)

    def reset_pheromone_smooth(self):
        avg = np.mean(self.pheromone_matrix)
        self.pheromone_matrix = (self.pheromone_matrix * 0.5) + (avg * 0.5)
        
    def ensure_consistent_size(self, n_nodes: int):
        """ Pastikan pheromone_matrix berukuran (n_nodes, n_nodes). """
        if self.pheromone_matrix is None or self.pheromone_matrix.shape != (n_nodes, n_nodes):
             self.logger.warning(
                 f"[SAFE] Pheromone matrix rebuilt from scratch due to size mismatch: {self.n_nodes} -> {n_nodes}"
             )
             self.n_nodes = n_nodes
             self._init_pheromone_matrix()


# ==========================================
# 4. MAIN ENGINE
# ==========================================

class ACOEngine:

    def __init__(self, config: dict):
        self.aco_cfg = config
        self.logger = logging.getLogger("ACO_Engine_Master")
        self.logger.setLevel(logging.DEBUG)
        
        self._load_parameters()
        
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output'))
        self.output_paths = {
            'aco_zoning_excel': os.path.join(base_path, 'aco_results/aco_zoning_data_for_lstm.xlsx'),
            'aco_epicenters_csv': os.path.join(base_path, 'aco_results/aco_epicenters.csv'),
            'aco_state_file': os.path.join(base_path, 'aco_results/aco_brain_state.pkl'),
            'aco_impact_html': os.path.join(base_path, 'aco_results/aco_impact_zones.html')
        }
        os.makedirs(os.path.dirname(self.output_paths['aco_zoning_excel']), exist_ok=True)
        
        self.env_manager: Optional[EnvironmentManager] = None
        self.colony: List[AntAgent] = []
        self.best_global_score: float = -np.inf
        self.stagnation_counter: int = 0

    def _load_parameters(self):
        self.n_ants = int(self.aco_cfg.get('n_ants', 50))
        self.n_iterations = int(self.aco_cfg.get('n_iterations', 100))
        self.n_steps = int(self.aco_cfg.get('n_epicenters', 20))
        self.alpha_base = float(self.aco_cfg.get('alpha', 1.0))
        self.beta_base = float(self.aco_cfg.get('beta', 2.0))
        self.rho_base = float(self.aco_cfg.get('evaporation_rate', 0.1))
        self.Q = float(self.aco_cfg.get('pheromone_deposit', 100.0))
        self.risk_threshold = float(self.aco_cfg.get('risk_threshold', 0.7))

    def _initialize_colony(self, n_nodes: int):
        self.colony = []
        start_probs = self.env_manager.heuristic_matrix.sum(axis=1)
        
        total = start_probs.sum()
        if total <= 0 or not np.isfinite(total):
            self.logger.warning("[SAFE] Menggunakan distribusi uniform sebagai fallback start_probs.")
            start_probs = np.ones(n_nodes, dtype=np.float64) / float(n_nodes)
        else:
            start_probs /= total
        
        for i in range(self.n_ants):
            if i < int(self.n_ants * 0.2):
                role = "Explorer"; alpha = self.alpha_base * 0.5; beta = self.beta_base * 1.5
            else:
                role = "Exploiter"; alpha = self.alpha_base * 1.2; beta = self.beta_base * 0.9
            
            start_node = np.random.choice(np.arange(n_nodes), p=start_probs)
            start_node = int(np.clip(start_node, 0, n_nodes - 1)) # Safety clip
            
            ant = AntAgent(i, start_node, alpha, beta, role)
            ant.reset(start_node, n_nodes)
            self.colony.append(ant)

    def _step_ants(self):
        active_ants = 0
        for ant in self.colony:
            if ant.current_node == -1 or ant.current_node is None:
                continue
            
            probs = self.env_manager.get_transition_probabilities(ant.current_node, ant)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0) # Safety NaN/Inf
            total = np.sum(probs)
            
            if total <= 0:
                ant.current_node = -1
                continue
            
            norm = probs / total
            try:
                next_node = np.random.choice(np.arange(self.env_manager.n_nodes), p=norm)
            except ValueError:
                 self.logger.warning("[SAFE] Probabilitas transisi tidak valid. Menggunakan random choice uniform.")
                 next_node = np.random.choice(np.arange(self.env_manager.n_nodes))
            
            next_node = int(np.clip(next_node, 0, self.env_manager.n_nodes - 1))
            
            risk_val = self.env_manager.heuristic_matrix[ant.current_node, next_node]
            ant.move_to(next_node, risk_val)
            active_ants += 1
            
        return active_ants

    def _manage_lifecycle(self, iteration: int):
        deposit_matrix = np.zeros_like(self.env_manager.pheromone_matrix)
        iter_best = -np.inf
        
        for ant in self.colony:
            score = ant.accumulated_risk
            iter_best = max(iter_best, score)
            
            if score > 0:
                dep = self.Q * score
                for k in range(len(ant.path) - 1):
                    u, v = ant.path[k], ant.path[k+1]
                    # Boundary check (sudah ada di kode lama, kita pertahankan)
                    if u >= self.env_manager.n_nodes or v >= self.env_manager.n_nodes or u < 0 or v < 0:
                         continue
                         
                    deposit_matrix[u, v] += dep
                    deposit_matrix[v, u] += dep

        current_rho = self.rho_base
        if self.stagnation_counter > 5:
            current_rho = min(0.9, self.rho_base * 1.5)
        
        self.env_manager.apply_global_update(current_rho, deposit_matrix)
        
        if iter_best > self.best_global_score:
            self.best_global_score = iter_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        if self.stagnation_counter > (self.n_iterations * 0.2):
            self.env_manager.reset_pheromone_smooth()
            self.stagnation_counter = 0
        
        # Reset ulang semua semut untuk iterasi berikutnya
        start_probs = self.env_manager.heuristic_matrix.sum(axis=1)
        total = start_probs.sum()
        n_nodes = self.env_manager.n_nodes
        
        if total <= 0 or not np.isfinite(total):
             start_probs = np.ones(n_nodes, dtype=np.float64) / float(n_nodes)
        else:
             start_probs /= total
             
        for ant in self.colony:
            new_start = np.random.choice(np.arange(self.env_manager.n_nodes), p=start_probs)
            ant.reset(new_start, self.env_manager.n_nodes)

    def run(self, df: pd.DataFrame):
        if df.empty:
            self.logger.warning("ACOEngine.run dipanggil dengan df kosong. Skip.")
            return df, {}
        
        # Reset index untuk konsistensi
        df = df.reset_index(drop=True)
        n_nodes = len(df)

        # ✅ Fallback: jika titik terlalu sedikit, jangan paksa ACO rumit
        if n_nodes < 3:
            self.logger.warning(
                f"[SAFE] Jumlah node sangat sedikit (N={n_nodes}). "
                f"ACO memakai fallback default."
            )
            df_out = df.copy()
            df_out['Pheromone_Score'] = 0.0
            df_out['Status_Zona'] = "Unknown"
            # radius sederhana (agar tetap ada untuk visual/lainnya)
            mag_col = 'Magnitudo_Original' if 'Magnitudo_Original' in df_out.columns else 'Magnitudo'
            mags = df_out[mag_col]
            df_out['Radius_Visual_KM'] = (mags ** 1.2).fillna(0.0)
            # tambahkan juga Context_Impact_Radius biar konsisten ke hilir
            df_out['Context_Impact_Radius'] = df_out['Radius_Visual_KM']
            # simpan output (pakai helper yang sudah ada)
            self._save_to_disk(df_out)
            self._generate_visuals(df_out)
            return df_out, {"info": "ACO fallback - node terlalu sedikit"}

        # ====== LANJUTKAN KODE LAMA-MU DI BAWAH INI (TIDAK DIUBAH) ======
        self.env_manager = EnvironmentManager(df, self.logger)
        
        # --- LOAD STATE DENGAN CHECK UKURAN (Tambahan dari kode lama) ---
        if os.path.exists(self.output_paths['aco_state_file']):
            try:
                with open(self.output_paths['aco_state_file'], 'rb') as f:
                    state = pickle.load(f)
                old_matrix = state.get('pheromone_matrix')
                if old_matrix is not None and old_matrix.shape == (len(df), len(df)):
                    self.env_manager.pheromone_matrix = old_matrix
                    self.logger.info(f"Memuat state ACO lama dengan shape {old_matrix.shape}.")
                else:
                    self.logger.warning(
                        f"State ACO lama diabaikan karena shape mismatch. "
                        f"Expected ({len(df)}, {len(df)}), got {old_matrix.shape if old_matrix is not None else 'None'}."
                    )
            except Exception as e:
                self.logger.error(f"Gagal memuat state ACO: {e}. Menggunakan matrix baru.")
        
        self.env_manager.ensure_consistent_size(len(df))  # Pastikan ukuran setelah load/init
        
        self._initialize_colony(self.env_manager.n_nodes)
        
        for it in range(self.n_iterations):
            for _ in range(self.n_steps - 1):
                if self._step_ants() == 0:
                    break
            self._manage_lifecycle(it)

        # Simpan state baru
        try:
            self.env_manager.ensure_consistent_size(len(df))  # Double check sebelum simpan
            with open(self.output_paths['aco_state_file'], 'wb') as f:
                pickle.dump({'pheromone_matrix': self.env_manager.pheromone_matrix.astype(np.float64)}, f)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan state ACO: {e}")
        
        return self._finalize_results(df)

    # ==========================================
    # 6. FINAL OUTPUT + RADIUS FIXED (CRITICAL FIX)
    # ==========================================

    def _finalize_results(self, df):
        n_df = len(df)
        self.env_manager.ensure_consistent_size(n_df)

    # ==========================================
    # HITUNG NODE IMPORTANCE
    # ==========================================
        node_importance = np.sum(self.env_manager.pheromone_matrix, axis=0)

    # 🚨 FIX KRITIS SHAPE MISMATCH 🚨
        if node_importance.shape[0] != n_df:
            self.logger.critical(
                f"❌ SHAPE MISMATCH KRITIS! Output ACO {node_importance.shape[0]} != Input DF {n_df}. MEMANGKAS ARRAY."
        )
            node_importance = node_importance[:n_df]

    # ==========================================
    # NORMALISASI PHEROMONE SCORE
    # ==========================================
        q01 = np.quantile(node_importance, 0.01)
        q99 = np.quantile(node_importance, 0.99)
        clipped = np.clip(node_importance, q01, q99)

        if q99 > q01:
            norm_scores = (clipped - q01) / (q99 - q01)
        else:
            norm_scores = np.zeros_like(clipped)

        df_out = df.copy()
        df_out['Pheromone_Score'] = norm_scores

        df_out['Status_Zona'] = df_out['Pheromone_Score'].apply(
            lambda x: 'Terdampak' if x >= self.risk_threshold else 'Aman'
        )

    # ==========================================
    # RADIUS DAMPAK (VISUAL & NUMERIK)
    # ==========================================
        mags = df_out['Magnitudo_Original'] if 'Magnitudo_Original' in df_out else df_out['Magnitudo']
        pher = df_out['Pheromone_Score']

        base_r = mags ** 1.5
        df_out['Radius_Visual_KM'] = base_r * (1 + pher * 0.3)

    # ==========================================
    # ACO NUMERIC SUMMARY (WEIGHTED CENTER & IMPACT)
    # ==========================================
        try:
            # === WEIGHTED CENTROID BERDASARKAN PHEROMONE ===
            pher = np.asarray(df_out['Pheromone_Score'].fillna(0.0).astype(float))
            lats = np.asarray(df_out['Lintang'].astype(float))
            lons = np.asarray(df_out['Bujur'].astype(float))

            if pher.sum() > 0:
                center_lat = float(np.sum(lats * pher) / pher.sum())
                center_lon = float(np.sum(lons * pher) / pher.sum())
            else:
                # fallback aman
                center_lat = float(lats.mean())
                center_lon = float(lons.mean())

            # === HITUNG RADIUS DAMPAK BERDASARKAN JARAK TERJAUH ===
            def _haversine_scalar_array(lat_arr, lon_arr, lat_c, lon_c):
                lat1 = np.radians(lat_arr)
                lon1 = np.radians(lon_arr)
                lat2 = np.radians(lat_c)
                lon2 = np.radians(lon_c)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                return 6371.0 * c

            dists = _haversine_scalar_array(lats, lons, center_lat, center_lon)
            impact_radius_km = float(np.max(dists))
            impact_area_km2 = math.pi * (impact_radius_km ** 2)

            # === SIMPAN KE DF (PENTING UNTUK LSTM) ===
            df_out['ACO_Center_Lat'] = center_lat
            df_out['ACO_Center_Lon'] = center_lon
            df_out['ACO_Impact_Radius_km'] = impact_radius_km

            # === SIMPAN SUMMARY JSON ===
            aco_summary = {
                "center_latitude": round(center_lat, 6),
                "center_longitude": round(center_lon, 6),
                "impact_radius_km": round(impact_radius_km, 2),
                "impact_area_km2": round(impact_area_km2, 2),
                "total_events": int(len(df_out))
            }

            summary_path = os.path.join(
                os.path.dirname(self.output_paths['aco_impact_html']),
                "aco_summary.json"
            )

            with open(summary_path, "w") as f:
                json.dump(aco_summary, f, indent=4)

            self.logger.info("ACO weighted center & impact summary berhasil dibuat")

        except Exception as e:
            self.logger.error(f"Gagal menghitung ACO weighted center: {e}")

    # ==========================================
    # SIMPAN FILE & VISUAL
    # ==========================================
        self._save_to_disk(df_out)
        self._generate_visuals(df_out)

        return df_out, {
        "pheromone_matrix": self.env_manager.pheromone_matrix,
        "aco_summary": aco_summary if 'aco_summary' in locals() else None
    }


    # ==========================================
    # 7. SAVE FILES (Tetap sama)
    # ==========================================

    def _save_to_disk(self, df):
        for c in ['ACO_Center_Lat', 'ACO_Center_Lon', 'ACO_Impact_Radius_km']:
            if c not in df.columns:
                df[c] = np.nan
        mag_col = 'Magnitudo_Original' if 'Magnitudo_Original' in df.columns else 'Magnitudo'
        depth_col = 'Kedalaman_Original' if 'Kedalaman_Original' in df.columns else 'Kedalaman_km'
        
        cols = [
            'Tanggal', 'Lintang', 'Bujur', mag_col, depth_col, 'Lokasi',
            'Pheromone_Score', 'Status_Zona', 'Radius_Visual_KM',
            'ACO_Center_Lat', 'ACO_Center_Lon', 'ACO_Impact_Radius_km'
        ]
        final_df = df[cols].rename(columns={mag_col: 'Magnitudo', depth_col: 'Kedalaman'})
        
        try:
            final_df.to_excel(self.output_paths['aco_zoning_excel'], index=False)
            final_df.to_csv(self.output_paths['aco_epicenters_csv'], index=False)
            self.logger.info(f"ACO zoning disimpan ke: {self.output_paths['aco_zoning_excel']}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan output: {e}")

    # ==========================================
    # 8. VISUAL FIXED (CIRCLE KUNING + POPUP) (Tetap sama)
    # ==========================================

    def _generate_visuals(self, df: pd.DataFrame):
        """
        VISUAL BARU:
        - Heatmap dihapus.
        - Hanya circle warna orange.
        - Ditambah titik pusat (marker kecil).
        - Popup tetap informatif.
        """
        try:
            center = [df['Lintang'].mean(), df['Bujur'].mean()]
            m = folium.Map(location=center, zoom_start=7, tiles='CartoDB positron')
            m.save(self.output_paths['aco_impact_html'])
            self.logger.info(f"Visual ACO Tersimpan dan Ditimpa: {self.output_paths['aco_impact_html']}")

            # === LOOP untuk setiap episenter ===
            for _, r in df.iterrows():

                # radius KM → meter
                radius_km = float(r['Radius_Visual_KM'])
                radius_km = min(radius_km, 20)
                radius_m = radius_km * 100

                lokasi = r.get('Lokasi', '-')
                tanggal = r.get('Tanggal', '-')
                mag = r.get('Magnitudo', r.get('Magnitudo_Original', '-'))
                depth = r.get('Kedalaman', r.get('Kedalaman_km', '-'))
                rad = round(r['Radius_Visual_KM'], 2)

                popup_html = f"""
                <b>Lokasi:</b> {lokasi}<br>
                <b>Tanggal:</b> {tanggal}<br>
                <b>Magnitudo:</b> {mag}<br>
                <b>Kedalaman:</b> {depth} km<br>
                <b>Radius Prediksi:</b> {rad} km<br>
                """

                popup = folium.Popup(popup_html, max_width=300)

                # === CIRCLE ZONA ===
                folium.Circle(
                    location=[r['Lintang'], r['Bujur']],
                    radius=radius_m,
                    color='orange',
                    fill=True,
                    fill_opacity=0.25,
                    weight=1.5,
                    popup=popup
                ).add_to(m)

                # === TITIK PUSAT EPISENTER ===
                folium.CircleMarker(
                    location=[r['Lintang'], r['Bujur']],
                    radius=1,
                    color='Red',
                    fill=True,
                    fill_opacity=0.9
                ).add_to(m)

                # === TITIK PUSAT AREA DAMPAK (ACO CENTER) ===
                folium.Marker(
                    location=[df['Lintang'].mean(), df['Bujur'].mean()],
                    popup=folium.Popup(
                        f"""
                        <b>Pusat Area Terdampak (ACO)</b><br>
                        Total Event: {len(df)}<br>
                        Luas Area: cek aco_summary.json
                        """,
                        max_width=250
                    ),
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)


            # simpan file
            m.save(self.output_paths['aco_impact_html'])
            self.logger.info(f"Visual Baru Tersimpan: {self.output_paths['aco_impact_html']}")

        except Exception as e:
            self.logger.error(f"Gagal membuat visual ACO: {e}")
            self.logger.critical(f"GAGAL MENYIMPAN VISUAL ACO: {e}. File mungkin dikunci!")