"""
TectonicAI - GA Engine (Titanium Edition).
Module: Evolutionary Tectonic Vector Prediction System.

Architecture: Multi-Island Distributed Genetic Algorithm (MIGA).
Core Task: Predict the propagation vector (Angle & Distance) of tectonic stress.
"""

import numpy as np
import pandas as pd
import logging
import os
import time
import math
import pickle  # disimpan untuk kompatibilitas bila nanti dipakai lagi
import folium
from typing import List, Dict, Any, Tuple, Optional

# --- GLOBAL CONSTANTS ---
R_EARTH_KM = 6371.0
EPSILON = 1e-9

# Batas fokus Jawa Timur (approx)
EJ_MIN_LAT = -8.8
EJ_MAX_LAT = -6.5
EJ_MIN_LON = 111.0
EJ_MAX_LON = 114.6

# Setup Logger
logger = logging.getLogger("GA_Titan_Engine")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
# ==========================================
# 1. MATH & PHYSICS KERNEL
# ==========================================


class GeodesicKernel:
    """Inti matematika untuk perhitungan geodesi pada permukaan bola bumi."""

    @staticmethod
    def project_destination(
        lat: float, lon: float, bearing: float, distance_km: float
    ) -> Tuple[float, float]:
        lat_rad, lon_rad, bear_rad = map(math.radians, [lat, lon, bearing])
        angular_dist = distance_km / R_EARTH_KM

        lat_dest = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist)
            + math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bear_rad)
        )
        lon_dest = lon_rad + math.atan2(
            math.sin(bear_rad) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(lat_dest),
        )

        return math.degrees(lat_dest), math.degrees(lon_dest)

    @staticmethod
    def haversine_vectorized(lat1_arr, lon1_arr, lat2_scalar, lon2_scalar):
        lat1 = np.radians(lat1_arr)
        lon1 = np.radians(lon1_arr)
        lat2 = np.radians(lat2_scalar)
        lon2 = np.radians(lon2_scalar)
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return R_EARTH_KM * c

    @staticmethod
    def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Menghitung bearing dari titik 1 ke titik 2 (0–360 derajat)."""
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_r - lon1_r
        y = math.sin(dlon) * math.cos(lat2_r)
        x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
            lat2_r
        ) * math.cos(dlon)
        brng = math.degrees(math.atan2(y, x))
        return (brng + 360.0) % 360.0
    

# ==========================================
# 2. EVOLUTIONARY DATA STRUCTURES
# ==========================================


class Genome:
    """Representasi genetik satu solusi vektor."""

    def __init__(self, lat, lon, angle, dist):
        self.lat = lat
        self.lon = lon
        self.angle = angle
        self.dist = dist
        # sigma untuk self-adaptive mutation
        self.sigma = np.array([0.05, 0.05, 5.0, 20.0])

    def to_array(self):
        return np.array([self.lat, self.lon, self.angle, self.dist])


class Individual:
    """Individu dalam populasi GA."""

    def __init__(self, genome: Genome):
        self.genome = genome
        self.fitness = 0.0
        self.is_evaluated = False
        self.age = 0

    def clone(self):
        new_g = Genome(
            self.genome.lat, self.genome.lon, self.genome.angle, self.genome.dist
        )
        new_g.sigma = self.genome.sigma.copy()
        new_ind = Individual(new_g)
        new_ind.fitness = self.fitness
        new_ind.is_evaluated = self.is_evaluated
        new_ind.age = self.age
        return new_ind


# ==========================================
# 3. FITNESS EVALUATOR
# ==========================================


class TectonicFitnessEvaluator:
    def __init__(self, df_context: pd.DataFrame):
        df_valid = df_context.dropna(subset=["Magnitudo", "Lintang", "Bujur"]).copy()
        mag_col = (
            "Magnitudo_Original"
            if "Magnitudo_Original" in df_valid.columns
            else "Magnitudo"
        )
        self.ref_lats = df_valid["Lintang"].values
        self.ref_lons = df_valid["Bujur"].values

        mags = df_valid[mag_col].values
        mags[mags <= 0] = EPSILON
        base_weights = np.power(mags, 2.5)

        # ----- NEW: incorporate Pheromone_Score if present -----
        if "Pheromone_Score" in df_valid.columns:
            pher = df_valid["Pheromone_Score"].fillna(0.0).values
            pher = np.clip(pher, 0.0, 1.0)  # normalize assumption
            pher_factor = 1.0 + 2.0 * pher  # tuning: pher influence factor
            self.ref_weights = base_weights * pher_factor
        else:
            self.ref_weights = base_weights

        self.total_weight = np.sum(self.ref_weights) + EPSILON
        if self.total_weight < 1.0:
            self.ref_weights = np.ones_like(self.ref_lats)
            self.total_weight = len(self.ref_weights) + EPSILON

    def evaluate(self, ind: Individual) -> float:
        # Proyeksi ujung vektor
        end_lat, end_lon = GeodesicKernel.project_destination(
            ind.genome.lat, ind.genome.lon, ind.genome.angle, ind.genome.dist
        )
        dists = GeodesicKernel.haversine_vectorized(
            self.ref_lats, self.ref_lons, end_lat, end_lon
        )
        mse_score = np.sum((dists**2) * self.ref_weights) / self.total_weight

        # Alignment terhadap kecenderungan fault lokal
        idx_nearest = np.argsort(dists)[:10]
        if len(idx_nearest) >= 3:
            try:
                slope, _ = np.polyfit(
                    self.ref_lons[idx_nearest], self.ref_lats[idx_nearest], 1
                )
                fault_angle = (math.degrees(math.atan(slope)) + 360) % 360
            except Exception:
                fault_angle = 0.0

            angle_diff = abs(ind.genome.angle - fault_angle) % 180
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            alignment_penalty = angle_diff * 0.5
        else:
            alignment_penalty = 50.0

        fitness = 1000.0 / (1.0 + mse_score + alignment_penalty)
        return fitness


# ==========================================
# 4. POPULATION ISLAND
# ==========================================


class PopulationIsland:
    """Pulau populasi terisolasi (MIGA)."""

    def __init__(
        self,
        island_id: int,
        config: dict,
        bounds: dict,
        evaluator: TectonicFitnessEvaluator,
    ):
        self.id = island_id
        self.cfg = config
        self.bounds = bounds
        self.evaluator = evaluator
        self.pop_size = int(config.get("population_size", 100))
        self.mutation_rate = float(config.get("mutation_rate", 0.05))

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.stagnation_count = 0

    def initialize(self, seed_df: pd.DataFrame):
        self.population = []
        lats = seed_df["Lintang"].values
        lons = seed_df["Bujur"].values

        for _ in range(self.pop_size):
            # start dari titik historis Jatim, bukan random bebas (80% dari populasi)
            if np.random.rand() < 0.8 and len(lats) > 0:
                idx = np.random.randint(len(lats))
                s_lat = float(lats[idx] + np.random.normal(0, 0.05))
                s_lon = float(lons[idx] + np.random.normal(0, 0.05))
            else:
                # penting: fallback ke random dalam bounds kalau tidak ada titik historis
                s_lat = np.random.uniform(
                    self.bounds["min_lat"], self.bounds["max_lat"]
                )
                s_lon = np.random.uniform(
                    self.bounds["min_lon"], self.bounds["max_lon"]
                )

            genome = Genome(
                s_lat, s_lon, np.random.uniform(0, 360), np.random.uniform(10, 500)
            )
            self.population.append(Individual(genome))

    def evaluate_generation(self):
        current_best = -1.0
        for ind in self.population:
            if not ind.is_evaluated:
                ind.fitness = self.evaluator.evaluate(ind)
                ind.is_evaluated = True

            if ind.fitness > current_best:
                current_best = ind.fitness

            if (
                self.best_individual is None
                or ind.fitness > self.best_individual.fitness
            ):
                self.best_individual = ind.clone()

    def reproduce_next_generation(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        next_gen: List[Individual] = []

        elite_count = max(1, int(self.pop_size * 0.05))
        next_gen.extend([ind.clone() for ind in self.population[:elite_count]])

        while len(next_gen) < self.pop_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()
            c1, c2 = self._crossover_blend(p1, p2)
            self._mutate_adaptive(c1)
            self._mutate_adaptive(c2)
            next_gen.extend([c1, c2])

        self.population = next_gen[: self.pop_size]

    def _tournament_select(self, k: int = 3) -> Individual:
        n = len(self.population)
        k = min(k, n)
        idxs = np.random.choice(n, size=k, replace=False)
        pool = [self.population[i] for i in idxs]
        return max(pool, key=lambda x: x.fitness)

    def _crossover_blend(
        self, p1: Individual, p2: Individual
    ) -> Tuple[Individual, Individual]:
        alpha = 0.5
        g1 = p1.genome.to_array()
        g2 = p2.genome.to_array()

        c1_arr = np.zeros(4)
        c2_arr = np.zeros(4)

        for i in range(4):
            mn, mx = min(g1[i], g2[i]), max(g1[i], g2[i])
            rng = mx - mn
            low, high = mn - rng * alpha, mx + rng * alpha

            c1_arr[i] = np.random.uniform(low, high)
            c2_arr[i] = np.random.uniform(low, high)

        def clamp_vals(arr):
            arr[0] = np.clip(arr[0], self.bounds["min_lat"], self.bounds["max_lat"])
            arr[1] = np.clip(arr[1], self.bounds["min_lon"], self.bounds["max_lon"])
            arr[2] = arr[2] % 360.0
            arr[3] = max(1.0, arr[3])
            return arr

        c1_arr = clamp_vals(c1_arr)
        c2_arr = clamp_vals(c2_arr)
        child1 = Individual(Genome(*c1_arr))
        child2 = Individual(Genome(*c2_arr))
        child1.genome.sigma = (p1.genome.sigma + p2.genome.sigma) / 2.0
        child2.genome.sigma = child1.genome.sigma.copy()
        return child1, child2

    def _mutate_adaptive(self, ind: Individual):
        if np.random.rand() < self.mutation_rate:
            tau = 1.0 / np.sqrt(2 * np.sqrt(4))
            ind.genome.sigma *= np.exp(tau * np.random.normal(0, 1, 4))

            if self.best_individual is not None:
                if ind.fitness > (self.best_individual.fitness * 0.9):
                    ind.genome.sigma *= 0.7

            noise = np.random.normal(0, ind.genome.sigma)
            new_vals = ind.genome.to_array() + noise

            # clamp tetap di dalam Jatim
            new_vals[0] = np.clip(
                new_vals[0], self.bounds["min_lat"], self.bounds["max_lat"]
            )
            new_vals[1] = np.clip(
                new_vals[1], self.bounds["min_lon"], self.bounds["max_lon"]
            )
            new_vals[2] = new_vals[2] % 360.0
            new_vals[3] = max(1.0, new_vals[3])

            ind.genome.sigma = np.clip(ind.genome.sigma, 1e-3, 50.0)

            ind.genome = Genome(*new_vals)
            ind.is_evaluated = False

    def trigger_catastrophe(self, seed_data: pd.DataFrame):
        logger.warning(f"Pulau {self.id} stagnan → reset populasi!")
        elite = self.best_individual.clone()
        elite.is_evaluated = True
        self.initialize(seed_data)
        self.population[0] = elite


# ==========================================
# 5. EXPORTER & VISUALIZER  (SNAKE + CHAOS + FINAL)
# ==========================================


class GAExporter:
    """Menangani output file dan visualisasi."""

    def __init__(self, output_paths: dict):
        self.paths = output_paths

    def save_trace_history(self, history: List[dict]):
        # TINGGALKAN HISTORY ASLI
        df = pd.DataFrame(history)

        # 🚨 FIX KRITIS: Pastikan Kunci Prediksi Ada, Jika Tidak Ada di History, Tambahkan Fallback

        # Tambahkan kolom-kolom yang diharapkan Dashboard Generator jika hilang
        if "Pred_Lat" not in df.columns:
            df["Pred_Lat"] = df.get("End_Lat", np.nan)
        if "Pred_Lon" not in df.columns:
            df["Pred_Lon"] = df.get("End_Lon", np.nan)
        if "Angle_Deg" not in df.columns:
            df["Angle_Deg"] = df.get("Angle", np.nan)

        # Tambahkan Is_Best (diperkirakan hilang karena GA tidak menemukan Global Best)
        if "Is_Best" not in df.columns:
            # Karena Global Best Fit selalu nan, kita set semua False atau hanya yang terakhir True
            if not df.empty:
                df["Is_Best"] = [False] * (len(df) - 1) + [True]
            else:
                df["Is_Best"] = False

        # ✅ FIX SAFETY: Tambahkan kolom Magnitudo/Kedalaman Original (jika ada di history)
        if "Magnitudo" not in df.columns:
            df["Magnitudo"] = 0.0
        if "Kedalaman_km" not in df.columns:
            df["Kedalaman_km"] = 0.0

        try:
            # Simpan hasil trace yang sudah di-fix kolomnya
            df.to_excel(self.paths["ga_prediction_excel"], index=False)
            df.tail(1).to_csv(self.paths["ga_best_chrom_csv"], index=False)
            logger.info(f"Jejak evolusi disimpan: {self.paths['ga_prediction_excel']}")
        except Exception as e:
            logger.error(f"Gagal menyimpan jejak evolusi (GAExporter): {e}")

    def _build_nearest_neighbor_order(self, df: pd.DataFrame) -> List[int]:
        """Bangun urutan titik dengan greedy nearest-neighbor (Opsi B)."""
        n = len(df)
        if n == 0:
            return []
        unused = set(range(n))
        # mulai dari magnitudo terbesar
        mags = df["Magnitudo"].values
        current = int(np.argmax(mags))
        order = [current]
        unused.remove(current)

        while unused:
            cur_lat = df.loc[current, "Lintang"]
            cur_lon = df.loc[current, "Bujur"]
            idx_list = np.array(list(unused))
            dists = GeodesicKernel.haversine_vectorized(
                df.loc[idx_list, "Lintang"].values,
                df.loc[idx_list, "Bujur"].values,
                cur_lat,
                cur_lon,
            )
            j = int(idx_list[int(np.argmin(dists))])
            order.append(j)
            unused.remove(j)
            current = j
        return order

    def generate_visual_map(self, history: List[dict], context_df: pd.DataFrame):
        # prefer context_df (historical points) for mapping
        try:
            # optional: baca hasil GA jika ingin menampilkan chromosomal trace
            df_ga = pd.read_excel(self.paths["ga_prediction_excel"]).reset_index(
                drop=True
            )
        except Exception as e:
            logger.debug(f"GA prediction excel not available or unreadable: {e}")
            df_ga = pd.DataFrame()

        if context_df is None or context_df.empty:
            logger.warning("Context DF kosong — skip visual GA.")
            return

        # gunakan context_df sebagai sumber titik (Lintang/Bujur)
        df = context_df.reset_index(drop=True).copy()

        # Urutan Snake berdasarkan NN dari df (historical)
        order = self._build_nearest_neighbor_order(df)

        # Hitung angle & distance per segmen (untuk popup)
        seg_angle = np.full(len(df), np.nan)
        seg_dist = np.full(len(df), np.nan)

        for i in range(len(order) - 1):
            idx_a = order[i]
            idx_b = order[i + 1]
            la, lo = df.loc[idx_a, "Lintang"], df.loc[idx_a, "Bujur"]
            lb, lb_lon = df.loc[idx_b, "Lintang"], df.loc[idx_b, "Bujur"]
            angle_ab = GeodesicKernel.bearing_between(la, lo, lb, lb_lon)
            dist_ab = GeodesicKernel.haversine_vectorized(
                np.array([la]), np.array([lo]), lb, lb_lon
            )[0]
            seg_angle[idx_a] = angle_ab
            seg_dist[idx_a] = dist_ab

        df["Seg_Angle"] = seg_angle
        df["Seg_Distance_km"] = seg_dist

        # Pusat peta: rata-rata titik Jatim
        center_lat = df["Lintang"].mean()
        center_lon = df["Bujur"].mean()
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB dark_matter"
        )

        # 1. Titik pusat gempa + popup
        hist_layer = folium.FeatureGroup(name="Gempa Historis")
        for i, r in df.iterrows():
            ang_info = "-" if np.isnan(r["Seg_Angle"]) else f"{r['Seg_Angle']:.2f}°"
            dist_info = (
                "-"
                if np.isnan(r["Seg_Distance_km"])
                else f"{r['Seg_Distance_km']:.2f} km"
            )

            popup_html = f"""
            <b>Lokasi:</b> {r.get('Lokasi','-')}<br>
            <b>Magnitudo:</b> {r['Magnitudo']}<br>
            <b>Lintang:</b> {r['Lintang']:.4f}<br>
            <b>Bujur:</b> {r['Bujur']:.4f}<br>
            <b>Angle (ke tetangga NN):</b> {ang_info}<br>
            <b>Distance (NN):</b> {dist_info}<br>
            """
            folium.CircleMarker(
                [r["Lintang"], r["Bujur"]],
                radius=4,
                color="orange",
                fill=True,
                fill_opacity=0.9,
                popup=popup_html,
            ).add_to(hist_layer)
        hist_layer.add_to(m)

        # 2. The Chaos (Connectivity) – hubungkan tiap titik ke k tetangga terdekat
        chaos_layer = folium.FeatureGroup(name="The Chaos (Connectivity)")
        k_neighbors = 3
        coords_lat = df["Lintang"].values
        coords_lon = df["Bujur"].values

        for i in range(len(df)):
            lat_i, lon_i = coords_lat[i], coords_lon[i]
            other_idx = np.array([j for j in range(len(df)) if j != i])
            dists = GeodesicKernel.haversine_vectorized(
                coords_lat[other_idx], coords_lon[other_idx], lat_i, lon_i
            )
            k = min(k_neighbors, len(other_idx))
            nn_idx = other_idx[np.argsort(dists)[:k]]
            for j in nn_idx:
                if j > i:  # supaya tidak double
                    folium.PolyLine(
                        [[lat_i, lon_i], [coords_lat[j], coords_lon[j]]],
                        color="#00ffff",
                        weight=1,
                        opacity=0.35,
                    ).add_to(chaos_layer)
        chaos_layer.add_to(m)

        # 3. The Snake (Best Path) – polyline mengikuti NN order
        snake_layer = folium.FeatureGroup(name="The Snake (Best Path)")
        snake_points = [[df.loc[idx, "Lintang"], df.loc[idx, "Bujur"]] for idx in order]
        folium.PolyLine(
            snake_points, color="#a020f0", weight=8, opacity=0.45  # ungu
        ).add_to(snake_layer)
        snake_layer.add_to(m)

        # 4. Prediksi Final – kalau ada history GA
        if history:
            last = history[-1]

            # raw end dari GA
            raw_end_lat = last["End_Lat"]
            raw_end_lon = last["End_Lon"]

            # snap end ke titik historis terdekat (Jatim)
            dists_end = GeodesicKernel.haversine_vectorized(
                coords_lat, coords_lon, raw_end_lat, raw_end_lon
            )
            snap_idx = int(np.argmin(dists_end))
            snap_end_lat = float(coords_lat[snap_idx])
            snap_end_lon = float(coords_lon[snap_idx])

            # start juga bisa di-snap dari genome GA (paling dekat)
            raw_start_lat = last["Start_Lat"]
            raw_start_lon = last["Start_Lon"]

            snap_start_lat = raw_start_lat
            snap_start_lon = raw_start_lon

            final_layer = folium.FeatureGroup(name="Prediksi Final")

            popup_final = folium.Popup(
                f"""
            <b>Prediksi Final GA (Snapped ke data historis)</b><br>
            <b>Start (snapped):</b> ({snap_start_lat:.4f}, {snap_start_lon:.4f})<br>
            <b>End (snapped):</b> ({snap_end_lat:.4f}, {snap_end_lon:.4f})<br>
            <b>Angle (genome):</b> {last['Angle']:.2f}°<br>
            <b>Distance (genome):</b> {last['Distance']:.2f} km<br>
            <b>Fitness:</b> {last['Fitness']:.4f}<br>
            <b>Generation:</b> {last['Generation']}<br>
            """,
                max_width=340,
            )

            folium.PolyLine(
                [[snap_start_lat, snap_start_lon], [snap_end_lat, snap_end_lon]],
                color="#00ffcc",
                weight=5,
                opacity=1.0,
                popup=popup_final,
                tooltip="Final Predicted Vector (Snapped)",
            ).add_to(final_layer)

            folium.CircleMarker(
                [snap_start_lat, snap_start_lon],
                radius=5,
                color="yellow",
                fill=True,
                fill_opacity=1,
                popup="Titik Awal (Snapped)",
            ).add_to(final_layer)

            folium.CircleMarker(
                [snap_end_lat, snap_end_lon],
                radius=6,
                color="red",
                fill=True,
                fill_opacity=1,
                popup="Titik Akhir (Snapped)",
            ).add_to(final_layer)

            final_layer.add_to(m)

        folium.LayerControl().add_to(m)

        try:
            m.get_root().html.add_child(
                folium.Element(
                    """
            <style>
            #ga-summary-panel {
                position: fixed;
                bottom: 30px;
                left: 30px;
                width: 260px;
                background: rgba(20,20,20,0.92);
                color: #00ffcc;
                padding: 12px;
                border-radius: 8px;
                font-family: monospace;
                font-size: 12px;
                z-index: 9999;
                box-shadow: 0 0 12px rgba(0,255,204,0.6);
            }
            #ga-summary-panel b {
                color: #ffffff;
            }
            </style>

            <div id="ga-summary-panel">
            <b>GA Final Summary</b><br>
            Loading...
            </div>

            <script>
            fetch("ga_summary.json")
            .then(res => res.json())
            .then(d => {
                document.getElementById("ga-summary-panel").innerHTML = `
                <b>GA Final Summary</b><br><br>
                Start : (${d.start_lat.toFixed(3)}, ${d.start_lon.toFixed(3)})<br>
                End&nbsp;&nbsp; : (${d.end_lat.toFixed(3)}, ${d.end_lon.toFixed(3)})<br>
                Angle: ${d.angle_deg.toFixed(2)}°<br>
                Dist&nbsp; : ${d.distance_km.toFixed(2)} km<br>
                Fitness: ${d.fitness.toFixed(4)}<br>
                Gen&nbsp;&nbsp; : ${d.generation}
                `;
            })
            .catch(err => {
                document.getElementById("ga-summary-panel").innerHTML =
                    "<b>GA Summary</b><br>Error loading JSON";
            });
            </script>
            """
                )
            )
            m.save(self.paths["ga_vector_map"])
            logger.info(f"Visualisasi GA disimpan: {self.paths['ga_vector_map']}")
        except Exception as e:
            logger.error(f"Gagal menyimpan peta GA: {e}")


# ==========================================
# 6. GA ENGINE ORCHESTRATOR
# ==========================================


class GAEngine:
    """Facade Class untuk GA Titanium Edition."""

    def __init__(self, config: dict):
        self.ga_cfg = config
        self.logger = logging.getLogger("GA_Orchestrator")
        self.logger.setLevel(logging.DEBUG)

        # Load Params
        self.n_islands = int(self.ga_cfg.get("n_islands", 4))
        self.migration_interval = int(self.ga_cfg.get("migration_interval", 10))
        self.migration_rate = float(self.ga_cfg.get("migration_rate", 0.1))
        self.mag_threshold = float(self.ga_cfg.get("magnitude_threshold", 5.0))
        self.generations = int(self.ga_cfg.get("generations", 200))
        self.pop_size = int(self.ga_cfg.get("population_size", 100))
        self.mutation_rate = float(self.ga_cfg.get("mutation_rate", 0.05))

        # Paths
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../output")
        )
        self.output_paths = {
            "ga_prediction_excel": os.path.join(
                base_path, "ga_results/ga_prediction_data_for_lstm.xlsx"
            ),
            "ga_best_chrom_csv": os.path.join(
                base_path, "ga_results/ga_best_chromosome.csv"
            ),
            "ga_vector_map": os.path.join(
                base_path, "ga_results/ga_vector_map.html"
            ),
            "ga_state": os.path.join(
                base_path, "ga_results/ga_state.pkl"
            ),
        }

        os.makedirs(
            os.path.dirname(self.output_paths["ga_prediction_excel"]),
            exist_ok=True
        )

        self.islands: List[PopulationIsland] = []
        self.history_log: List[dict] = []
        self.prev_best_fitness = None

    # ------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame):
        mask_jatim = (
            (df["Lintang"] >= EJ_MIN_LAT)
            & (df["Lintang"] <= EJ_MAX_LAT)
            & (df["Bujur"] >= EJ_MIN_LON)
            & (df["Bujur"] <= EJ_MAX_LON)
        )

        df_jatim = df[mask_jatim].copy()
        if df_jatim.empty:
            self.logger.warning(
                "Tidak ada data di dalam bounding Jawa Timur. Menggunakan seluruh data."
            )
            df_jatim = df.copy()

        sig_df = df_jatim.copy()
        if len(sig_df) < 5:
            self.logger.warning(
                "Data gempa signifikan sedikit, fallback ke semua data Jatim."
            )

        sig_df = sig_df.dropna(
            subset=["Magnitudo", "Lintang", "Bujur"]
        ).copy()

        if sig_df.empty:
            self.logger.warning(
                "[SAFE GA] Data setelah filter kosong, gunakan data asli minimal."
            )
            sig_df = df.copy()

        bounds = {
            "min_lat": max(EJ_MIN_LAT, sig_df["Lintang"].min() - 0.1),
            "max_lat": min(EJ_MAX_LAT, sig_df["Lintang"].max() + 0.1),
            "min_lon": max(EJ_MIN_LON, sig_df["Bujur"].min() - 0.1),
            "max_lon": min(EJ_MAX_LON, sig_df["Bujur"].max() + 0.1),
        }

        return sig_df, bounds

    # ------------------------------------------------------------------

    def _migrate(self):
        migrants_buffer: List[List[Individual]] = []

        for isl in self.islands:
            count = int(isl.pop_size * self.migration_rate)
            sorted_pop = sorted(
                isl.population,
                key=lambda x: x.fitness,
                reverse=True
            )
            migrants_buffer.append(
                [c.clone() for c in sorted_pop[:count]]
            )

        for i in range(self.n_islands):
            target = self.islands[(i + 1) % self.n_islands]
            incoming = migrants_buffer[i]

            target.population.sort(key=lambda x: x.fitness)

            for j, new_ind in enumerate(incoming):
                if j < len(target.population):
                    new_ind.is_evaluated = True
                    target.population[j] = new_ind

    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame):
        start_t = time.time()
        self.logger.info("=== GA TITAN ENGINE START ===")

        self.history_log = []
    # SAFE GUARD
        if len(df) < 3:
            self.logger.warning(
                "[SAFE GA] Data sangat sedikit, GA masuk mode fallback."
            )
            df["GA_Vector_Angle"] = 0.0
            df["GA_Vector_Distance_KM"] = 0.0
            df["GA_Confidence"] = 0.0
            return df, {"vector_data": None}

    # PREPARE DATA
        try:
            sig_df, bounds = self._prepare_data(df)
            evaluator = TectonicFitnessEvaluator(sig_df)
        except Exception as e:
            self.logger.error(f"GA Error: Data preparation failed - {e}")
            return df, {}

        # INIT ISLANDS
        self.islands = [
            PopulationIsland(i, self.ga_cfg, bounds, evaluator)
            for i in range(self.n_islands)
        ]

        for isl in self.islands:
            isl.initialize(sig_df)

        global_best_ind: Optional[Individual] = None

        # ================= MAIN LOOP =================
        for gen in range(self.generations):
            for isl in self.islands:
                isl.evaluate_generation()
                isl.reproduce_next_generation()

                # STAGNATION HANDLER (PER ISLAND)
                if isl.stagnation_count >= 30:
                    isl.trigger_catastrophe(sig_df)
                    isl.stagnation_count = 0

                # UPDATE GLOBAL BEST
                if isl.best_individual:
                    if global_best_ind is None:
                        global_best_ind = isl.best_individual.clone()
                        self.prev_best_fitness = global_best_ind.fitness
                    else:
                        if isl.best_individual.fitness > global_best_ind.fitness:
                            global_best_ind = isl.best_individual.clone()
                            self.prev_best_fitness = global_best_ind.fitness
                        else:
                            isl.stagnation_count += 1
            # MIGRATION
            if gen > 0 and gen % self.migration_interval == 0:
                self._migrate()

            # LOG BEST VECTOR
            if global_best_ind is not None:
                g = global_best_ind.genome
                e_lat, e_lon = GeodesicKernel.project_destination(
                    g.lat, g.lon, g.angle, g.dist
                )

                sig_reset = sig_df.reset_index()

                dists_end = GeodesicKernel.haversine_vectorized(
                    sig_reset["Lintang"].values,
                    sig_reset["Bujur"].values,
                    e_lat,
                    e_lon
                )

                snap_end_pos = int(np.argmin(dists_end))
                snap_end_orig_index = int(
                    sig_reset.loc[snap_end_pos, "index"]
                )

                dists_start = GeodesicKernel.haversine_vectorized(
                    sig_reset["Lintang"].values,
                    sig_reset["Bujur"].values,
                    g.lat,
                    g.lon
                )

                snap_start_pos = int(np.argmin(dists_start))
                snap_start_orig_index = int(
                    sig_reset.loc[snap_start_pos, "index"]
                )

                self.history_log.append({
                    "Generation": gen,
                    "Start_Lat": g.lat,
                    "Start_Lon": g.lon,
                    "End_Lat": e_lat,
                    "End_Lon": e_lon,
                    "Angle": g.angle,
                    "Distance": g.dist,
                    "Fitness": global_best_ind.fitness,
                    "Snap_Start_Idx": snap_start_orig_index,
                    "Snap_End_Idx": snap_end_orig_index,
                })

            # LOG PROGRESS
            if (gen + 1) % 20 == 0 and global_best_ind is not None:
                self.logger.info(
                    f"Gen {gen+1} | Global Best Fit: "
                    f"{global_best_ind.fitness:.5f}"
                )

        # ================= FINAL EXPORT =================
        exporter = GAExporter(self.output_paths)
        exporter.save_trace_history(self.history_log)
        exporter.generate_visual_map(self.history_log, sig_df)

        self.logger.info(
            f"GA Selesai. Waktu: {time.time() - start_t:.2f}s"
        )

        # SUMMARY
        if not self.history_log:
            ga_summary = {
                "start_lat": 0.0,
                "start_lon": 0.0,
                "end_lat": 0.0,
                "end_lon": 0.0,
                "angle_deg": 0.0,
                "distance_km": 0.0,
                "fitness": 0.0,
                "generation": 0,
            }
        else:
            last = self.history_log[-1]
            ga_summary = {
                "start_lat": float(last["Start_Lat"]),
                "start_lon": float(last["Start_Lon"]),
                "end_lat": float(last["End_Lat"]),
                "end_lon": float(last["End_Lon"]),
                "angle_deg": float(last["Angle"]),
                "distance_km": float(last["Distance"]),
                "fitness": float(last["Fitness"]),
                "generation": int(last["Generation"]),
            }

        return df, {
            "vector_data": (
                self.history_log[-1] if self.history_log else None
            ),
            "ga_summary": ga_summary,
        }
