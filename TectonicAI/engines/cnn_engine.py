"""
CNN ENGINE – TITANIUM SAFE EDITION (v3.3)
Anti-crash, Anti-broadcast error, Auto-repair model, Self-contained.
"""

import numpy as np
import pandas as pd
import logging
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from math import radians, sin, cos, atan2, degrees

# [FIX] Tambahkan Type Hinting yang diperlukan
from typing import Optional, Dict, List, Any, Tuple

# --- DEEP LEARNING STACK ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D,
        Conv2DTranspose, concatenate,
        BatchNormalization, Activation, Add, Dropout, GlobalAveragePooling2D, Dense, Softmax
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    # Optional: set seed & GPU memory growth
    tf.keras.utils.set_random_seed(42)
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    HAS_TF = True
except Exception:
    HAS_TF = False
    logging.warning("TensorFlow not found or incompatible. CNN Engine disabled.")


# ============================================================
# SYNC PANJANG ARRAY (FIX broadcast error) & METRICS
# ============================================================

def sync_len(*arrays):
    """Memotong array agar memiliki panjang yang sama (min_len)."""
    if not arrays:
        return arrays
    eff = [a for a in arrays if a is not None]
    if not eff:
        return arrays
    min_len = min(len(a) for a in eff)
    synced = []
    for a in arrays:
        if a is None:
            synced.append(None)
        else:
            synced.append(a[:min_len])
    return tuple(synced)

def calculate_angular_diff(pred_angle, true_angle):
    """Menghitung selisih sudut terkecil antara dua arah (0-360)."""
    diff = abs(pred_angle - true_angle)
    return min(diff, 360.0 - diff)

def dir_from_angle(angle_deg: float) -> str:
    angle = angle_deg % 360.0
    # Menggunakan pembagian 45 derajat per arah (8 arah mata angin)
    if angle >= 337.5 or angle < 22.5:
        return "Utara"
    elif 22.5 <= angle < 67.5:
        return "Timur Laut"
    elif 67.5 <= angle < 112.5:
        return "Timur"
    elif 112.5 <= angle < 157.5:
        return "Tenggara"
    elif 157.5 <= angle < 202.5:
        return "Selatan"
    elif 202.5 <= angle < 247.5:
        return "Barat Daya"
    elif 247.5 <= angle < 292.5:
        return "Barat"
    else:
        return "Barat Laut"


def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Menghitung bearing (azimuth) dari titik 1 ke titik 2 (0–360°)
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    return (degrees(atan2(x, y)) + 360) % 360

def project_point(lat, lon, bearing_deg, distance_km):
    """
    Proyeksi titik dari (lat, lon) sejauh distance_km ke arah bearing_deg.
    Menggunakan great-circle navigation (geodesi standar).
    """
    R = 6371.0  # radius bumi (km)
    bearing = radians(bearing_deg)

    lat1 = radians(lat)
    lon1 = radians(lon)

    lat2 = asin(
        sin(lat1) * cos(distance_km / R) +
        cos(lat1) * sin(distance_km / R) * cos(bearing)
    )

    lon2 = lon1 + atan2(
        sin(bearing) * sin(distance_km / R) * cos(lat1),
        cos(distance_km / R) - sin(lat1) * sin(lat2)
    )

    return degrees(lat2), degrees(lon2)


def extract_direction_and_angle(mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Ekstrak arah (azimuth), sudut sebaran, dan confidence dari output CNN mask.
    """
    if mask is None or mask.size == 0:
        return 0.0, 0.0, 0.0

    m = mask.squeeze()
    h, w = m.shape

    total = np.sum(m)
    if total <= 1e-6:
        return 0.0, 0.0, 0.0

    y, x = np.mgrid[0:h, 0:w]

    cx = np.sum(x * m) / total
    cy = np.sum(y * m) / total

    dx = cx - (w / 2)
    dy = (h / 2) - cy  # koordinat kartesian

    # Azimuth (0–360 derajat)
    azimuth = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

    # Sudut deviasi (sebaran area terdampak)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    spread = np.sum(dist * m) / total

    # Confidence sederhana
    confidence = float(np.clip(np.mean(m), 0, 1))

    return float(azimuth), float(spread), confidence


# ============================================================
#  METRICS (Dice & IoU tetap ada utk kompatibilitas load model)
# ============================================================

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    union = (
        tf.reduce_sum(tf.abs(y_true), axis=[1, 2, 3]) +
        tf.reduce_sum(tf.abs(y_pred), axis=[1, 2, 3]) - intersection
    )
    return tf.reduce_mean((intersection + smooth) / (union + smooth))

# ============================================================
#  SAFEST INPUT BUILDER
# ============================================================

class TensorConstructor:
    def __init__(self, grid_size: int, logger):
        self.grid_size = grid_size
        self.logger = logger

    def construct_input_tensor(self, row: pd.Series) -> np.ndarray:
        """
        Membangun Input 5 Channel Sesuai Request Client:
        1. ACO Center (H)
        2. ACO Area Impact (H)
        3. ACO Center (H-1 / Previous)
        4. ACO Area Impact (H-1 / Previous)
        5. LSTM Feature
        """
        gs = self.grid_size
        
        # --- DATA SAAT INI (H) ---
        cx = float(row.get("ACO_center_x", 0.5))
        cy = float(row.get("ACO_center_y", 0.5))
        rad = float(row.get("Context_Impact_Radius", 0.0))

        # --- DATA MASA LALU (H-1) ---
        cx_prev = float(row.get("ACO_center_x_prev", cx)) 
        cy_prev = float(row.get("ACO_center_y_prev", cy))
        rad_prev = float(row.get("Radius_prev", rad))

        # NORMALISASI LAT/LON → 0–1
        cx = (cx + 90.0) / 180.0
        cy = (cy + 180.0) / 360.0

        cx_prev = (cx_prev + 90.0) / 180.0
        cy_prev = (cy_prev + 180.0) / 360.0

        # --- LSTM OUTPUT ---
        lstm_val = float(row.get("LSTM_pred", 0.0))

        # Helper Grid
        xv, yv = np.meshgrid(np.linspace(0,1,gs), np.linspace(0,1,gs))
        y_idx, x_idx = np.ogrid[:gs, :gs]
        km_per_unit = 100.0 / gs

        # ---------------------------------------------------
        # Channel 1: Pusat Gempa H (Gaussian Heatmap)
        # ---------------------------------------------------
        sigma = 0.05 # Standar deviasi kecil untuk titik pusat
        c1_center_h = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2*sigma*sigma))

        # ---------------------------------------------------
        # Channel 2: Area Terdampak H (Binary Mask)
        # ---------------------------------------------------
        pix_r = np.clip(rad / km_per_unit, 0, gs)
        cx_p, cy_p = int(cx * (gs-1)), int(cy * (gs-1))
        dist = np.sqrt((x_idx - cx_p)**2 + (y_idx - cy_p)**2)
        c2_area_h = (dist <= pix_r).astype(np.float32)

        # ---------------------------------------------------
        # Channel 3: Pusat Gempa H-1 (Gaussian Heatmap)
        # ---------------------------------------------------
        c3_center_prev = np.exp(-((xv - cx_prev)**2 + (yv - cy_prev)**2) / (2*sigma*sigma))

        # ---------------------------------------------------
        # Channel 4: Area Terdampak H-1 (Binary Mask)
        # ---------------------------------------------------
        pix_r_prev = np.clip(rad_prev / km_per_unit, 0, gs)
        cx_p_prev, cy_p_prev = int(cx_prev * (gs-1)), int(cy_prev * (gs-1))
        dist_prev = np.sqrt((x_idx - cx_p_prev)**2 + (y_idx - cy_p_prev)**2)
        c4_area_prev = (dist_prev <= pix_r_prev).astype(np.float32)

        # ---------------------------------------------------
        # Channel 5: LSTM Feature (Scalar Broadcast)
        # ---------------------------------------------------
        c5_lstm = np.full((gs, gs), np.tanh(lstm_val), dtype=np.float32) # tanh agar range -1 s/d 1

        # Stack menjadi (Grid, Grid, 5)
        stacked = np.stack([c1_center_h, c2_area_h, c3_center_prev, c4_area_prev, c5_lstm], axis=-1)
        return stacked

    def construct_ground_truth(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """
        [NEW METHOD] Membuat Target Label untuk Training.
        Mengubah data kolom menjadi format yang bisa dilatih oleh CNN.
        """
        # 1. Ambil Sudut Aktual (Handling kolom nama yang mungkin beda)
        sudut_deg = float(row.get("Arah_Derajat", row.get("angle", 0.0))) % 360.0

        # 2. Tentukan Kelas Arah (4 Sumbu: Timur, Barat, Selatan, Utara)
        # Mapping: 0=Timur, 1=Barat, 2=Selatan, 3=Utara
        if (sudut_deg >= 315 or sudut_deg < 45):
            arah_idx = 3 # Utara
        elif (sudut_deg >= 45 and sudut_deg < 135):
            arah_idx = 0 # Timur
        elif (sudut_deg >= 135 and sudut_deg < 225):
            arah_idx = 2 # Selatan
        else:
            arah_idx = 1 # Barat

        # One-Hot Encoding untuk 4 Sumbu
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[int(arah_idx)] = 1.0

        # Normalisasi Sudut (0-1) untuk Regresi
        angle_norm = np.array([sudut_deg / 360.0], dtype=np.float32)

        return {
            "dir_output": dir_onehot,
            "angle_output": angle_norm
        }



# ============================================================
#  RES-UNET ARCHITECTURE (SAFE)
# ============================================================

class CNNModelArchitect:
    def build_model(
        self,
        input_shape=(32, 32, 5), # menerima 5 input dan grid 32x32
        hidden_nodes=[32, 16] # hidden node yang digunakan ada 2 (32 dan 16)
    ) -> tf.keras.Model:
        """
        SIMPLE CNN — 1 CONV BLOCK (CLIENT VERSION)

        - Input   : 32×32×5
        - Conv    : 1 blok (32 filter)
        - Hidden  : Dense 32 → 16
        - Output  : 2 head (arah + sudut)
        Total layer 11 , Total Hidden layer 2
        Rumus hitung bobot Conv2D: (kernel_height × kernel_width × input_channel + bias) × jumlah_filter
        """

        # =========================
        # INPUT
        # =========================
        # menerima input 5 channel dan grid 32x32
        inp = Input(shape=input_shape, name="cnn_input") # 1 layer (dihitung sebagai layer arsitektur)

        # =========================
        # SINGLE CONV BLOCK
        # =========================
        # membaca pola (arah sebaran, pergeseran pusat, dampak area)
        # setiap filter = satu detektor pola
        # 1 layer
        x = Conv2D(
            filters=32, # jumlah filter
            kernel_size=(3, 3), # ukuran kernel
            padding="same",
            activation="relu", # menggunakan RelU karena cepat, tidak saturasi seperti sigmoid, dan cocok untuk CNN
            name="conv_block_1"
        )(inp)
        # Perhitungan bobot untuk 1 Filter:
        #  3 × 3 × 5 = 45 bobot
        # Karena tidak memberikan paramater (use_bias=False) maka tiap filter punya 1 bias
        # 45 bobot + 1 bias = 46 paramater
        # Lalu dikalikan dengan jumlah filter
        # 46 x 32 = 1.472 parameter
        # karena tiap filter punya 1 bias, maka blok ini ada 32 bias.

        # Menormalkan output Conv2D dan membantu training lebih stabil
        # Ini adalah layer yang punya 2 paramater yaitu:
        # Gamma (γ) mengatur skala fitur
        # Beta (β) menggeser nilai fitur
        # per channel
        x = BatchNormalization(name="bn_1")(x) # 1 layer
        # jumlah channel 32 dari var x
        # oleh karena itu: 32 x 2 = 64

        # tidak ada bobot dan bias 
        # termasuk functional layer (non paramatic layer)
        x = MaxPooling2D( # 1 layer
            pool_size=(2, 2),
            name="pool_1"
        )(x)
        # jadi bentuk setelah pooling: (None, 16, 16, 32)

        # =========================
        # FEATURE COMPRESSION
        # =========================
        x = GlobalAveragePooling2D(name="gap")(x) # 1 layer
        # Output: (None, 32)
        """
        GlobalAveragePooling2D menghasilkan 32 node karena ia merangkum setiap feature map menjadi satu nilai rata-rata.
        Karena sebelumnya ada 32 channel hasil Conv2D, maka output GAP otomatis menjadi vektor sepanjang 32.    
        """
        # Meratakan setiap feature map dan menghindari flatten besar
        # setiap channelnya 1 niali, jadi 16x16x32 = 32 node
        # tidak ada bobot

        # =========================
        # HIDDEN LAYERS
        # =========================
        # disini melakukan looping sebanyak hidden layer: 2 kali
        for i, nodes in enumerate(hidden_nodes): # 2 layer karena loop 2 x
            x = Dense(
                nodes,
                activation="relu",
                name=f"dense_hidden_{i+1}"
            )(x)
                    # Dense Hidden 1 (32 node)
                    # input dari GAP: 32 Node
                    # Perhitungan Bobot dan Bias
                    # (32 + 1) × 32 = 1.056 parameter 
                    # Total Bias: 32

                    # Dense Hidden 2 (16 node)
                    # perhitungan 
                    # (32 + 1) × 16 = 528 parameter
                    # Total Bias: 16

            # 1 layer 
            x = Dropout(0.3, name=f"dropout_{i+1}")(x) #Menonaktifkan 30% neuron saat training

        # =========================
        # OUTPUT HEADS
        # =========================
        # dense: 2 layer (karena menghasilkan 2 output)
        # Output direction head
        # Total : 4 bias / 1 fitur 1 bias / 1 bias per neuron
        dir_out = Dense( 
            4,
            activation="softmax", # karena output berupa probabilitas
            name="dir_output"
        )(x)
        # (16 + 1) × 4 = 68 parameter

        # Total : 1 bias 
        angle_out = Dense(
            1,
            activation="sigmoid",
            name="angle_output"
        )(x)
        # (16 + 1) × 1 = 17 parameter

        # =========================
        # MODEL BUILD
        # =========================
        model = Model(
            inputs=inp,
            outputs=[dir_out, angle_out],
            name="SimpleCNN_OneBlock_v1"
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                "dir_output": "categorical_crossentropy",
                "angle_output": "mse"
            },
            metrics={
                "dir_output": "accuracy",
                "angle_output": "mae"
            }
        )

        return model



# ============================================================
#  VISUALIZATION PIPELINE
# ============================================================

class VisualizationPipeline:
    def __init__(self, paths, logger):
        self.paths = paths
        self.logger = logger

    def visualize_last(self, prediction, row):
        try:
            if prediction is None or prediction.size == 0:
                return
            
            # Jika paths belum ready (empty dict), abort visualization
            if not self.paths or "realtime_viz" not in self.paths:
                return

            plt.figure(figsize=(6, 6))
            sns.heatmap(
                prediction[:, :, 0],
                cmap="hot",
                vmin=0, vmax=1,
                square=True
            )
            lokasi = row.get('Lokasi', 'Unknown')
            mag = row.get('Magnitudo', '?')
            plt.title(f"{lokasi} | M {mag}")
            plt.axis("off")

            viz_path = self.paths.get("realtime_viz", "")
            if not viz_path:
                return

            viz_dir = os.path.dirname(viz_path) or "."
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception as e:
            self.logger.warning(f"Heatmap render error: {e}")

# ============================================================
#  MAIN CNN ENGINE (FINAL SAFE EDITION)
# ============================================================

class CNNEngine:
    
    # Init sebagai dict kosong (Safety First)
    paths: Dict[str, str] = {} 

    def __init__(self, config: dict = None):
        self.cnn_cfg = config if config is not None else {}
        self.logger = logging.getLogger("CNN_Engine")
        self.logger.setLevel(logging.INFO)
        self.grid_size = 32
        self.epochs = 25
        self.batch_size = 16
        
        # Inisialisasi komponen
        self.tensor_builder = TensorConstructor(self.grid_size, self.logger)
        self.architect = CNNModelArchitect()
        self.model = None 

        # --- BLOK KRITIS: INISIALISASI PATHS ---
        # inside __init__
        try:
            # fallback jika __file__ tidak tersedia (notebook / REPL)
            if '__file__' in globals():
                current_dir = os.path.dirname(os.path.abspath(__file__))
            else:
                current_dir = os.getcwd()

            project_root = os.path.abspath(os.path.join(current_dir, "../.."))
            base_output = os.path.join(project_root, "output")

            # default paths
            self.paths = {
                "lstm_bridge_in": os.path.join(base_output, "lstm_results", "lstm_data_for_cnn.xlsx"),
                "model_file": os.path.join(base_output, "cnn_results", "ResUNetXL.keras"),
                "training_log": os.path.join(base_output, "cnn_results", "training_log.csv"),
                "realtime_viz": os.path.join(base_output, "cnn_results", "latest_heatmap.png"),
                "cnn_prediction_out": os.path.join(base_output, "cnn_results", "cnn_next_earthquake_prediction.csv"),
            }

            # allow overriding via config (handy ketika run di PC/laptop kamu)
            if isinstance(self.cnn_cfg.get("paths"), dict):
                for k, v in self.cnn_cfg["paths"].items():
                    if v:
                        self.paths[k] = v

            os.makedirs(os.path.dirname(self.paths["model_file"]), exist_ok=True)
            self.logger.info("CNN DEBUG: [Init] Path output berhasil dikonfigurasi.")
        except Exception as e:
            self.paths = {}
            self.logger.critical(f"CNN CRASH INIT: Gagal membuat/mengakses path output: {e}. CNN running in 'Disabled Mode'.")

        # Inisialisasi komponen pendukung
        self.tensor_builder = TensorConstructor(self.grid_size, self.logger)
        self.architect = CNNModelArchitect()
        self.viz = VisualizationPipeline(self.paths, self.logger) 
        self.model = None


    def _report_value(self, val, label="DATA_TIDAK_ADA"):
        """
        Helper untuk kebutuhan laporan:
        - Jika None / NaN → tampilkan label teks
        - Jika ada nilai → tampilkan nilai asli
        """
        if val is None:
            return label
        try:
            if isinstance(val, float) and np.isnan(val):
                return label
        except Exception:
            pass
        return val


    # --------------------------------------------------------
    def _load_lstm_bridge(self):
        if not self.paths:
            return pd.DataFrame()

        path_in = self.paths.get("lstm_bridge_in", "")
        if not path_in or not os.path.exists(path_in):
            self.logger.warning(f"CNN BRIDGE: File {path_in} not found. Skipping.")
            return pd.DataFrame()

        try:
            if path_in.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(path_in).fillna(0.0)
            elif path_in.lower().endswith('.csv'):
                df = pd.read_csv(path_in).fillna(0.0)
            else:
                # try excel first, fallback to csv
                try:
                    df = pd.read_excel(path_in).fillna(0.0)
                except Exception:
                    df = pd.read_csv(path_in).fillna(0.0)
            return df
        except Exception as e:
            self.logger.error(f"Gagal membaca LSTM bridge file {path_in}: {e}")
            return pd.DataFrame()

    # --------------------------------------------------------
    def _inject_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper internal untuk membuat fitur H-1 (Lagging) secara otomatis.
        FIX:
        - Support angka dengan koma (Indonesia / Excel)
        - Mencegah input CNN = nol semua
        - Aman untuk data historis & 2025
        """
        if df.empty:
            return df

        df = df.copy()

        # ===============================
        # UTIL: SAFE FLOAT (KOMA → TITIK)
        # ===============================
        def _safe_float(val, default=0.0):
            try:
                if isinstance(val, str):
                    val = val.replace(",", ".")
                return float(val)
            except Exception:
                return default

        # ===============================
        # 1. NORMALISASI & SORT TIMESTAMP
        # ===============================
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
        elif 'Tanggal' in df.columns:  # Support Bahasa Indonesia
            df['timestamp'] = pd.to_datetime(df['Tanggal'], errors='coerce')

        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # ===============================
        # 2. DETEKSI KOLOM SUMBER
        # ===============================
        cols = df.columns.tolist()

        # Latitude / X
        col_x = None
        for candidate in [
            'ACO_center_x', 'center_x',
            'latitude', 'lat', 'x',
            'Lintang'
        ]:
            if candidate in cols:
                col_x = candidate
                break

        # Longitude / Y
        col_y = None
        for candidate in [
            'ACO_center_y', 'center_y',
            'longitude', 'long', 'lon', 'y',
            'Bujur'
        ]:
            if candidate in cols:
                col_y = candidate
                break

        # Radius
        col_r = None
        for candidate in [
            'Context_Impact_Radius',
            'radius_km', 'radius',
            'R_true', 'impact_radius'
        ]:
            if candidate in cols:
                col_r = candidate
                break

        # ===============================
        # 3. STANDARISASI & LAGGING (H-1)
        # ===============================
        if col_x and col_y and col_r:
            self.logger.info(
                f"CNN DEBUG: Mapping kolom -> X:{col_x}, Y:{col_y}, R:{col_r}"
            )

            # 🔧 FIX UTAMA: konversi aman ke float
            df['ACO_center_x'] = df[col_x].apply(_safe_float)
            df['ACO_center_y'] = df[col_y].apply(_safe_float)
            df['Context_Impact_Radius'] = df[col_r].apply(_safe_float)

            # H-1 (lagging)
            df['ACO_center_x_prev'] = (
                df['ACO_center_x'].shift(1).fillna(df['ACO_center_x'])
            )
            df['ACO_center_y_prev'] = (
                df['ACO_center_y'].shift(1).fillna(df['ACO_center_y'])
            )
            df['Radius_prev'] = (
                df['Context_Impact_Radius']
                .shift(1)
                .fillna(df['Context_Impact_Radius'])
            )
            # ===============================
            # 4. HITUNG ARAH AKTUAL (DERIVED)
            # ===============================
            try:
                df["Arah_Derajat"] = df.apply(
                    lambda r: bearing_deg(
                        r["ACO_center_x_prev"],
                        r["ACO_center_y_prev"],
                        r["ACO_center_x"],
                        r["ACO_center_y"]
                    )
                    if not pd.isna(r["ACO_center_x_prev"]) and not pd.isna(r["ACO_center_y_prev"])
                    else np.nan,
                    axis=1
                )
            except Exception as e:
                self.logger.warning(f"Gagal menghitung Arah_Derajat: {e}")
                df["Arah_Derajat"] = np.nan

        else:
            # Fallback aman (tidak crash)
            self.logger.warning(
                f"KOLOM HILANG: lat/lon/radius tidak ditemukan di {cols}. "
                f"Menggunakan default aman."
            )
            df['ACO_center_x'] = 0.5
            df['ACO_center_y'] = 0.5
            df['Context_Impact_Radius'] = 0.0

            df['ACO_center_x_prev'] = 0.5
            df['ACO_center_y_prev'] = 0.5
            df['Radius_prev'] = 0.0

        return df


    def _haversine_km(self, lon1, lat1, lon2, lat2):
        """Return distance in kilometers between two (lon,lat)."""
        # handle missing
        try:
            lon1, lat1, lon2, lat2 = map(float, (lon1, lat1, lon2, lat2))
        except Exception:
            return None
        R = 6371.0  # Earth radius km
        dlon = radians(lon2 - lon1)
        dlat = radians(lat2 - lat1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c

    def _angle_diff(self, a, b):
        """Smallest angular difference between a and b in degrees."""
        # ensure floats
        a = float(a) % 360.0
        b = float(b) % 360.0
        diff = abs((a - b + 180) % 360 - 180)
        return diff

    def _accuracy_from_angle(self, diff_angle, threshold=180.0):
        """
        Soft accuracy:
        - 0°   = 100%
        - 180° = 0%
        """
        try:
            diff_angle = abs(float(diff_angle))
        except:
            return 0.0

        diff_angle = min(diff_angle, 180.0)
        acc = 1.0 - (diff_angle / 180.0)
        return round(acc * 100.0, 2)

    def validate_predictions_2025(
        self,
        df_main: pd.DataFrame = None,
        pred_csv: str = None,
        time_window_days: int = 30,
        angle_threshold: float = 60.0,
        distance_km_threshold: float = None,
        out_csv: str = None
    ) -> pd.DataFrame:
        """
        Validasi prediksi yang ada di CSV terhadap kejadian aktual tahun 2025.
        - df_main: dataframe sumber event (harus mengandung kolom timestamp dan, bila tersedia, lat/lon)
        - pred_csv: path ke CSV prediksi (fallback ke self.paths["cnn_prediction_out"])
        - time_window_days: jangka waktu setelah basis_data_terakhir untuk mencari match
        - angle_threshold: ambang selisih sudut (degrees) untuk dikatakan match
        - distance_km_threshold: (optional) ambang jarak (km) antara pusat prediksi & actual
        - out_csv: path untuk menyimpan hasil validasi (default ke cnn_results/validation_2025.csv)
        Returns dataframe hasil validasi.
        """
        pred_csv = pred_csv or self.paths.get("cnn_prediction_out")
        if pred_csv is None or not os.path.exists(pred_csv):
            self.logger.error(f"Validation aborted: pred CSV not found at {pred_csv}")
            return pd.DataFrame()

        pred_df = pd.read_csv(pred_csv, encoding='utf-8-sig').fillna("")
        # Ensure datetime parsing
        if 'basis_data_terakhir' in pred_df.columns:
            pred_df['basis_data_terakhir'] = pd.to_datetime(pred_df['basis_data_terakhir'], errors='coerce')
        else:
            pred_df['basis_data_terakhir'] = pd.to_datetime(pred_df.get('timestamp', pd.NaT), errors='coerce')

        pred_df['pred_angle'] = pd.to_numeric(pred_df.get('arah_derajat', pred_df.get('angle', 0.0)), errors='coerce')
        pred_df['prediction_id'] = pred_df.get('prediction_id', [str(uuid.uuid4()) for _ in range(len(pred_df))])

        # Load main events if not provided
        if df_main is None or df_main.empty:
            df_main = self._load_lstm_bridge()
        if df_main is None or df_main.empty:
            self.logger.error("Validation aborted: no main event dataframe available.")
            return pd.DataFrame()

        # Normalize timestamp column
        if 'timestamp' not in df_main.columns:
            # try known alternatives
            for alt in ['time', 'Tanggal']:
                if alt in df_main.columns:
                    df_main['timestamp'] = pd.to_datetime(df_main[alt], errors='coerce')
                    break
        else:
            df_main['timestamp'] = pd.to_datetime(df_main['timestamp'], errors='coerce')

        # Filter only 2025 events (atau > cutoff)
        df_proc = self._inject_history_features(df_main)
        cutoff = pd.Timestamp("2024-12-31 23:59:59")
        test_df = df_proc[df_proc['timestamp'] > cutoff].copy()
        if test_df.empty:
            self.logger.info("No 2025 events found in provided data (test_df empty).")
        # prepare lat/lon columns if available
        lat_col = None
        lon_col = None
        for c in ['ACO_center_x','center_x','latitude','lat','x','Lintang']:
            if c in df_proc.columns:
                lat_col = c
                break
        for c in ['ACO_center_y','center_y','longitude','long','lon','y','Bujur']:
            if c in df_proc.columns:
                lon_col = c
                break

        results = []
        for _, prow in pred_df.iterrows():
            pid = prow.get('prediction_id', str(uuid.uuid4()))
            base_date = prow.get('basis_data_terakhir', pd.NaT)
            pred_angle = float(prow.get('pred_angle', 0.0) if not pd.isna(prow.get('pred_angle')) else 0.0)

            # Candidate selection: events after basis_date up to window, fallback whole 2025
            if not pd.isna(base_date):
                start = base_date
                end = base_date + pd.Timedelta(days=int(time_window_days))
                candidates = test_df[(test_df['timestamp'] >= start) & (test_df['timestamp'] <= end)].copy()
            else:
                candidates = test_df.copy()

            best = None
            best_diff = 999.0
            best_dist = None

            # If no candidates in window, expand to whole 2025
            if candidates.empty:
                candidates = test_df.copy()

            for _, act in candidates.iterrows():
                actual_angle = float(act.get('Arah_Derajat', act.get('angle', 0.0)) or 0.0)
                diff = self._angle_diff(pred_angle, actual_angle)
                dist_km = None
                if lat_col and lon_col:
                    # get lon/lat of prediction (use basis row values if stored in pred CSV: check 'basis_lat' etc.)
                    # Try to extract lat/lon from act row for distance
                    try:
                        # If pred CSV stored center coords, use them; otherwise we only compute dist between pred basis coords and actual coords is not possible.
                        pred_lat = prow.get('ACO_center_x', prow.get('center_x', None))
                        pred_lon = prow.get('ACO_center_y', prow.get('center_y', None))
                        act_lat = act.get(lat_col)
                        act_lon = act.get(lon_col)
                        if pred_lat not in [None, "", "nan"] and pred_lon not in [None, "", "nan"]:
                            dist_km = self._haversine_km(float(pred_lon), float(pred_lat), float(act_lon), float(act_lat))
                        else:
                            # If pred lat/lon not available, set dist None or compute 0
                            dist_km = None
                    except Exception:
                        dist_km = None

                if diff < best_diff:
                    best_diff = diff
                    best = act
                    best_dist = dist_km

            match_flag = False
            if best is not None and best_diff <= float(angle_threshold):
                if distance_km_threshold is None:
                    match_flag = True
                else:
                    if best_dist is not None and best_dist <= float(distance_km_threshold):
                        match_flag = True
                    else:
                        match_flag = False

            result_row = {
                "prediction_id": pid,
                "pred_basis_date": base_date,
                "pred_angle": pred_angle,
                "best_match_timestamp": best.get('timestamp') if best is not None else pd.NaT,
                "best_match_angle": float(best.get('Arah_Derajat', best.get('angle', float('nan')))) if best is not None else None,
                "angle_diff_deg": float(best_diff) if best is not None else None,
                "best_match_distance_km": float(best_dist) if best_dist is not None else None,
                "match_flag": bool(match_flag),
                "search_window_days": int(time_window_days),
                "angle_threshold_deg": float(angle_threshold),
                "distance_threshold_km": float(distance_km_threshold) if distance_km_threshold is not None else None
            }
            results.append(result_row)

        valid_df = pd.DataFrame(results)

        # save
        if out_csv is None:
            out_csv = os.path.join(os.path.dirname(self.paths.get("cnn_prediction_out",".")), "validation_2025.csv")
        try:
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            valid_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
            self.logger.info(f"Validation saved to {out_csv}")
        except Exception as e:
            self.logger.error(f"Failed to save validation CSV: {e}")

        return valid_df

    # --------------------------------------------------------
    #  (VALIDASI & SPLIT TAHUN)
    # --------------------------------------------------------
    def train_and_predict(self, df_main: pd.DataFrame, train_indices=None, test_indices=None, **kwargs):
        """
        Main Pipeline: Smart Split, Validasi, Smart Backfill, dan FORMAT CSV YANG SESUAI DASHBOARD LAMA.
        """
        self.logger.info("CNN DEBUG: [Step 1] Memulai CNN Execution (Smart Split Mode).")

        # 0. Check TensorFlow
        if not HAS_TF:
            self.logger.error("CNN Disabled (TensorFlow missing).")
            df_main["CNN_Risk_Array"] = np.array([0.0])
            return df_main

        # 1. Load & Prepare Data
        df_proc = self._inject_history_features(df_main)
        
        # Pastikan kolom timestamp ada
        if 'timestamp' not in df_proc.columns:
            df_proc['timestamp'] = pd.date_range(start='2022-01-01', periods=len(df_proc), freq='D')

        # --------------------------------------------------------
        # 2. SMART SPLIT LOGIC
        # --------------------------------------------------------
        max_date = df_proc['timestamp'].max()
        has_2025 = max_date.year >= 2025

        if has_2025:
            cutoff_date = pd.Timestamp("2024-12-31 23:59:59")
            self.logger.info("CNN SCENARIO: Data 2025 Terdeteksi. Mode Validasi: Train 2022-2024 -> Test 2025.")
        else:
            cutoff_date = pd.Timestamp("2023-12-31 23:59:59")
            self.logger.info("CNN SCENARIO: Data 2025 Kosong. Mode Fallback: Train 2022-2023 -> Test 2024.")

        train_df = df_proc[df_proc['timestamp'] <= cutoff_date].copy()
        test_df = df_proc[df_proc['timestamp'] > cutoff_date].copy()

        if len(train_df) < 5:
            self.logger.warning("CNN WARN: Data training terlalu sedikit (<5). Pakai semua data.")
            train_df = df_proc.copy()
            test_df = pd.DataFrame()

        # --------------------------------------------------------
        # 3. TRAINING PHASE
        # --------------------------------------------------------
        train_samples = train_df[
            (train_df.get("Context_Impact_Radius", 0) > 0) | 
            (train_df.get("R_true", 0) > 0)
        ]

        X_train = None
        y_train_dict = None

        if len(train_samples) > 0:
            X_train = np.array([self.tensor_builder.construct_input_tensor(r) for _, r in train_samples.iterrows()])
            gt_list = [self.tensor_builder.construct_ground_truth(r) for _, r in train_samples.iterrows()]
            
            y_train_dict = {
                "dir_output": np.array([g["dir_output"] for g in gt_list]),     
                "angle_output": np.array([g["angle_output"] for g in gt_list])  
            }

        if self.model is None:
             self.model = self.architect.build_model(input_shape=(self.grid_size, self.grid_size, 5))

        if X_train is not None and len(X_train) >= 2:
            callbacks_list = [
                ModelCheckpoint(self.paths["model_file"], monitor='loss', save_best_only=True, verbose=0),
                CSVLogger(self.paths.get("training_log", "training_log.csv"), append=True) 
            ]
            self.model.fit(X_train, y_train_dict, batch_size=self.batch_size, epochs=self.epochs, verbose=0, callbacks=callbacks_list)

        # --------------------------------------------------------
        # 4. PREDICTION LOGIC (WITH SMART BACKFILL)
        # --------------------------------------------------------
        if len(train_df) == 0:
            return df_main

        # Cek apakah CSV sudah ada isinya?
        out_path = self.paths.get("cnn_prediction_out")
        csv_exists_and_filled = False
        if out_path and os.path.exists(out_path):
            try:
                with open(out_path, 'r') as f:
                    if len(f.readlines()) > 1:
                        csv_exists_and_filled = True
            except:
                pass
        
        # =====================================================
        # LOGIC PREDICTION TARGET (FULL FILL MODE - SAFE)
        # =====================================================

        # Sync index (WAJIB)
        df_proc = df_proc.reset_index(drop=True)
        df_main = df_main.reset_index(drop=True)

        # Pastikan kolom CNN ada
        for col in ["CNN_Pred_Sudut", "CNN_Pred_Arah"]:
            if col not in df_main.columns:
                df_main[col] = np.nan

        # Mask baris yang belum diprediksi
        mask_unpredicted = df_main["CNN_Pred_Sudut"].isna()

        rows_to_predict = df_proc.loc[mask_unpredicted].iterrows()

        self.logger.info(
            f"CNN MODE: Full CSV Fill aktif. "
            f"Total baris diprediksi = {mask_unpredicted.sum()}"
        )


        # Variabel penampung hasil akhir untuk injection
        final_risk_array = np.array([0.0])
        final_pred_arah = "Unknown"
        final_pred_sudut = 0.0
        final_validation_note = "No Data"

        for idx_row, row_data in rows_to_predict:
            # =====================================================
            # 1. BUILD INPUT
            # =====================================================
            X_pred = np.array([self.tensor_builder.construct_input_tensor(row_data)])

            # =====================================================
            # 2. DEBUG INPUT (ANTI SILENT ERROR)
            # =====================================================
            import hashlib
            h = hashlib.md5(X_pred.tobytes()).hexdigest()
            self.logger.info(
                f"CNN DEBUG: Predict row idx={idx_row} "
                f"basis={row_data.get('timestamp')} "
                f"input_hash={h} mean={X_pred.mean():.6f}"
            )

            # =====================================================
            # 3. MODEL PREDICTION
            # =====================================================
            preds = self.model.predict(X_pred, verbose=0)
            pred_dir_probs, pred_angle_norm = preds

            arah_idx = np.argmax(pred_dir_probs[0])
            dir_map = {0: "Timur", 1: "Barat", 2: "Selatan", 3: "Utara"} 
            pred_arah = dir_map.get(arah_idx, "Unknown")

            pred_sudut = (float(pred_angle_norm[0][0]) * 360.0) % 360.0
            confidence = float(np.max(pred_dir_probs[0]))
            risk_array = np.array([confidence])

            # =====================================================
            # 4. CONSISTENCY CHECK (CLASS vs ANGLE)
            # =====================================================
            dir_by_angle = dir_from_angle(pred_sudut)
            consistency_flag = (dir_by_angle == pred_arah)

            # =====================================================
            # 5. RULE-BASED RECONCILIATION (SAFE OVERRIDE)
            # =====================================================
            CONF_THRESHOLD = 0.6

            if not consistency_flag and confidence < CONF_THRESHOLD:
                self.logger.info(
                    f"CNN DEBUG: Inconsistent class/angle, "
                    f"low confidence ({confidence:.3f}) "
                    f"-> override class with angle-derived {dir_by_angle}"
                )
                pred_arah = dir_by_angle
                confidence *= 0.8
                risk_array = np.array([confidence])

            # Optional: penalti confidence jika tidak konsisten
            if not consistency_flag:
                confidence *= 0.7  # turunkan 30%
                risk_array = np.array([confidence])


            # --- VALIDASI ---
            validation_note = "Menunggu Data Masa Depan"
            diff_angle = -1.0
            status_validasi = "PENDING"
            PROJ_DISTANCE_KM = 150.0   # asumsi konservatif (WAJIB dijelaskan di laporan)
            SPATIAL_RADIUS_KM = 50.0   # radius wilayah target

            # =====================================================
            # ROBUST VALIDATION (SPATIAL PROJECTION – NO GT ANGLE)
            # =====================================================

            threshold_angle = 60.0
            spatial_radius_km = 50.0
            window_days = 30

            found = False
            best_diff = None
            best_row = None
            closest_candidates = []

            # ---------------------------------
            # 1. WINDOW-BASED SAMPLING
            # ---------------------------------
            base_time = row_data.get("timestamp")

            if "timestamp" in test_df.columns and base_time is not None:
                candidates_df = test_df[
                    (test_df["timestamp"] >= base_time) &
                    (test_df["timestamp"] <= base_time + pd.Timedelta(days=window_days))
                ]
            else:
                candidates_df = test_df.copy()

            if candidates_df.empty:
                candidates_df = test_df.copy()

            # ---------------------------------
            # 2. SAFETY GUARD: KOORDINAT WAJIB
            # ---------------------------------
            required_cols = {"ACO_center_x", "ACO_center_y"}

            if not required_cols.issubset(candidates_df.columns):
                self.logger.warning("VALIDATION SKIPPED: Koordinat event 2025 tidak lengkap.")
                status_validasi = "KOORDINAT_TIDAK_LENGKAP"
                diff_angle = 180.0
                validation_note = "Validasi dilewati karena data koordinat tidak tersedia."

            else:
                # ---------------------------------
                # 3. PROYEKSI TITIK DARI HASIL CNN
                # ---------------------------------
                try:
                    proj_lat, proj_lon = project_point(
                        row_data["ACO_center_x"],
                        row_data["ACO_center_y"],
                        pred_sudut,
                        spatial_radius_km
                    )
                except Exception:
                    proj_lat, proj_lon = None, None

                # ---------------------------------
                # 4. ITERASI EVENT 2025
                # ---------------------------------
                for _, actual_event in candidates_df.iterrows():
                    try:
                        actual_sudut = bearing_deg(
                            row_data["ACO_center_x"],
                            row_data["ACO_center_y"],
                            actual_event["ACO_center_x"],
                            actual_event["ACO_center_y"]
                        )

                        dist_km = self._haversine_km(
                            proj_lat, proj_lon,
                            actual_event["ACO_center_x"],
                            actual_event["ACO_center_y"]
                        )
                    except Exception:
                        continue

                    diff = self._angle_diff(pred_sudut, actual_sudut)

                    closest_candidates.append({
                        "timestamp": actual_event.get("timestamp"),
                        "angle": actual_sudut,
                        "diff": diff,
                        "distance_km": dist_km
                    })

                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_row = actual_event

                    if diff <= threshold_angle and dist_km is not None and dist_km <= spatial_radius_km:
                        found = True
                        break

                # ---------------------------------
                # 5. POST-PROCESSING
                # ---------------------------------
                if best_diff is None:
                    best_diff = 180.0

                diff_angle = best_diff

                closest_candidates = sorted(
                    closest_candidates,
                    key=lambda x: x["diff"]
                )[:2]

                if found:
                    status_validasi = "VALID"
                    validation_note = "Event 2025 ditemukan sesuai arah dan proyeksi spasial."
                else:
                    status_validasi = "MENYIMPANG"
                    validation_note = (
                        f"Tidak ada event dalam threshold. "
                        f"Selisih terdekat {best_diff:.1f}°."
                    )


            # =====================================================
            # PROYEKSI TITIK DARI HASIL CNN (INTI PERMINTAAN CLIENT)
            # =====================================================
            try:
                proj_lat, proj_lon = project_point(
                    row_data["ACO_center_x"],   # lat basis (2024)
                    row_data["ACO_center_y"],   # lon basis
                    pred_sudut,                 # arah hasil CNN
                    PROJ_DISTANCE_KM
                )
            except Exception:
                proj_lat, proj_lon = None, None


            accuracy_percent = self._accuracy_from_angle(
                diff_angle,
                threshold=threshold_angle
            )

            # --- PENYIMPANAN DATA (FIX KEYERROR: KEMBALIKAN NAMA KOLOM LAMA) ---
            if out_path:
                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    new_data = {
                        "timestamp": pd.Timestamp.now(),
                        "basis_data_terakhir": row_data.get("timestamp"),

                        "arah_prediksi": pred_arah,
                        "arah_derajat": pred_sudut,
                        "dir_inferred_from_angle": dir_by_angle,

                        "confidence_scalar": confidence,
                        "akurasi_prediksi_persen": accuracy_percent,

                        "status_validasi": status_validasi,
                        "validasi_note": validation_note,
                        "selisih_sudut": diff_angle,

                        "ACO_Center_Lat": row_data.get("ACO_center_x"),
                        "ACO_Center_Lon": row_data.get("ACO_center_y"),
                        "ACO_Impact_Radius_km": row_data.get("Context_Impact_Radius"),

                        "proj_distance_km": PROJ_DISTANCE_KM,
                        "proj_target_lat": proj_lat,
                        "proj_target_lon": proj_lon,

                        "alt_sampling_1_angle": self._report_value(
                            closest_candidates[0]["angle"] if len(closest_candidates) > 0 else None
                        ),
                        "alt_sampling_1_diff": self._report_value(
                            closest_candidates[0]["diff"] if len(closest_candidates) > 0 else None
                        ),
                        "alt_sampling_2_angle": self._report_value(
                            closest_candidates[1]["angle"] if len(closest_candidates) > 1 else None
                        ),
                        "alt_sampling_2_diff": self._report_value(
                            closest_candidates[1]["diff"] if len(closest_candidates) > 1 else None
                        ),

                        "sumber": "SimpleCNN_SmartBackfill",
                        "consistency_flag": consistency_flag
                    }


                    # =====================================================
                    # 🔗 ACO → LSTM BRIDGE (WAJIB, FIX KOLOM MISSING)
                    # =====================================================
                    new_data.update({
                        # pusat ACO (HARUS SAMA NAMA DENGAN LSTM)
                        "ACO_Center_Lat": row_data.get("ACO_center_x"),
                        "ACO_Center_Lon": row_data.get("ACO_center_y"),

                        # radius dampak (opsional tapi penting)
                        "ACO_Impact_Radius_km": row_data.get("Context_Impact_Radius"),
                    })


                    # ===============================
                    # TAMBAHAN: SAMPLING ALTERNATIF
                    # ===============================
                    new_data.update({
                        "alt_sampling_1_angle": self._report_value(
                            closest_candidates[0]["angle"]
                            if isinstance(closest_candidates, list) and len(closest_candidates) > 0
                            else None
                        ),
                        "alt_sampling_1_diff": self._report_value(
                            closest_candidates[0]["diff"]
                            if isinstance(closest_candidates, list) and len(closest_candidates) > 0
                            else None
                        ),
                        "alt_sampling_2_angle": self._report_value(
                            closest_candidates[1]["angle"]
                            if isinstance(closest_candidates, list) and len(closest_candidates) > 1
                            else None
                        ),
                        "alt_sampling_2_diff": self._report_value(
                            closest_candidates[1]["diff"]
                            if isinstance(closest_candidates, list) and len(closest_candidates) > 1
                            else None
                        ), 
                    })

                    output_df = pd.DataFrame([new_data])

                    file_exists = os.path.exists(out_path)
                    output_df.to_csv(
                        out_path,
                        mode='w',        
                        header=True,       
                        index=False,
                        encoding='utf-8-sig'
                    )


                except PermissionError:
                    self.logger.error(
                        f"GAGAL SIMPAN CSV: File {out_path} sedang dibuka! Tutup file Excelnya."
                    )
                except Exception as e:
                    self.logger.error(f"Error saving CSV: {e}")

            # Update variabel akhir
            final_pred_arah = pred_arah
            final_pred_sudut = pred_sudut
            final_validation_note = validation_note
            final_risk_array = risk_array

            # =====================================================
            # ✅ FIX UTAMA: INJECTION PER BARIS (BUKAN TERAKHIR SAJA)
            # =====================================================

            # Pastikan kolom ada
            for col in [
                "CNN_Risk_Array",
                "CNN_Pred_Arah",
                "CNN_Pred_Sudut",
                "CNN_Validasi_Msg"
            ]:
                if col not in df_main.columns:
                    df_main[col] = None


            # Inject langsung ke baris yang sedang diprediksi
            df_main.at[idx_row, "CNN_Risk_Array"] = risk_array
            df_main.at[idx_row, "CNN_Pred_Arah"] = pred_arah
            df_main.at[idx_row, "CNN_Pred_Sudut"] = pred_sudut
            df_main.at[idx_row, "CNN_Validasi_Msg"] = validation_note


        self.logger.info("CNN PREDICTION: Selesai memproses batch prediksi (Backfill/Realtime).")

        return df_main