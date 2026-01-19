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
            activation="linear", # karena regresi nilai kontinu dan tidak membatasi range
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
        """Helper internal untuk membuat fitur H-1 (Lagging) secara otomatis"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # 1. Pastikan urut waktu
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif 'time' in df.columns: 
            df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif 'Tanggal' in df.columns: # Support kolom 'Tanggal' (Indonesian)
            df['timestamp'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
        # 2. DETEKSI KOLOM SUMBER (UPDATE: SUPPORT BAHASA INDONESIA)
        cols = df.columns.tolist()
        
        # Cari kolom Latitude/X (Menambahkan 'Lintang')
        col_x = None
        # Urutan prioritas: Nama standar -> Inggris -> Indonesia -> Singkatan
        for candidate in ['ACO_center_x', 'center_x', 'latitude', 'lat', 'x', 'Lintang']:
            if candidate in cols:
                col_x = candidate
                break
                
        # Cari kolom Longitude/Y (Menambahkan 'Bujur')
        col_y = None
        for candidate in ['ACO_center_y', 'center_y', 'longitude', 'long', 'lon', 'y', 'Bujur']:
            if candidate in cols:
                col_y = candidate
                break
                
        # Cari kolom Radius
        col_r = None
        for candidate in ['Context_Impact_Radius', 'radius_km', 'radius', 'R_true', 'impact_radius']:
            if candidate in cols:
                col_r = candidate
                break
        
        # 3. Buat kolom H-1 (Prev) jika kolom sumber ditemukan
        if col_x and col_y and col_r:
            self.logger.info(f"CNN DEBUG: Mapping kolom ditemukan -> X: {col_x}, Y: {col_y}, R: {col_r}")
            
            # Kita STANDARISASI nama kolom ke internal engine agar konsisten
            df['ACO_center_x'] = df[col_x]
            df['ACO_center_y'] = df[col_y]
            df['Context_Impact_Radius'] = df[col_r]
            
            # Shift data untuk H-1
            df['ACO_center_x_prev'] = df['ACO_center_x'].shift(1).fillna(df['ACO_center_x'])
            df['ACO_center_y_prev'] = df['ACO_center_y'].shift(1).fillna(df['ACO_center_y'])
            df['Radius_prev']       = df['Context_Impact_Radius'].shift(1).fillna(df['Context_Impact_Radius'])
        else:
            # Jika masuk sini, artinya nama kolom masih belum match
            self.logger.warning(f"KOLOM HILANG: Tidak bisa menemukan lat/lon/radius di {cols}. Menggunakan default.")
            df['ACO_center_x_prev'] = 0.5
            df['ACO_center_y_prev'] = 0.5
            df['Radius_prev'] = 0.0

        return df

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
        
        # LOGIC BACKFILL: Jika CSV kosong, ambil 10 data terakhir. Jika tidak, ambil 1 data terakhir.
        if not csv_exists_and_filled:
            self.logger.info("CNN INFO: Log CSV kosong. Mengaktifkan 'Smart Backfill' (10 data historis)...")
            rows_to_predict = train_df.tail(10).iterrows() 
        else:
            rows_to_predict = train_df.tail(1).iterrows()

        # Variabel penampung hasil akhir untuk injection
        final_risk_array = np.array([0.0])
        final_pred_arah = "Unknown"
        final_pred_sudut = 0.0
        final_validation_note = "No Data"

        for idx_row, row_data in rows_to_predict:
            # Predict
            X_pred = np.array([self.tensor_builder.construct_input_tensor(row_data)])
            preds = self.model.predict(X_pred, verbose=0)
            pred_dir_probs, pred_angle_norm = preds

            arah_idx = np.argmax(pred_dir_probs[0])
            dir_map = {0: "Timur", 1: "Barat", 2: "Selatan", 3: "Utara"} 
            pred_arah = dir_map.get(arah_idx, "Unknown")
            pred_sudut = float(pred_angle_norm[0][0]) * 360.0
            confidence = float(np.max(pred_dir_probs[0]))
            risk_array = np.array([confidence])

            # --- VALIDASI ---
            validation_note = "Menunggu Data Masa Depan"
            diff_angle = -1.0
            status_validasi = "PENDING"
            
            if not test_df.empty:
                actual_event = test_df.iloc[0]
                actual_sudut = float(actual_event.get("Arah_Derajat", actual_event.get("angle", 0.0)))
                diff_angle = abs(pred_sudut - actual_sudut)
                diff_angle = min(diff_angle, 360 - diff_angle)
                status_validasi = "RELEVAN" if diff_angle <= 60 else "MENYIMPANG"
                validation_note = (
                    f"Prediksi {pred_arah} ({pred_sudut:.0f}°), "
                    f"Aktual ({actual_sudut:.0f}°). Selisih {diff_angle:.1f}°. Status: {status_validasi}"
                )

            # --- PENYIMPANAN DATA (FIX KEYERROR: KEMBALIKAN NAMA KOLOM LAMA) ---
            if out_path:
                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    # [PENTING] Menggunakan nama kolom yang SAMA PERSIS dengan Dashboard lama
                    new_data = {
                        "timestamp": pd.Timestamp.now(),  # <--- INI KUNCI PERBAIKANNYA (Bukan timestamp_prediksi)
                        "arah_prediksi": pred_arah,       # Dashboard pakai ini
                        "arah_derajat": pred_sudut,       # Dashboard pakai ini
                        "risk_k_array": str(risk_array.tolist()), # Dashboard pakai ini
                        "confidence_scalar": confidence,  # Dashboard pakai ini
                        "sumber": "SimpleCNN_SmartBackfill",
                        
                        # Metadata tambahan untuk validasi (Dashboard tidak akan error karena ini kolom ekstra)
                        "basis_data_terakhir": row_data.get('timestamp', pd.Timestamp.now()),
                        "validasi_note": validation_note,
                        "selisih_sudut": diff_angle,
                        "status_validasi": status_validasi
                    }
                    output_df = pd.DataFrame([new_data])
                    
                    file_exists = os.path.exists(out_path)
                    # Mode append, header hanya jika file belum ada
                    output_df.to_csv(out_path, mode='a', header=not file_exists, index=False)
                    
                except PermissionError:
                    self.logger.error(f"GAGAL SIMPAN CSV: File {out_path} sedang dibuka! Tutup file Excelnya.")
                except Exception as e:
                    self.logger.error(f"Error saving CSV: {e}")

            # Update variabel akhir
            final_pred_arah = pred_arah
            final_pred_sudut = pred_sudut
            final_validation_note = validation_note
            final_risk_array = risk_array

        self.logger.info("CNN PREDICTION: Selesai memproses batch prediksi (Backfill/Realtime).")

        # --------------------------------------------------------
        # 6. INJECTION KE DATAFRAME UTAMA
        # --------------------------------------------------------
        target_time_col = None
        for col_name in ['timestamp', 'time', 'Tanggal']:
            if col_name in df_main.columns:
                target_time_col = pd.to_datetime(df_main[col_name], errors='coerce')
                break
        
        idx = df_main.index[-1]
        last_ts = train_df.iloc[-1].get('timestamp', None)

        if target_time_col is not None and last_ts is not None:
            matches = df_main[target_time_col == last_ts].index
            if not matches.empty:
                idx = matches[0]

        if "CNN_Risk_Array" not in df_main.columns:
            df_main["CNN_Risk_Array"] = None
            df_main["CNN_Risk_Array"] = df_main["CNN_Risk_Array"].astype(object)

        df_main.at[idx, "CNN_Risk_Array"] = final_risk_array
        df_main.at[idx, "CNN_Pred_Arah"] = final_pred_arah
        df_main.at[idx, "CNN_Pred_Sudut"] = final_pred_sudut
        df_main.at[idx, "CNN_Validasi_Msg"] = final_validation_note

        return df_main