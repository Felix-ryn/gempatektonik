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
    def build_model(self, input_shape=(32,32,5), hidden_nodes=[128,64]) -> tf.keras.Model:
        """
        ARSITEKTUR MODEL: SIMPLE CNN (BUKAN U-NET / BUKAN DEEP CNN)

        RINGKASAN:
        - Input           : 32×32×5
        - Convolution     : 2 blok (32 filter, 64 filter)
        - Hidden Layer    : 2 Dense layer (128 node, 64 node)
        - Output          : 2 head (4 node arah, 1 node sudut)
        """

        # =========================
        # INPUT LAYER
        # =========================
        # Input bukan layer trainable (tidak punya bobot & bias)
        inp = Input(shape=input_shape, name="cnn_input")
        x = inp

        # =====================================================
        # FEATURE EXTRACTION (CONVOLUTIONAL PART)
        # =====================================================

        # -----------------------------------------------------
        # BLOK CONV 1
        # -----------------------------------------------------
        # Conv2D:
        # - Jumlah filter     : 32
        # - Kernel            : 3×3
        # - Output            : 32 feature map
        # - Parameter         : (3×3×5)×32 + 32 bias
        x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)   # Layer trainable ke-1

        # BatchNormalization:
        # - Menstabilkan distribusi aktivasi
        # - Parameter         : 32 gamma + 32 beta
        x = BatchNormalization()(x)                                    # Layer trainable ke-2

        # MaxPooling:
        # - Downsampling 2×2
        # - Ukuran berubah   : 32×32 → 16×16
        # - Tidak punya parameter
        x = MaxPooling2D((2, 2))(x)                                    # Layer non-trainable

        # Total BLOK 1:
        # - 3 layer
        # - 32 filter (feature extractor)

        # -----------------------------------------------------
        # BLOK CONV 2
        # -----------------------------------------------------
        # Conv2D:
        # - Jumlah filter     : 64
        # - Kernel            : 3×3
        # - Input channel     : 32
        # - Parameter         : (3×3×32)×64 + 64 bias
        x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)   # Layer trainable ke-3

        # BatchNormalization:
        # - Parameter         : 64 gamma + 64 beta
        x = BatchNormalization()(x)                                    # Layer trainable ke-4

        # MaxPooling:
        # - Downsampling 2×2
        # - Ukuran berubah   : 16×16 → 8×8
        x = MaxPooling2D((2, 2))(x)                                    # Layer non-trainable

        # Total BLOK 2:
        # - 3 layer
        # - 64 filter

        # =====================================================
        # FLATTENING / FEATURE COMPRESSION
        # =====================================================

        # GlobalAveragePooling:
        # - Mengambil rata-rata tiap feature map
        # - Mengubah 8×8×64 → 64 node
        # - Tidak punya parameter
        x_flat = GlobalAveragePooling2D()(x)

        # =====================================================
        # HIDDEN LAYERS (FULLY CONNECTED)
        # =====================================================

        # Hidden Layer 1:
        # - Dense 128 node
        # - Parameter         : 64×128 + 128 bias
        # Hidden Layer 2:
        # - Dense 64 node
        # - Parameter         : 128×64 + 64 bias
        for nodes in hidden_nodes:
            x_flat = Dense(nodes, activation="relu")(x_flat)          # Layer trainable
            x_flat = Dropout(0.3)(x_flat)                              # Regularization (tidak trainable)

        # Total Hidden Layer:
        # - 2 hidden layer
        # - Node: 128 → 64

        # =====================================================
        # OUTPUT LAYERS (MULTI-HEAD)
        # =====================================================

        # OUTPUT HEAD 1 — ARAH
        # - Dense 4 node
        # - Softmax
        # - Mewakili 4 kelas: Timur, Barat, Selatan, Utara
        # - Parameter         : 64×4 + 4 bias
        dir_out = Dense(4, activation="softmax", name="dir_output")(x_flat)

        # OUTPUT HEAD 2 — SUDUT
        # - Dense 1 node
        # - Linear (regresi)
        # - Parameter         : 64×1 + 1 bias
        angle_out = Dense(1, activation="linear", name="angle_output")(x_flat)

        # =====================================================
        # BUILD & COMPILE MODEL
        # =====================================================
        model = Model(
            inputs=inp,
            outputs=[dir_out, angle_out],
            name="SimpleCNN_Adaptive_v4"
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
    def train_and_predict(self, df_main: pd.DataFrame, train_indices=None, test_indices=None, **kwargs):
        """
        Main Pipeline: Load Data -> Build Tensors (5 Channels) -> Train -> Predict (Arah, Sudut, Risk Array)
        Sesuai Spesifikasi Client: Simple CNN Adaptif.
        [FIXED]: Auto-detect input shape mismatch untuk mencegah crash loading model lama.
        """
        self.logger.info("CNN DEBUG: [Step 1] Memulai CNN Execution.")
        
        # 0. Check TensorFlow
        if not HAS_TF:
            self.logger.error("CNN Disabled (TensorFlow missing). Returning fallback values.")
            df_main["CNN_Risk_Array"] = np.array([0.0])
            return df_main
            
        # 1. Load & Prepare Data (Inject History H-1)
        df_proc = self._inject_history_features(df_main)
        self.logger.info(f"CNN DEBUG: [Step 2] Data diproses, {len(df_proc)} baris. Fitur H-1 injected.")

        # --------------------------------------------------------
        # 2. TRAINING PHASE (Tensor Building)
        # --------------------------------------------------------
        train_samples = df_proc[
            (df_proc.get("Context_Impact_Radius", 0) > 0) | 
            (df_proc.get("R_true", 0) > 0)
        ]
        
        self.logger.info(f"CNN DEBUG: [Step 3] Membangun tensor training ({len(train_samples)} valid samples).")
        
        X_train = None
        y_train_dict = None

        if len(train_samples) > 0:
            X_train = np.array([self.tensor_builder.construct_input_tensor(r) for _, r in train_samples.iterrows()])
            gt_list = [self.tensor_builder.construct_ground_truth(r) for _, r in train_samples.iterrows()]
            
            y_train_dict = {
                "dir_output": np.array([g["dir_output"] for g in gt_list]),     
                "angle_output": np.array([g["angle_output"] for g in gt_list])  
            }

        # --------------------------------------------------------
        # 3. MODEL MANAGEMENT (Robust Load vs Rebuild)
        # --------------------------------------------------------
        model_path = self.paths.get("model_file", "")
        model_exists = os.path.exists(model_path)
        self.model = None # Reset model state
        
        if model_exists:
            try:
                self.logger.info(f"CNN DEBUG: Mencoba memuat model dari {model_path}...")
                temp_model = load_model(model_path, compile=False)
                
                # --- [SAFETY CHECK 1]: Validasi Struktur Output (2 Heads) ---
                valid_output = isinstance(temp_model.output, list) and len(temp_model.output) == 2
                
                # --- [SAFETY CHECK 2]: Validasi Input Shape (Grid Size Match) ---
                # Input shape biasanya (None, H, W, C). Kita cek H dan W.
                input_shape = temp_model.input_shape
                # Handle jika input_shape berupa list (multiple inputs) atau tuple
                if isinstance(input_shape, list): 
                    input_shape = input_shape[0]
                
                # Cek dimensi (biasanya index 1 dan 2 adalah H dan W)
                # input_shape[1] harus sama dengan self.grid_size
                current_shape_match = (input_shape[1] == self.grid_size) and (input_shape[2] == self.grid_size)

                if valid_output and current_shape_match:
                    self.model = temp_model
                    self.model.compile(
                        optimizer=Adam(0.001),
                        loss={"dir_output": "categorical_crossentropy", "angle_output": "mse"},
                        metrics={"dir_output": "accuracy", "angle_output": "mae"}
                    )
                    self.logger.info("CNN DEBUG: Model valid & kompatibel (Grid Size cocok).")
                else:
                    if not valid_output:
                        self.logger.warning("CNN DEBUG: Model lama struktur outputnya salah (Bukan 2 Head).")
                    if not current_shape_match:
                        self.logger.warning(f"CNN DEBUG: Model lama beda ukuran grid! (Model: {input_shape[1]} vs Config: {self.grid_size}).")
                    
                    self.logger.warning("CNN DEBUG: Membuang model lama & memaksa Rebuild.")
                    self.model = None # Force Rebuild

            except Exception as e:
                self.logger.warning(f"CNN DEBUG: Gagal load model (Corrupt/Error): {e}. Force rebuild.")
                self.model = None

        # Jika model None (karena belum ada ATAU tidak valid), bangun baru
        if self.model is None:
            self.logger.info(f"CNN DEBUG: Membangun ulang Simple CNN baru (Grid: {self.grid_size})...")
            self.model = self.architect.build_model(
                input_shape=(self.grid_size, self.grid_size, 5),
                hidden_nodes=[128, 64]
            )

        # --------------------------------------------------------
        # 4. TRAINING LOOP
        # --------------------------------------------------------
        if X_train is not None and len(X_train) >= 2:
            self.logger.info(f"CNN DEBUG: [Step 4] Training Start ({self.epochs} epochs)...")
            
            callbacks_list = [
                ModelCheckpoint(self.paths["model_file"], save_best_only=True, verbose=0),
                CSVLogger(self.paths.get("training_log", "training_log.csv"), append=True) 
            ]
            
            try:
                self.model.fit(
                    X_train, y_train_dict,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=0,
                    callbacks=callbacks_list
                )
                self.logger.info("CNN DEBUG: Training selesai.")
            except Exception as e:
                self.logger.error(f"CNN Training Error: {e}")

        # --------------------------------------------------------
        # 5. PREDICTION PHASE
        # --------------------------------------------------------
        try:
            self.logger.info("CNN DEBUG: [Step 5] Prediksi Data Terkini.")

            X_full = np.array([self.tensor_builder.construct_input_tensor(r) for _, r in df_proc.iterrows()])

            # Safety Check Dimensi Tensor sebelum masuk model
            if X_full.ndim != 4:
                 self.logger.warning("CNN DEBUG: Dimensi tensor salah.")
            
            # Predict
            preds = self.model.predict(X_full, verbose=0)
            pred_dir_probs, pred_angle_norm = preds

            # --- AMBIL DATA EVENT TERAKHIR (LATEST) ---
            last_idx = -1
            
            # 1. Output Arah
            latest_dir_probs = pred_dir_probs[last_idx]
            arah_idx = np.argmax(latest_dir_probs)
            dir_map = {0: "Timur", 1: "Barat", 2: "Selatan", 3: "Utara"} 
            arah_str = dir_map.get(arah_idx, "Unknown")
            
            # 2. Output Sudut
            az = float(pred_angle_norm.flatten()[last_idx]) * 360.0
            
            # 3. Output Risiko (k) sebagai ARRAY
            raw_conf = float(np.max(latest_dir_probs))
            conf_val = min(raw_conf, 0.95)
            risk_array = np.array([conf_val])

            # Simpan Log
            output_df = pd.DataFrame([{
                "timestamp": pd.Timestamp.now(),
                "arah_prediksi": arah_str,
                "arah_derajat": az,
                "risk_k_array": str(risk_array.tolist()), 
                "confidence": conf_val,
                "sumber": "SimpleCNN_V2"
            }])
            
            out_path = self.paths.get("cnn_prediction_out")
            if out_path:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                output_df.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)

            self.logger.info(f"CNN RESULT: Arah={arah_str}, Sudut={az:.2f}, RiskArray={risk_array}")

            # Inject ke DataFrame
            if "CNN_Risk_Array" not in df_main.columns:
                df_main["CNN_Risk_Array"] = None
                df_main["CNN_Risk_Array"] = df_main["CNN_Risk_Array"].astype(object)

            df_main.at[df_main.index[-1], "CNN_Risk_Array"] = risk_array
            df_main.at[df_main.index[-1], "CNN_Pred_Arah"] = arah_str
            df_main.at[df_main.index[-1], "CNN_Pred_Sudut"] = az

        except Exception as e:
            self.logger.critical(f"CNN CRASH PRED: Error saat prediksi: {e}")
            df_main.at[df_main.index[-1], "CNN_Risk_Array"] = np.array([0.0])

        return df_main