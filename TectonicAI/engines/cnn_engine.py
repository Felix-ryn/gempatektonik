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

    def _safe_map(self, value, scale=1.0):
        try:
            v = float(value)
            if np.isnan(v) or np.isinf(v) or abs(v) > 1e10: 
                return 0.0
            return v * scale
        except:
            return 0.0

    def construct_input_tensor(self, row: pd.Series) -> np.ndarray:
        gs = self.grid_size

        # --- Ambil nilai ACO
        cx_rel = float(row.get("ACO_center_x", 0.5))
        cy_rel = float(row.get("ACO_center_y", 0.5))
        impact_radius_km = float(row.get("Context_Impact_Radius", row.get("R_true", 0.0)))

        # LSTM output scalar
        lstm_feat = float(row.get("LSTM_pred", 0.0))

        # Magnitude & log depth
        mag = float(row.get("Magnitudo", 0.0))
        kedalaman_safe = float(row.get("Kedalaman_km", 0.0))
        log_depth = np.log1p(abs(kedalaman_safe)) if kedalaman_safe is not None else 0.0

        # Channel 1: ACO center map
        xv, yv = np.meshgrid(np.linspace(0,1,gs), np.linspace(0,1,gs))
        sigma = max(1e-3, (impact_radius_km / 100.0) * 0.5 + 0.01)
        center_map = np.exp(-((xv - cx_rel)**2 + (yv - cy_rel)**2) / (2*sigma*sigma)).astype(np.float32)

        # Channel 2: ACO area mask
        km_per_unit = 100.0 / gs
        pixel_radius = np.clip(impact_radius_km / km_per_unit, 0, gs)
        cx_pixel = int(cx_rel * (gs-1))
        cy_pixel = int(cy_rel * (gs-1))
        y_idx, x_idx = np.ogrid[:gs, :gs]
        dist = np.sqrt((x_idx - cx_pixel)**2 + (y_idx - cy_pixel)**2)
        area_map = (dist <= pixel_radius).astype(np.float32)

        # Channel 3: LSTM feature
        c_lstm = np.full((gs, gs), np.clip(lstm_feat, -1e3, 1e3), dtype=np.float32)

        # Channel 4: Magnitude normalized
        c_mag = np.full((gs, gs), np.clip(mag / 10.0, 0.0, 10.0), dtype=np.float32)

        # Channel 5: log depth broadcast
        c_depth = np.full((gs, gs), log_depth, dtype=np.float32)

        # --- Stack 5 channel saja
        stacked = np.stack([center_map, area_map, c_lstm, c_mag, c_depth], axis=-1)

        # Safety dimension fix
        if stacked.shape != (gs, gs, 5):
            fixed = np.zeros((gs, gs, 5), dtype=np.float32)
            h = min(gs, stacked.shape[0])
            w = min(gs, stacked.shape[1])
            d = min(5, stacked.shape[2])
            fixed[:h, :w, :d] = stacked[:h, :w, :d]
            stacked = fixed

        return stacked.astype(np.float32)


    def construct_ground_truth(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """
        Output ground truth untuk SimpleCNN_SimpleHead:
        - dir_output: one-hot 4 class (timur, barat, selatan, utara)
        - angle_output: scalar sudut 0-1
        """
        # Ambil arah dari data (jika ada)
        arah_label = row.get("Arah_Class", 0)  # 0=Timur,1=Barat,2=Selatan,3=Utara
        if arah_label not in [0,1,2,3]:
            arah_label = 0
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[int(arah_label)] = 1.0

        # Ambil sudut derajat dari data, ubah ke 0-1
        sudut_deg = float(row.get("Arah_Derajat", 0.0)) % 360.0
        angle_norm = np.array([sudut_deg / 360.0], dtype=np.float32)

        return {
            "dir_output": dir_onehot,
            "angle_output": angle_norm
        }



# ============================================================
#  RES-UNET ARCHITECTURE (SAFE)
# ============================================================

class CNNModelArchitect:
    def _conv_block(self, x, filters):
        """Satu blok Conv2D -> ReLU -> BatchNorm -> MaxPool"""
        x = Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        return x

    def build_model(self, input_shape=(32,32,5), hidden_nodes=[128,64], use_mask=False) -> tf.keras.Model:
        inp = Input(shape=input_shape, name="cnn_input")
        x = inp
        
        # Satu blok Conv sederhana
        x = Conv2D(32, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(64, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        # Flatten untuk dense layers
        x_flat = GlobalAveragePooling2D()(x)

        for nodes in hidden_nodes:
            x_flat = Dense(nodes, activation="relu")(x_flat)
            x_flat = Dropout(0.2)(x_flat)

        # Output 1: arah (4 kelas)
        dir_out = Dense(4, activation="softmax", name="dir_output")(x_flat)
        # Output 2: sudut (regression)
        angle_out = Dense(1, activation="linear", name="angle_output")(x_flat)

        model = Model(inputs=inp, outputs=[dir_out, angle_out], name="SimpleCNN_SimpleHead")
        model.compile(
            optimizer=Adam(0.001),
            loss=["categorical_crossentropy", "mse"],
            metrics=["accuracy"]
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

        self.grid_size = int(self.cnn_cfg.get("grid_size", 32))
        self.epochs = int(self.cnn_cfg.get("epochs", 25))
        self.batch_size = int(self.cnn_cfg.get("batch_size", 16))
        self.input_channels = 5 # Standard input (Mag, Depth, Pheromone, Temporal, Cluster)

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
    def train_and_predict(self, df_main: pd.DataFrame, train_indices=None, test_indices=None, **kwargs):
        """Main Pipeline: Load Data -> Build Tensors -> Load/Train Model -> Predict"""

        self.logger.info("CNN DEBUG: [Step 1] Memulai CNN Execution.")
        
        # 0. Check TensorFlow
        if not HAS_TF:
            self.logger.error("CNN Disabled (TensorFlow missing). Returning fallback values.")
            df_main["Proporsi_Grid_Terdampak_CNN"] = 0.0
            return df_main
            
        # 1. Load Bridge Data
        df_bridge = self._load_lstm_bridge()
        if df_bridge.empty or not self.paths: 
            self.logger.warning("CNN DEBUG: Bridge data kosong. Menggunakan df_main (Mode Mandiri/Fallback).")
            # Fallback jika bridge kosong, mungkin data pertama
            if len(df_main) > 0:
                df_bridge = df_main.copy()
            else:
                df_main["Proporsi_Grid_Terdampak_CNN"] = 0.0
                return df_main
            
        self.logger.info(f"CNN DEBUG: [Step 2] Data dimuat, {len(df_bridge)} baris.")

        # --------------------------------------------------------
        # 2. TRAINING PHASE (Tensor Building)
        # --------------------------------------------------------
        # Filter hanya data yang punya radius > 0 (artinya pernah gempa signifikan) untuk Training
        train_samples = df_bridge[
            (df_bridge.get("Context_Impact_Radius", 0) > 0) | 
            (df_bridge.get("R_true", 0) > 0)
        ]
        
        self.logger.info(f"CNN DEBUG: [Step 3] Membangun tensor training ({len(train_samples)} valid samples).")
        X_train = None
        y_train = None

        if len(train_samples) > 0:
            X_train = np.array([self.tensor_builder.construct_input_tensor(r) for _, r in train_samples.iterrows()])

            # y_train_dict harus dictionary 2 output
            y_train_dict = {
                "dir_output": np.array([self.tensor_builder.construct_ground_truth(r)["dir_output"]
                                        for _, r in train_samples.iterrows()]),
                "angle_output": np.array([self.tensor_builder.construct_ground_truth(r)["angle_output"]
                                          for _, r in train_samples.iterrows()])
            }


        # --------------------------------------------------------
        # 3. MODEL MANAGEMENT (Load vs Rebuild)
        # --------------------------------------------------------
        model_exists = os.path.exists(self.paths["model_file"])

        if model_exists:
            try:
                self.model = load_model(self.paths["model_file"], compile=False)
                # Pastikan model memiliki 2 output
                if not isinstance(self.model.output, list) or len(self.model.output) != 2:
                    self.logger.warning("Model lama tidak kompatibel → rebuild required.")
                    self.model = None
                    model_exists = False
                else:
                    # Recompile dengan metrics custom
                    self.model.compile(
                        optimizer=Adam(0.001),
                        loss=["categorical_crossentropy", "mse"],
                        metrics={"dir_output": "accuracy", "angle_output": "mae"}
                    )
                    self.logger.info("Model CNN berhasil dimuat dari disk dan valid.")
            except Exception as e:
                self.logger.warning(f"Model corrupt/incompatible → force rebuild. Error: {e}")
                self.model = None
                model_exists = False

        if not model_exists or self.model is None:
            self.logger.info("Membangun ulang model CNN dari awal...")

            # Sesuai catatan client: 5 input, hidden 2–3 layer, output 2 node
            self.model = self.architect.build_model(
                input_shape=(self.grid_size, self.grid_size, self.input_channels),  # 5 channel
                hidden_nodes=[128, 64]  # atau [128,64,32] bebas
            )

            # Compile dengan loss sesuai output
            self.model.compile(
                optimizer=Adam(0.001),
                loss=["categorical_crossentropy", "mse"],
                metrics={"dir_output": "accuracy", "angle_output": "mae"}
            )

        # --------------------------------------------------------
        # 5. TRAINING LOOP (SISIPKAN INI)
        # Bagian ini wajib ada supaya training_log.csv terupdate
        # --------------------------------------------------------
        if X_train is not None and len(X_train) >= 2:
            self.logger.info(f"CNN DEBUG: [Step 4] Training Start ({self.epochs} epochs)...")
            
            # Definisi Callbacks
            callbacks_list = [
                # Simpan model terbaik
                ModelCheckpoint(self.paths["model_file"], save_best_only=True, verbose=0),
                # CATAT LOG TRAINING (Ini yang Anda cari)
                CSVLogger(self.paths["training_log"], append=True) 
            ]
            
            try:
                # Lakukan Training
                self.model.fit(
                    X_train, y_train_dict,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=0,
                    callbacks=callbacks_list
                )
                self.logger.info("CNN DEBUG: Training selesai, log tersimpan.")
            except Exception as e:
                self.logger.error(f"CNN Training Error: {e}")
        else:
            self.logger.warning("CNN DEBUG: Data training tidak cukup (<2 samples), skip training.")

        # --------------------------------------------------------
        # 4. PREDICTION PHASE
        # --------------------------------------------------------
        try:
            self.logger.info("CNN DEBUG: [Step 5] Prediksi pada Full Data.")

            X_full = np.array([self.tensor_builder.construct_input_tensor(r)
                               for _, r in df_main.iterrows()])

            # Safety fix shape
            if X_full.ndim != 4 or X_full.shape[-1] != self.input_channels:
                fixed = np.zeros((len(X_full), self.grid_size, self.grid_size, self.input_channels), dtype=np.float32)
                h = min(X_full.shape[1], self.grid_size)
                w = min(X_full.shape[2], self.grid_size)
                d = min(X_full.shape[3], self.input_channels)
                fixed[:, :h, :w, :d] = X_full[:, :h, :w, :d]
                X_full = fixed

            preds = self.model.predict(X_full, verbose=0)

            # Validasi output model
            if not isinstance(preds, (list, tuple)) or len(preds) != 2:
                self.logger.critical(f"CNN CRASH: Output model tidak sesuai.")
                return df_main

            pred_dir_probs, pred_angle_norm = preds

            # Ambil data baris terakhir untuk disimpan sebagai state terkini
            last_idx = -1
            
            # 1. Output Arah (4 Sumbu)
            dir_probs = pred_dir_probs[last_idx]
            arah_idx = np.argmax(dir_probs)
            # Mapping 4 sumbu
            dir_map = {0: "Timur", 1: "Barat", 2: "Selatan", 3: "Utara"} 
            arah_str = dir_map.get(arah_idx, "Unknown")
            
            # 2. Output Sudut
            az = float(pred_angle_norm.flatten()[last_idx]) * 360.0

            # 3. Output Risiko (k) sebagai ARRAY [PERBAIKAN UTAMA]
            # Client minta risiko (k) dibuat array. Kita ambil max probability sebagai confidence.
            conf_val = float(np.max(dir_probs))
            risk_array = np.array([conf_val]) # <-- DIBUAT ARRAY SESUAI REQUEST

            # Simpan ke CSV
            output_df = pd.DataFrame([{
                "timestamp": pd.Timestamp.now(),
                "arah_prediksi": arah_str,
                "arah_derajat": az,
                "risk_k_array": str(risk_array.tolist()), 
                "confidence_scalar": conf_val,
                "sumber": "CNN_Simple_Adaptive"
            }])

            out_path = self.paths.get("cnn_prediction_out")
            if out_path:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                # Cek apakah file sudah ada. Jika sudah, append tanpa header.
                # Jika belum, buat baru dengan header.
                file_exists = os.path.isfile(out_path)
                
                output_df.to_csv(
                    out_path, 
                    mode='a',          # 'a' berarti APPEND (tambahkan ke bawah)
                    header=not file_exists, # Tulis header hanya jika file belum ada
                    index=False
                )
                # ----------------------------------------------------

                self.logger.info(
                    f"CNN OUTPUT SAVED -> Arah: {arah_str} ({az:.2f}°) | Risk Array: {risk_array}"
                )

            # [PENTING] Kembalikan nilai ke df_main agar bisa diambil GA
            # Kita inject kolom khusus untuk array risiko
            df_main["CNN_Risk_Array"] = None 
            df_main.at[df_main.index[-1], "CNN_Risk_Array"] = risk_array

        except Exception as e:
            self.logger.critical(f"CNN CRASH PRED: Error saat prediksi: {e}")
            df_main["Proporsi_Grid_Terdampak_CNN"] = 0.0

        return df_main