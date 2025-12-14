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
    from keras.models import Model, load_model
    from keras.layers import (
        Input, Conv2D, MaxPooling2D,
        Conv2DTranspose, concatenate,
        BatchNormalization, Activation, Add, Dropout
    )
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("TensorFlow not found. CNN Engine disabled.")

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
        
        kedalaman_safe = row.get("Kedalaman_km", 0)
        log_depth = np.log1p(kedalaman_safe) if kedalaman_safe > -1 else 0.0
        
        # Channel Construction (Safe Map)
        c1 = np.full((gs, gs), self._safe_map(row.get("Magnitudo"), 1/10), dtype=np.float32)
        c2 = np.full((gs, gs), self._safe_map(log_depth), dtype=np.float32)
        c3 = np.full((gs, gs), self._safe_map(row.get("Context_Risk_Pheromone")), dtype=np.float32)
        c4 = np.full((gs, gs), np.clip(self._safe_map(row.get("Temporal_Risk_Factor")), 0, 1), dtype=np.float32)
        c5 = np.full((gs, gs), self._safe_map(row.get("Cluster"), 1/10), dtype=np.float32)

        stacked = np.stack([c1, c2, c3, c4, c5], axis=-1)

        # Safety Check Dimension
        if stacked.shape != (gs, gs, 5):
            # self.logger.warning(f"Auto-resizing tensor from {stacked.shape} to {(gs,gs,5)}")
            # Slice/Pad safely if dimensions mismatch
            fixed = np.zeros((gs, gs, 5), dtype=np.float32)
            h = min(gs, stacked.shape[0])
            w = min(gs, stacked.shape[1])
            d = min(5, stacked.shape[2])
            fixed[:h, :w, :d] = stacked[:h, :w, :d]
            stacked = fixed

        return stacked.astype(np.float32)

    def construct_ground_truth(self, row: pd.Series) -> np.ndarray:
        gs = self.grid_size
        mask = np.zeros((gs, gs), dtype=np.float32)

        rad = row.get("Context_Impact_Radius", row.get("R_true", 0.0))

        try:
            rad = float(rad)
            if np.isnan(rad) or rad < 0:
                rad = 0
        except:
            rad = 0

        if rad <= 0:
            return mask[..., None]

        km_per_pixel = 100.0 / gs
        pixel_radius = rad / km_per_pixel

        cx = cy = gs // 2
        y, x = np.ogrid[:gs, :gs]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        mask[dist <= pixel_radius] = 1.0
        return mask[..., None]


# ============================================================
#  RES-UNET ARCHITECTURE (SAFE)
# ============================================================

class CNNModelArchitect:
    def _res_block(self, x, filters):
        shortcut = x

        x = Conv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding="same")(shortcut)

        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x

    # [FIX] Renamed to build_model to match caller expectation
    def build_model(self, params: Dict = None, input_shape=None):
        if input_shape is None:
            # Fallback jika input_shape tidak disediakan: (grid_size, grid_size, 5)
            gs = params.get("grid_size", 32) if params else 32
            input_shape = (gs, gs, 5)

        inputs = Input(input_shape)

        c1 = self._res_block(inputs, 16)
        p1 = MaxPooling2D()(c1)

        c2 = self._res_block(p1, 32)
        p2 = MaxPooling2D()(c2)

        c3 = self._res_block(p2, 64)
        c3 = Dropout(0.15)(c3)

        u4 = Conv2DTranspose(32, 2, strides=2, padding="same")(c3)
        u4 = concatenate([u4, c2])
        c4 = self._res_block(u4, 32)

        u5 = Conv2DTranspose(16, 2, strides=2, padding="same")(c4)
        u5 = concatenate([u5, c1])
        c5 = self._res_block(u5, 16)

        outputs = Conv2D(1, 1, activation="sigmoid")(c5)

        model = Model(inputs, outputs)
        
        # Compile menggunakan fungsi metrik lokal agar independen
        model.compile(
            optimizer=Adam(0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", dice_coef, iou_score],
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

            os.makedirs(os.path.dirname(self.paths["realtime_viz"]), exist_ok=True)
            plt.savefig(self.paths["realtime_viz"], dpi=150, bbox_inches="tight")
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
        try:
            # Gunakan lokasi file ini sebagai referensi untuk naik ke root project
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../..")) 
            base_output = os.path.join(project_root, "output")
            
            # ATUR self.paths di sini
            self.paths = {
                "lstm_bridge_in": os.path.join(base_output, "lstm_results/lstm_data_for_cnn.xlsx"),
                "model_file": os.path.join(base_output, "cnn_results/ResUNetXL.keras"),
                "training_log": os.path.join(base_output, "cnn_results/training_log.csv"),
                "realtime_viz": os.path.join(base_output, "cnn_results/latest_heatmap.png"),
            }
            # Buat direktori output jika belum ada
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
        """Memuat data bridging dari LSTM Engine."""
        if not self.paths:
             return pd.DataFrame()
        
        path_in = self.paths.get("lstm_bridge_in", "")
        if not path_in or not os.path.exists(path_in):
            # Silent fail agar log tidak penuh spam jika file belum terbuat
            self.logger.warning(f"CNN BRIDGE: File {path_in} not found. Skipping.")
            return pd.DataFrame()

        try:
            df = pd.read_excel(path_in).fillna(0.0)
            return df
        except Exception as e:
            self.logger.error(f"Gagal membaca LSTM bridge Excel: {e}")
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
            try:
                X_train = np.array([
                    self.tensor_builder.construct_input_tensor(r) for _, r in train_samples.iterrows()
                ])
                y_train = np.array([
                    self.tensor_builder.construct_ground_truth(r) for _, r in train_samples.iterrows()
                ])
            except Exception as e:
                self.logger.error(f"CNN ERROR: Gagal membuat training tensors: {e}")
                X_train, y_train = None, None

        # --------------------------------------------------------
        # 3. MODEL MANAGEMENT (Load vs Rebuild)
        # --------------------------------------------------------
        model_exists = os.path.exists(self.paths["model_file"])

        if model_exists:
            try:
                # [FIX SYNTAX]: Menutup kurung dengan benar dan passing local custom objects
                self.model = load_model(
                    self.paths["model_file"],
                    custom_objects={
                        "dice_coef": dice_coef,
                        "iou_score": iou_score
                    }
                ) 
                self.logger.info("Model CNN berhasil dimuat dari disk.")
            except Exception as e:
                self.logger.warning(f"Model corrupt/incompatible → force rebuild. Error: {e}")
                model_exists = False
        
        # Build New Model if needed
        if not model_exists or self.model is None:
            self.logger.info("Membangun ulang model CNN dari awal...")
            
            # Panggil build_model (parameter params & shape input explicit)
            self.model = self.architect.build_model(
                params=self.cnn_cfg, 
                input_shape=(self.grid_size, self.grid_size, self.input_channels)
            )

            # TRAIN Loop (hanya jika data cukup)
            if X_train is not None and len(X_train) >= 5: # Threshold minimal 5 sampel
                self.logger.info(f"Training CNN on {len(X_train)} samples...")
                
                try:
                    self.model.fit(
                        X_train, y_train,
                        batch_size=self.batch_size, 
                        epochs=self.epochs, 
                        validation_split=0.2,
                        callbacks=[
                            EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0),
                            ModelCheckpoint(self.paths["model_file"], save_best_only=True, verbose=0),
                            CSVLogger(self.paths["training_log"], append=True)
                        ],
                        verbose=0
                    )
                except Exception as e:
                    self.logger.error(f"CNN Training Failed: {e}")
            else:
                self.logger.warning("Data training < 5. Model menggunakan bobot random (untrained).")

        # --------------------------------------------------------
        # 4. PREDICTION PHASE
        # --------------------------------------------------------
        self.logger.info("CNN DEBUG: [Step 5] Prediksi pada Full Data.")
        
        try:
            # Gunakan df_main sebagai target prediksi (karena ini output yang direquest orchestrator)
            X_full = np.array([
                self.tensor_builder.construct_input_tensor(r)
                for _, r in df_main.iterrows()
            ])

            if len(X_full) == 0:
                df_main["Proporsi_Grid_Terdampak_CNN"] = 0.0
                return df_main

            preds = self.model.predict(X_full, verbose=0)
            
            # Menghitung proporsi dampak (rata-rata pixel mask > 0)
            # Karena activation sigmoid, ini adalah probabilitas rata-rata
            impact_scores = np.mean(preds, axis=(1, 2, 3))

            # Sync panjang array untuk menghindari Broadcast Error
            impact_scores, = sync_len(impact_scores)
            n_safe = min(len(df_main), len(impact_scores))

            # Assignment dengan slicing aman
            df_main["Proporsi_Grid_Terdampak_CNN"] = 0.0
            # [FIX] Assignment menggunakan iloc/loc yang alignment-safe
            df_main.iloc[:n_safe, df_main.columns.get_loc("Proporsi_Grid_Terdampak_CNN")] = impact_scores[:n_safe]

            # Visualization Sample (Last Valid Row)
            if n_safe > 0:
                self.viz.visualize_last(preds[n_safe-1], df_main.iloc[n_safe-1])

        except Exception as e:
             self.logger.critical(f"CNN CRASH PRED: Error saat prediksi: {e}")
             df_main["Proporsi_Grid_Terdampak_CNN"] = 0.0

        return df_main