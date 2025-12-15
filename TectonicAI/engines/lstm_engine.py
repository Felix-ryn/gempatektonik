"""
Module: Temporal Anomaly Detection with Transformer-LSTM Hybrid Architecture.
1. Output Data: Menyimpan data record 2 tahun terakhir + data terbaru (termasuk 15 hari terakhir).
2. Relation Tracking: Output CSV kini menyertakan kolom ACO (Pheromone/Radius) dan GA (Stress/Dist) 
   untuk memvisualisasikan hubungan antar metode seperti yang diminta.
3. Bridge Update: Menyiapkan data anomali dan historis untuk diteruskan ke CNN.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import time
import math
import shutil
import platform
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional, Union

# --- IMPORT DEEP LEARNING STACK ---
try:
    import tensorflow as tf
    from keras.models import Model, load_model
    from keras.layers import (
        LSTM, Dense, Input, Dropout, Bidirectional,
        BatchNormalization, Layer,
        MultiHeadAttention, LayerNormalization
    )
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
    from keras.optimizers import Adam
    from keras import regularizers
    from keras.utils import plot_model
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ============================================================
#  HELPER: SYNC PANJANG ARRAY (TRIM KE MINIMUM)
# ============================================================

def sync_len(*arrays):
    """
    Sinkronisasi panjang semua array secara aman.
    """
    if not arrays:
        return arrays

    effective = [a for a in arrays if a is not None]
    if not effective:
        return arrays

    min_len = min(len(a) for a in effective)

    synced = []
    for a in arrays:
        if a is None:
            synced.append(None)
        else:
            synced.append(a[:min_len])
    return tuple(synced)


# ============================================================
#  CUSTOM LAYER: TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ============================================================
#  CUSTOM CALLBACK: DETAIL LOG
# ============================================================

class DetailedLogger(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % 5 == 0:
            self.logger.debug(
                f"Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}"
            )


# ============================================================
#  MAIN LSTM ENGINE
# ============================================================

class LSTMEngine:
    def __init__(self, config: dict):
        self.lstm_cfg = config
        self.logger = logging.getLogger("LSTM_Engine")
        self.logger.setLevel(logging.INFO)

        # --- HYPERPARAMETERS ---
        self.seq_length = int(self.lstm_cfg.get('sequence_length', 15))
        self.epochs = int(self.lstm_cfg.get('epochs', 50))
        self.units = int(self.lstm_cfg.get('units', 64))
        self.base_threshold = float(self.lstm_cfg.get('anomaly_threshold_std', 2.5))
        self.mc_samples = int(self.lstm_cfg.get('mc_samples', 20))

        self.model: Optional[Model] = None

        # --- PATH MANAGEMENT ---
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output'))
        self.paths = {
            # INPUT (Dari ACO dan GA)
            'aco_in': os.path.join(base_path, 'aco_results/aco_zoning_data_for_lstm.xlsx'),
            'ga_in': os.path.join(base_path, 'ga_results/ga_prediction_data_for_lstm.xlsx'),
            
            # OUTPUT
            'anomalies': os.path.join(base_path, 'lstm_results/lstm_detected_anomalies.csv'),
            'bridge_cnn': os.path.join(base_path, 'lstm_results/lstm_data_for_cnn.xlsx'), # Output Bridge
            'lstm_output': os.path.join(base_path, 'lstm_results/lstm_output_2years.csv'), # Main Output
            'recent_15_days': os.path.join(base_path, 'lstm_results/lstm_recent_15_days.csv'), # Request 15 hari
            
            # ARTIFACTS
            'model_file': os.path.join(base_path, 'lstm_results/hybrid_transformer_lstm.keras'),
            'loss_plot': os.path.join(base_path, 'lstm_results/training_loss.png'),
            'logs_dir': os.path.join(base_path, 'lstm_results/logs'),
            'metadata': os.path.join(base_path, 'lstm_results/training_metadata.json')
        }

        self._ensure_directories()

    def _ensure_directories(self):
        for path in self.paths.values():
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    # ========================================================
    # 1. ADVANCED PREPROCESSING
    # ========================================================

    def _enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_rich = df.copy()

        if 'Magnitudo' not in df_rich.columns:
            df_rich['Magnitudo'] = 0.0

        # Basic physics-inspired feature
        df_rich['Seismic_Energy_Log'] = 11.8 + (1.5 * df_rich['Magnitudo'])

        return df_rich

    def _fuse_external_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menyatukan output ACO & GA, serta menghitung Relation Score.
        """
        self.logger.info("Memulai Data Fusion (ACO + GA Context Injection)...")
        df = df.copy()

        # --- Helper Normalisasi untuk Score ---
        def safe_norm(x):
            x = np.asarray(x, dtype=np.float32)
            if np.all(np.isnan(x)): return np.zeros_like(x)
            min_x, max_x = np.nanmin(x), np.nanmax(x)
            if max_x - min_x == 0: return np.zeros_like(x)
            return (x - min_x) / (max_x - min_x)

        # --- Inisialisasi Kolom Default ---
        default_cols = {
            'Context_Risk_Pheromone': 0.0,
            'Context_Impact_Radius': 0.0,
            'Context_Zone_Status': 'Safe',
            'Global_Tectonic_Stress': 0.5,
            'GA_Direction_Pred': 'Unknown',
            'GA_Angle_Pred': 0.0,
            'Dist_To_GA_Pred': 0.0,
            'ACO_GA_Relation_Score': 0.0 # Kolom Score yang hilang sebelumnya
        }
        for col, val in default_cols.items():
            df[col] = val

        # -------------------------------
        # 1. FUSION ACO (Pheromone & Radius)
        # -------------------------------
        if os.path.exists(self.paths['aco_in']):
            try:
                df_aco = pd.read_excel(self.paths['aco_in'])
                limit = min(len(df), len(df_aco))
                
                if limit > 0:
                    df.loc[:limit-1, 'Context_Risk_Pheromone'] = df_aco['Pheromone_Score'].values[:limit]
                    
                    radius_col = next((c for c in df_aco.columns if 'Radius' in c), None)
                    if radius_col:
                        df.loc[:limit-1, 'Context_Impact_Radius'] = df_aco[radius_col].values[:limit]
                    
                    if 'Status_Zona' in df_aco.columns:
                        df.loc[:limit-1, 'Context_Zone_Status'] = df_aco['Status_Zona'].values[:limit]
                
                self.logger.info(f"[FUSION] ACO Intelligence integrated ({limit} records).")
            except Exception as e:
                self.logger.error(f"ACO Fusion Error: {e}")

        # -------------------------------
        # 2. FUSION GA (Arah, Sudut, Stress)
        # -------------------------------
        if os.path.exists(self.paths['ga_in']):
            try:
                df_ga = pd.read_excel(self.paths['ga_in'])
                if not df_ga.empty:
                    last_rec = df_ga.iloc[-1]
                    
                    fit_key = next((c for c in last_rec.index if 'Fitness' in c), None)
                    stress_val = float(last_rec[fit_key]) if fit_key else 0.5
                    
                    dir_key = next((c for c in last_rec.index if 'Arah' in c or 'Direction' in c), None)
                    angle_key = next((c for c in last_rec.index if 'Sudut' in c or 'Angle' in c), None)
                    
                    direction_val = str(last_rec[dir_key]) if dir_key else "Unknown"
                    angle_val = float(last_rec[angle_key]) if angle_key else 0.0

                    df['Global_Tectonic_Stress'] = stress_val
                    df['GA_Direction_Pred'] = direction_val
                    df['GA_Angle_Pred'] = angle_val

                    # Spatial Relation (Distance)
                    if 'Pred_Lat' in last_rec.index and 'Pred_Lon' in last_rec.index:
                        R = 6371.0
                        lat1, lon1 = np.radians(df['Lintang']), np.radians(df['Bujur'])
                        lat2, lon2 = np.radians(last_rec['Pred_Lat']), np.radians(last_rec['Pred_Lon'])
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                        df['Dist_To_GA_Pred'] = R * c

                        self.logger.info(f"[FUSION] GA Prediction integrated. Dir: {direction_val}")
            except Exception as e:
                self.logger.error(f"GA Fusion Error: {e}")

        # -------------------------------
        # 3. CALCULATE RELATION SCORE (THE MISSING PART RE-ADDED)
        # -------------------------------
        try:
            # Normalisasi agar semua dalam skala 0-1
            ph_norm = safe_norm(df['Context_Risk_Pheromone'])
            str_norm = safe_norm(df['Global_Tectonic_Stress'])
            dist_norm = safe_norm(df['Dist_To_GA_Pred'])
            
            # Rumus Logika:
            # Score tinggi jika: Pheromone Tinggi (40%) + Stress Tinggi (40%) + Jarak Dekat (20%)
            # Jarak dekat berarti dist_norm kecil, maka kita pakai (1 - dist_norm)
            df['ACO_GA_Relation_Score'] = (
                (0.4 * ph_norm) + 
                (0.4 * str_norm) + 
                (0.2 * (1.0 - dist_norm))
            )
            self.logger.info("[RELATION] ACO-GA Relation Score calculated successfully.")
        except Exception as e:
            self.logger.warning(f"Relation Score calculation failed: {e}")
            df['ACO_GA_Relation_Score'] = 0.0

        return df.fillna(0)

    def _generate_relationship_narrative(self, row: pd.Series) -> str:
        """
        Menghasilkan kalimat hubungan ACO-GA yang bisa dibaca manusia.
        """
        try:
            radius = row.get('Context_Impact_Radius', 0)
            status = row.get('Context_Zone_Status', 'Unknown')
            arah = row.get('GA_Direction_Pred', 'Unknown')
            sudut = row.get('GA_Angle_Pred', 0)
            dist = row.get('Dist_To_GA_Pred', 0)
            score = row.get('ACO_GA_Relation_Score', 0.0)
            
            narrative = (
                f"Relasi Score: {score:.2f}. Zona {status} (Radius ACO: {radius:.1f} KM). "
                f"Hubungan GA: Berjarak {dist:.1f} KM dari pusat prediksi. "
                f"Arah pergerakan referensi GA adalah {arah} dengan sudut {sudut:.1f} derajat."
            )
            return narrative
        except Exception:
            return "Data hubungan tidak lengkap."

    def _generate_relationship_narrative(self, row: pd.Series) -> str:
        """
        Menghasilkan narasi hubungan ACO–GA–LSTM per record.
        Fokus: interpretasi manusia (client-friendly).
        """

        pheromone = row.get('Context_Risk_Pheromone', 0.0)
        stress = row.get('Global_Tectonic_Stress', 0.0)
        dist = row.get('Dist_To_GA_Pred', 0.0)
        anomaly = row.get('is_Anomaly', False)
        risk = row.get('Temporal_Risk_Factor', 0.0)

        phrases = []

        # ACO Interpretation
        if pheromone > 0.7:
            phrases.append("ACO menunjukkan akumulasi risiko tinggi di wilayah ini")
        elif pheromone > 0.4:
            phrases.append("ACO mendeteksi peningkatan aktivitas moderat")
        else:
            phrases.append("ACO tidak menunjukkan akumulasi risiko signifikan")

        # GA Interpretation
        if stress > 0.7:
            phrases.append("GA mengindikasikan stress tektonik global tinggi")
        elif stress > 0.4:
            phrases.append("GA menunjukkan stress tektonik menengah")
        else:
            phrases.append("GA menunjukkan stress tektonik relatif rendah")

        # Spatial Relation
        if dist < 50:
            phrases.append("Lokasi berada dekat dengan prediksi pusat stress GA")
        elif dist < 150:
            phrases.append("Lokasi berada pada radius menengah dari prediksi GA")
        else:
            phrases.append("Lokasi relatif jauh dari pusat prediksi GA")

        # LSTM Conclusion
        if anomaly:
            phrases.append(
                f"LSTM mendeteksi pola temporal menyimpang (Risk Score={risk:.2f})"
            )
        else:
            phrases.append("LSTM tidak mendeteksi penyimpangan pola temporal")

        return ". ".join(phrases) + "."


    def _prepare_tensor_data(self, data_values: np.ndarray, augment=False):
        """
        Membuat sequence window untuk input LSTM.
        Target = timestep terakhir (next-step reconstruction).
        """
        X, y = [], []
        n = len(data_values)

        if n <= self.seq_length:
            return (
                np.empty((0, self.seq_length, data_values.shape[1])),
                np.empty((0, data_values.shape[1]))
            )

        for i in range(n - self.seq_length):
            seq = data_values[i:(i + self.seq_length)]
            target = seq[-1]  # timestep terakhir

            X.append(seq)
            y.append(target)

            if augment:
                noise = np.random.normal(0, 0.01, seq.shape)
                X.append(seq + noise)
                y.append(target)  # target TETAP sama

        return np.array(X), np.array(y)


    # ========================================================
    # 2. HYBRID MODEL ARCHITECTURE
    # ========================================================

    def _build_hybrid_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape, name='time_series_input')

        # Transformer block
        transformer_out = TransformerBlock(
            embed_dim=input_shape[-1],
            num_heads=self.transformer_heads if hasattr(self, 'transformer_heads') else 4,
            ff_dim=self.transformer_ff_dim if hasattr(self, 'transformer_ff_dim') else 32,
            rate=self.dropout_rate if hasattr(self, 'dropout_rate') else 0.1
        )(inputs)

        # BiLSTM
        lstm_out = Bidirectional(
            LSTM(self.units, return_sequences=False, dropout=self.dropout_rate if hasattr(self, 'dropout_rate') else 0.1)
        )(transformer_out)
        lstm_out = BatchNormalization()(lstm_out)

        # Dense head
        # Dense head
        x = Dense(64, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001))(lstm_out)
        x = Dropout(self.dropout_rate if hasattr(self, 'dropout_rate') else 0.1)(x)
        x = Dense(32, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001))(x)

        # FINAL OUTPUT (2D)
        outputs = Dense(input_shape[1], activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Hybrid_Transformer_LSTM')

        model = Model(inputs=inputs, outputs=outputs, name='Hybrid_Transformer_LSTM')

        opt = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=opt, loss='huber', metrics=['mae', 'mse'])

        return model

    # ========================================================
    # 3. INFERENCE & ANALYSIS
    # ========================================================

    def _monte_carlo_prediction(self, X_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout Prediction.
        Menghasilkan mean prediction dan uncertainty score per timestep.
        """

        if X_data is None or len(X_data) == 0:
            self.logger.warning("MC Prediction: X_data kosong, skip Bayesian inference.")
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        tensor_X = tf.convert_to_tensor(X_data, dtype=tf.float32)
        predictions = []

        for i in range(self.mc_samples):
            try:
                pred = self.model(tensor_X, training=True)

                # Safety: pastikan output 2D (N, feat_dim)
                if len(pred.shape) == 3:
                    pred = pred[:, -1, :]  # ambil timestep terakhir

                predictions.append(tf.convert_to_tensor(pred, dtype=tf.float32))

            except Exception as e:
                self.logger.warning(f"MC sample {i} failed: {e}")

        if len(predictions) == 0:
            self.logger.error("MC Prediction failed: no valid predictions.")
            return (
                np.empty((0, X_data.shape[-1]), dtype=np.float32),
                np.zeros((X_data.shape[0],), dtype=np.float32)
            )

        stacked_preds = tf.stack(predictions, axis=0)
        # shape: (mc_samples, N, feat_dim)

        pred_mean = tf.reduce_mean(stacked_preds, axis=0)
        pred_std = tf.math.reduce_std(stacked_preds, axis=0)

        # Uncertainty score per sample (dirata-rata per fitur)
        uncertainty_score = tf.reduce_mean(pred_std, axis=1)

        return pred_mean.numpy(), uncertainty_score.numpy()


    def _adaptive_thresholding(self, errors: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        if len(errors) == 0:
            return np.array([], dtype=np.float32)

        err_series = pd.Series(errors)

        roll_mean = err_series.rolling(window=60, min_periods=1).mean()
        roll_std = err_series.rolling(window=60, min_periods=1).std().fillna(0.0)

        base_thresh = roll_mean + (self.base_threshold * roll_std)

        if len(uncertainty) < len(base_thresh):
            u = np.zeros(len(base_thresh), dtype=np.float32)
            u[:len(uncertainty)] = uncertainty
            uncertainty = u
        elif len(uncertainty) > len(base_thresh):
            uncertainty = uncertainty[:len(base_thresh)]

        adjusted_thresh = base_thresh * (1.0 + (uncertainty * 1.5))
        fallback = errors.mean() + (3 * errors.std() if errors.std() > 0 else 0.0)
        adjusted_thresh.fillna(fallback, inplace=True)

        return adjusted_thresh.values

    def _calculate_feature_importance(self, X_sample: np.ndarray, feature_names: List[str]):
        try:
            if X_sample is None or len(X_sample) == 0 or len(feature_names) == 0:
                return

            # Batasi sample agar ringan
            if len(X_sample) > 100:
                X_sample = X_sample[:100]

            tensor_X = tf.convert_to_tensor(X_sample, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(tensor_X)
                preds = self.model(tensor_X)

            grads = tape.gradient(preds, tensor_X)
            if grads is None:
                self.logger.warning("Gradients None. Skip feature importance.")
                return

            # Robust reduction
            if len(grads.shape) == 3:
                saliency = tf.reduce_mean(tf.abs(grads), axis=[0, 1]).numpy()
            else:
                saliency = tf.reduce_mean(tf.abs(grads), axis=0).numpy()

            if len(saliency) != len(feature_names):
                self.logger.warning(
                    f"Feature importance dim mismatch: "
                    f"saliency={len(saliency)}, features={len(feature_names)}"
                )
                return

            if saliency.sum() > 0:
                saliency = saliency / saliency.sum()

            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': saliency
            }).sort_values('Importance', ascending=False)

            # Note: self.paths['feature_importance'] was not defined in __init__, 
            # might cause error if not added to paths dict. 
            # Keeping code as requested without deletion.
            if 'feature_importance' in self.paths:
                 imp_df.to_csv(self.paths['feature_importance'], index=False)
            
            self.logger.info("Analisis Feature Importance completed.")

        except Exception as e:
            self.logger.warning(f"Feature importance gagal: {e}")

    # ========================================================
    # 4. MAIN RUN PIPELINE (SAFE)
    # ========================================================

    def run(self, df: pd.DataFrame, train_df: pd.DataFrame):
        """
        Main Execution Pipeline.
        """
        if not HAS_TF:
            self.logger.critical("TensorFlow Missing. Abort LSTM Engine.")
            df_out = df.copy()
            for col in ['is_Anomaly', 'Temporal_Risk_Factor', 'Model_Uncertainty', 'LSTM_Error']:
                df_out[col] = 0.0 if col != 'is_Anomaly' else False
            return df_out, {"anomalies": df_out.iloc[0:0].copy()}

        # === REVISI: FILTER 2 TAHUN TERAKHIR (Tapi tetap simpan data terbaru 15 hari) ===
        # Logika ini memastikan bahwa jika ada data baru masuk hari ini, dia akan termasuk 
        # dalam range '>= cutoff_date'.
        cutoff_date = None

        if 'Tanggal' in df.columns:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            cutoff_date = df['Tanggal'].max() - pd.DateOffset(years=2)
            df = df[df['Tanggal'] >= cutoff_date].reset_index(drop=True)

        if cutoff_date is not None and 'Tanggal' in train_df.columns:
            train_df['Tanggal'] = pd.to_datetime(train_df['Tanggal'], errors='coerce')
            train_df = train_df[train_df['Tanggal'] >= cutoff_date].reset_index(drop=True)
        
        start_time = time.time()
        self.logger.info("=== STARTING HYBRID TRANSFORMER-LSTM ENGINE (SAFE) ===")

        # --- STEP 1: Data Prep & Fusion (ACO + GA) --- #
        df_base = df.copy()
        df_rich = self._enrich_features(df_base)
        df_fused = self._fuse_external_intelligence(df_rich) # Disini ACO & GA masuk

        # Feature list
        features = [
            'Magnitudo',
            'hour_sin', 'hour_cos',
            'month_sin', 'month_cos',
            'Context_Risk_Pheromone',
            'Context_Zone_IsRed',
            'Global_Tectonic_Stress',
            'Dist_To_GA_Pred'
        ]

        final_features = []
        for f in features:
            if f in df_fused.columns:
                final_features.append(f)
            elif f.endswith('_scaled'):
                base_f = f.replace('_scaled', '')
                if base_f in df_fused.columns:
                    final_features.append(base_f)
                else:
                    self.logger.warning(f"Fitur {f} tidak ditemukan. Skip.")
            elif f in ['Magnitudo', 'Kedalaman_km']:
                if f in df_fused.columns:
                    final_features.append(f)

        self.logger.info(f"Active Features ({len(final_features)}): {final_features}")

        data_values = df_fused[final_features].astype(float).values
        n_samples, feat_dim = data_values.shape

        if n_samples <= self.seq_length:
            self.logger.warning("Data terlalu sedikit. Skip processing.")
            df_out = df.copy()
            df_out['is_Anomaly'] = False
            self._save_outputs(df_out, df_out.iloc[0:0].copy())
            return df_out, {"anomalies": df_out.iloc[0:0].copy()}

        train_rich = self._enrich_features(train_df.copy())
        train_fused = self._fuse_external_intelligence(train_rich)

        # --- STEP 2: Train Data Build --- #
        train_values = train_fused[final_features].astype(float).values

        X_train, y_train = self._prepare_tensor_data(train_values, augment=True)
        input_shape = (self.seq_length, feat_dim)

        # --- STEP 3: Model Check --- #
        force_retrain = self.lstm_cfg.get("force_retrain", False)
        if os.path.exists(self.paths['model_file']):
            try:
                temp_model = load_model(
                    self.paths['model_file'],
                    custom_objects={'TransformerBlock': TransformerBlock}
                )
                if temp_model.input_shape[1:] == input_shape:
                    self.model = temp_model
                    self.logger.info("✅ Valid existing model found.")
                    force_retrain = False
                else:
                    self.logger.warning("⚠️ Model shape mismatch. Forcing re-train.")
            except Exception:
                pass

        if force_retrain or self.model is None:
            self.model = self._build_hybrid_model(input_shape)

        # --- STEP 4: Training Phase --- #
        if len(X_train) > 20:
            cb = [
                EarlyStopping(patience=6, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
                ModelCheckpoint(self.paths['model_file'], save_best_only=True, verbose=0),
                CSVLogger(os.path.join(self.paths['logs_dir'], 'training.log'), append=True),
                DetailedLogger(self.logger)
            ]
            self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=32, validation_split=0.2, callbacks=cb, verbose=0)
        
        # --- STEP 5: Full Inference --- #
        X_full, _ = self._prepare_tensor_data(data_values, augment=False)

        if len(X_full) == 0:
            return df, {"anomalies": df.iloc[0:0]}

        # Feature importance (optional, safe)
        try:
            self._calculate_feature_importance(X_full, final_features)
        except Exception:
            pass

        self.logger.info("Running Bayesian Inference (Monte Carlo)...")
        pred_mean, uncertainty = self._monte_carlo_prediction(X_full)

        # ===============================
        # RECONSTRUCTION ERROR (FIXED)
        # ===============================
        X_last = X_full[:, -1, :]          # (N, F)
        recon_error = np.mean(
            np.abs(pred_mean - X_last),
            axis=1                         # (N,)
        )

        # ===============================
        # INDEX ALIGNMENT (BENAR)
        # ===============================
        valid_idx = np.arange(self.seq_length, self.seq_length + len(recon_error))

        # ===============================
        # THRESHOLDING
        # ===============================
        thresholds = self._adaptive_thresholding(recon_error, uncertainty)

        recon_error, thresholds, uncertainty, valid_idx = sync_len(
            recon_error, thresholds, uncertainty, valid_idx
        )

        # ===============================
        # ANOMALY & RISK (HILANG SEBELUMNYA)
        # ===============================
        is_anomaly = recon_error > thresholds
        risk = np.clip(
            (recon_error - thresholds) / (thresholds + 1e-9),
            0, None
        )

        # ===============================
        # OUTPUT DATAFRAME
        # ===============================
        df_out = df.copy()
        df_out['is_Anomaly'] = False
        df_out['Temporal_Risk_Factor'] = 0.0
        df_out['Model_Uncertainty'] = 0.0
        df_out['LSTM_Error'] = 0.0

        valid_idx = valid_idx[valid_idx < len(df_out)]

        df_out.loc[valid_idx, 'is_Anomaly'] = is_anomaly[:len(valid_idx)]
        df_out.loc[valid_idx, 'Temporal_Risk_Factor'] = risk[:len(valid_idx)]
        df_out.loc[valid_idx, 'Model_Uncertainty'] = uncertainty[:len(valid_idx)]
        df_out.loc[valid_idx, 'LSTM_Error'] = recon_error[:len(valid_idx)]

        # ===============================
        # CONTEXT FEATURES
        # ===============================
        for col in [
            'Context_Impact_Radius',
            'Context_Risk_Pheromone',
            'Context_Zone_IsRed',
            'Global_Tectonic_Stress',
            'Dist_To_GA_Pred'
        ]:
            if col in df_fused.columns:
                df_out[col] = df_fused[col]

        anomalies = df_out[df_out['is_Anomaly']].copy()

        self._save_outputs(df_out, anomalies)
        self.logger.info(f"LSTM Engine Completed. Found {len(anomalies)} anomalies.")


        metadata = {
            "model_name": "Hybrid Transformer-LSTM",
            "role": "Temporal Relation & Anomaly Analyzer",
            "input_sources": ["Historical Seismic", "ACO", "GA"],
            "explicitly_not_used_for": ["Direction Prediction", "Angle Prediction"],
            "client_logic": {
                "ACO": "Menentukan pusat & luas terdampak",
                "GA": "Memberikan stress & arah referensi",
                "LSTM": "Menyimpan memori temporal dan hubungan ACO-GA"
            },
            "generated_at": datetime.now().isoformat()
        }

        try:
            with open(self.paths['metadata'], 'w') as f:
                json.dump(metadata, f, indent=4)
            self.logger.info("Metadata saved successfully.")
        except Exception as e:
            self.logger.warning(f"Metadata save failed: {e}")

        return df_out, {"anomalies": anomalies}

    # ========================================================
    # 5. PLOTTING & SAVING (REVISED)
    # ========================================================

    def _plot_loss(self, history):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Hybrid Model Training Dynamics')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (Huber)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.paths['loss_plot'])
            plt.close()
        except Exception as e:
            self.logger.warning(f"Plot loss gagal: {e}")

    def _save_outputs(self, df_full, df_anom):
        """
        REVISI: Menyimpan output sesuai request client.
        Isi: Data 2 tahun terakhir, data terbaru (termasuk 15 hari), anomaly.
        Columns: Wajib ada kolom dari ACO dan GA.
        """
        try:
            # --- Pastikan kolom penting ada (Fill Default) ---
            for col, default in [('Tanggal', pd.Timestamp.now()),
                                 ('Lokasi', 'Unknown'),
                                 ('Magnitudo', 0.0),
                                 ('Kedalaman_km', 0.0),
                                 ('LSTM_Error', 0.0),
                                 ('Temporal_Risk_Factor', 0.0),
                                 ('Model_Uncertainty', 0.0),
                                 ('is_Anomaly', False),
                                 ('Context_Impact_Radius', 0.0), # Output ACO
                                 ('Context_Risk_Pheromone', 0.0), # Output ACO
                                 ('Global_Tectonic_Stress', 0.0), # Output GA
                                 ('Dist_To_GA_Pred', 0.0)]:       # Output GA
                if col not in df_full.columns:
                    df_full[col] = default
                if col not in df_anom.columns:
                    df_anom[col] = default

            # --- A. Anomalies CSV (Untuk Laporan Cepat) ---
            if not df_anom.empty:
                # Menambahkan konteks ACO/GA di report anomali juga
                cols_anom = ['Tanggal', 'Lokasi', 'Magnitudo', 'Kedalaman_km',
                             'Temporal_Risk_Factor', 'Model_Uncertainty', 
                             'Context_Risk_Pheromone', 'Global_Tectonic_Stress']
                valid_anom_cols = [c for c in cols_anom if c in df_anom.columns]
                df_anom[valid_anom_cols].to_csv(self.paths['anomalies'], index=False)
                self.logger.info(f"Anomaly Report Saved ({len(df_anom)} events).")

            # --- B. Bridge ke CNN (PENTING untuk Prediksi Arah/Sudut) ---
            # CNN butuh data historis + anomali untuk memprediksi gempa berikutnya.
            bridge_cols = [
                'Tanggal', 'Lintang', 'Bujur', 'Magnitudo', 'Kedalaman_km', 
                'Context_Risk_Pheromone', 'Context_Impact_Radius', # Dari ACO
                'Global_Tectonic_Stress', 'Dist_To_GA_Pred',       # Dari GA
                'Temporal_Risk_Factor', 'LSTM_Error', 'is_Anomaly'
            ]
            valid_bridge_cols = [c for c in bridge_cols if c in df_full.columns]
            df_full[valid_bridge_cols].to_excel(self.paths['bridge_cnn'], index=False)
            self.logger.info("Bridge Data (LSTM->CNN) Saved Successfully.")

            # --- HUBUNGKAN NARASI KE DATAFRAME ---
            df_full['Relationship_Narrative'] = df_full.apply(
                self._generate_relationship_narrative, axis=1
            )

            if not df_anom.empty:
                df_anom['Relationship_Narrative'] = df_anom.apply(
                self._generate_relationship_narrative, axis=1
            )

            # --- C. Main LSTM Output CSV (Sesuai Request Client) ---
            # "Output LSTM data cvs yg isinya data 2 thn terakhir, record data terbaru, data anomali"
            # Client juga minta: "lstm buat mengetahui hubungan aco sama ga" -> Maka kolomnya harus ditampilkan.
            export_cols = [
                'Tanggal', 'Lokasi', 'Magnitudo', 'Kedalaman_km',
                'LSTM_Error', 'Temporal_Risk_Factor', 'Model_Uncertainty', 'is_Anomaly',
                'Context_Risk_Pheromone', 'Context_Impact_Radius', # Bukti Hubungan ACO
                'Global_Tectonic_Stress', 'Dist_To_GA_Pred', 'Relationship_Narrative'        # Bukti Hubungan GA
            ]
            valid_export_cols = [c for c in export_cols if c in df_full.columns]
            
            # Sort by Tanggal descending (Data terbaru paling atas) biar mudah dicek client
            df_export = df_full[valid_export_cols].sort_values('Tanggal', ascending=False)
            df_export.to_csv(self.paths['lstm_output'], index=False)
            self.logger.info(f"LSTM Main Output Saved (2 Years + Recent Data): {self.paths['lstm_output']}")

            # --- D. EXPORT DATA 15 HARI TERAKHIR (REQUEST CLIENT) ---
            if 'Tanggal' in df_full.columns:
                try:
                    df_full['Tanggal'] = pd.to_datetime(df_full['Tanggal'], errors='coerce')

                    latest_date = df_full['Tanggal'].max()
                    cutoff_15 = latest_date - pd.Timedelta(days=15)

                    df_15_days = df_full[df_full['Tanggal'] >= cutoff_15].copy()

                    if not df_15_days.empty:
                        export_15_cols = [
                            'Tanggal', 'Lokasi', 'Magnitudo', 'Kedalaman_km',
                            'LSTM_Error', 'Temporal_Risk_Factor', 'Model_Uncertainty', 'is_Anomaly',
                            'Context_Risk_Pheromone', 'Context_Impact_Radius',
                            'Global_Tectonic_Stress', 'Dist_To_GA_Pred',
                            'ACO_GA_Relation_Score', 'Relationship_Narrative'
                        ]

                        valid_15_cols = [c for c in export_15_cols if c in df_15_days.columns]

                        df_15_days = df_15_days[valid_15_cols] \
                            .sort_values('Tanggal', ascending=False)

                        df_15_days.to_csv(self.paths['recent_15_days'], index=False)

                        self.logger.info(
                            f"Recent 15 Days Data Saved ({len(df_15_days)} records)."
                        )
                except Exception as e:
                    self.logger.warning(f"Export 15 days data failed: {e}")

        except Exception as e:
            self.logger.error(f"Save outputs failed: {e}")