# ============================================================
# direction_deviation_lstm_engine.py
# ------------------------------------------------------------
# CLIENT ENGINE (FINAL)
# Deteksi anomali berbasis deviasi arah & sudut (GA vs CNN)
# Output: 2 Excel bersih sesuai permintaan client
# ------------------------------------------------------------
# Filosofi:
# - BUKAN reconstruction error
# - BUKAN risk / confidence CNN
# - LSTM = classifier deviasi antar-kejadian
# ============================================================

import math
import os
from typing import Tuple

import numpy as np
import pandas as pd

# --- TensorFlow (opsional, tapi direkomendasikan) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "direction_lstm_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "direction_deviation_prediction.csv"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def circular_angle_diff(a: float, b: float) -> float:
    """Selisih sudut aman (0 - 180 derajat)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def direction_distance(dir_a: str, dir_b: str) -> float:
    """
    Jarak arah kompas diskret.
    Output:
      0 = sama
      1 = bersebelahan
      2 = beda 90 derajat
      3 = hampir berlawanan
      4 = berlawanan
    """
    compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    if dir_a not in compass or dir_b not in compass:
        return 0.0
    i, j = compass.index(dir_a), compass.index(dir_b)
    d = abs(i - j)
    return min(d, 8 - d)


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Jarak haversine (km)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# MAIN ENGINE
# ============================================================

class DirectionDeviationLSTMEngine:
    """
    Engine LSTM khusus CLIENT.

    Input  : deviasi antar kejadian (GA vs CNN + perubahan ACO)
    Output : label anomali (True / False)
    """

    # --------------------------------------------------------
    # INIT
    # --------------------------------------------------------

    def __init__(
        self,
        seq_len: int = 2,
        angle_threshold: float = 30.0,
        dir_threshold: float = 2.0,
        model_path: str = 'direction_deviation_lstm.keras'
    ):
        self.seq_len = seq_len
        self.angle_threshold = angle_threshold
        self.dir_threshold = dir_threshold
        self.model_path = os.path.join(OUTPUT_DIR, model_path)
        self.model: Model | None = None

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Membangun fitur antar-kejadian.
        Asumsi kolom input:
        - GA_arah, GA_sudut
        - CNN_arah, CNN_sudut
        - ACO_lat, ACO_lon, ACO_area
        """
        # === [ANTI-CRASH CHECK] ===
        required_cols = [
            'GA_arah', 'GA_sudut',
            'CNN_arah', 'CNN_sudut',
            'ACO_lat', 'ACO_lon', 'ACO_area'
        ]

        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"[Direction LSTM] Kolom wajib tidak ditemukan: {c}")

        features = []

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            row = {
                'delta_angle': circular_angle_diff(prev['GA_sudut'], curr['CNN_sudut']),
                'delta_direction': direction_distance(prev['GA_arah'], curr['CNN_arah']),
                'delta_aco_center': haversine(
                    prev['ACO_lat'], prev['ACO_lon'],
                    curr['ACO_lat'], curr['ACO_lon']
                ),
                'delta_aco_area': abs(curr['ACO_area'] - prev['ACO_area'])
            }
            features.append(row)

        return pd.DataFrame(features)

    # --------------------------------------------------------
    # MODEL DEFINITION
    # --------------------------------------------------------

    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        inp = Input(shape=input_shape)
        x = LSTM(32, activation='tanh')(inp)
        out = Dense(1, activation='sigmoid')(x)

        model = Model(inp, out)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train(self, df: pd.DataFrame, epochs: int = 20, batch_size: int = 16):
        if not HAS_TF:
            raise RuntimeError('TensorFlow tidak tersedia')

        feat_df = self._build_features(df)

        # Label berbasis logika client (ground truth sederhana)
        labels = (
            (feat_df['delta_angle'] > self.angle_threshold) |
            (feat_df['delta_direction'] > self.dir_threshold)
        ).astype(int)

        X, y = [], []
        for i in range(len(feat_df) - self.seq_len + 1):
            X.append(feat_df.iloc[i:i + self.seq_len].values)
            y.append(labels.iloc[i + self.seq_len - 1])

        X = np.array(X)
        y = np.array(y)

        self.model = self._build_model((self.seq_len, X.shape[-1]))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.model.save(self.model_path)

    # --------------------------------------------------------
    # PREDICT & EXPORT
    # --------------------------------------------------------

    def predict_and_export(
        self,
        df: pd.DataFrame,
        out_old: str,
        out_new: str
    ):
        if not HAS_TF:
            raise RuntimeError('TensorFlow tidak tersedia')

        if self.model is None:
            if not os.path.exists(self.model_path):
                raise RuntimeError('Model belum dilatih')
            self.model = load_model(self.model_path)

        feat_df = self._build_features(df)

        X = []
        for i in range(len(feat_df) - self.seq_len + 1):
            X.append(feat_df.iloc[i:i + self.seq_len].values)
        X = np.array(X)

        preds = (self.model.predict(X, verbose=0).flatten() > 0.5)

        # Align ke dataframe asli (mulai dari index 1)
        df_out = df.iloc[1:].copy().reset_index(drop=True)
        df_out['anomali'] = False
        df_out.loc[self.seq_len - 1:, 'anomali'] = preds

        export_cols = [
            'Tanggal',
            'ACO_pusat',
            'ACO_area',
            'GA_arah',
            'GA_sudut',
            'anomali'
        ]

        df_old = df_out[df_out['Tanggal'].dt.year <= 2024][export_cols]
        df_new = df_out[df_out['Tanggal'].dt.year == 2025][export_cols]

        df_old.to_excel(out_old, index=False)
        df_new.to_excel(out_new, index=False)


    def run(
        self,
        df_dynamic: pd.DataFrame,
        train_context: pd.DataFrame
    ):
        """
        Entry point untuk ORCHESTRATOR
        """
        df_dynamic = df_dynamic.reset_index(drop=True)
        meta = {}

        # --- TRAIN ---
        if train_context is not None and len(train_context) > self.seq_len:
            self.train(train_context)
            meta["trained"] = True
        else:
            meta["trained"] = False

        # --- PREDICT ---
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise RuntimeError("Direction LSTM model tidak tersedia")
            self.model = load_model(self.model_path)

        feat_df = self._build_features(df_dynamic)

        if len(feat_df) < self.seq_len:
            df_dynamic["direction_anomaly"] = False
            meta["status"] = "insufficient_data"
            return df_dynamic, meta

        X = []
        for i in range(len(feat_df) - self.seq_len + 1):
            X.append(feat_df.iloc[i:i + self.seq_len].values)
        X = np.array(X)

        preds = (self.model.predict(X, verbose=0).flatten() > 0.5)

        # --- ALIGN KE DF ---
        df_dynamic = df_dynamic.copy()
        df_dynamic["direction_anomaly"] = False
        start_idx = 1 + (self.seq_len - 1)
        df_dynamic.iloc[
            start_idx : start_idx + len(preds),
            df_dynamic.columns.get_loc("direction_anomaly")
        ] = preds


        export_cols = [
            "Tanggal",
            "GA_arah",
            "GA_sudut",
            "CNN_arah",
            "CNN_sudut",
            "direction_anomaly"
        ]

        export_df = df_dynamic.reindex(columns=export_cols)
        export_df.to_csv(CSV_PATH, index=False)


        meta["export_path"] = CSV_PATH
        meta["total_predictions"] = int(preds.sum())

        return df_dynamic, meta

# ============================================================
# END OF FILE
# ============================================================
