"""
Evaluation Engine – SAFE EDITION (v3.1) - TECTONIC
Anti-crash, auto-sync, fault-tolerant, and stable under any data condition.

FIXED:
- Phase 8: Added explicit 'labels' parameter to classification_report and 
  confusion_matrix to handle scenarios where 'y_test' contains only one class 
  (e.g., 'RINGAN'), preventing the "Number of classes, 1, does not match size of 
  target_names, 3" error.
"""

import json
import pandas as pd
import numpy as np
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class EvaluationEngine:

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("Grand_Jury_Titan_SAFE")
        self.logger.setLevel(logging.INFO)

        # --- ENSEMBLE BASE MODELS (Fungsi Penting) ---
        self.nb = CalibratedClassifierCV(GaussianNB(), method='sigmoid')

        self.rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=42
        )

        self.gb = GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        self.ensemble = VotingClassifier(
            estimators=[("nb", self.nb), ("rf", self.rf), ("gb", self.gb)],
            voting="soft",
            weights=[1, 2, 1]
        )

        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output"))
        self.paths = {
            "model": os.path.join(base_path, "evaluation_model.pkl"),
            "metrics": os.path.join(base_path, "system_metrics.json"),
            "log": os.path.join(base_path, "decision_history.csv")
        }

        # Fix label order
        self.fixed_labels = ["RINGAN", "SEDANG", "PARAH"]
        self.encoder.fit(self.fixed_labels) # Menggunakan encoder untuk mapping 0, 1, 2

    # ---------------------------------------------------------
    # SAFETY UTILITIES (Fungsi Penting)
    # ---------------------------------------------------------

    def _safe_column(self, df: pd.DataFrame, col: str, default=0.0):
        """If column missing → create one safely."""
        if col not in df.columns:
            df[col] = default
        # Suppress Future Warnings for inplace operations (Python 3.11+)
        # df[col].replace([np.inf, -np.inf, None], default, inplace=True) 
        df[col] = df[col].replace([np.inf, -np.inf, None], default) # Ganti ke non-inplace
        df[col] = df[col].fillna(default)
        return df[col]

    def _sync_length(self, A, B):
        """Return two arrays with equal length by trimming to min."""
        L = min(len(A), len(B))
        return A[:L], B[:L]

    def _inject_synthetic_samples(self, X_train, y_train_encoded):
        """
        Fungsi Penting: Mengatasi Single-Class Crash pada Training.
        """
        unique = np.unique(y_train_encoded)
        if len(unique) >= 2:
            return X_train, y_train_encoded

        rare_class = 2  # PARAH (Label 2)
        synthetic_X = np.mean(X_train, axis=0, keepdims=True)
        synthetic_X = np.tile(synthetic_X, (5, 1))
        synthetic_y = np.full(5, rare_class)

        X_new = np.vstack([X_train, synthetic_X])
        y_new = np.concatenate([y_train_encoded, synthetic_y])

        self.logger.warning("Injected synthetic samples to avoid single-class crash.")

        return X_new, y_new

    # ---------------------------------------------------------
    # FEATURE GATHERING (Fungsi Penting)
    # ---------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required features always exist, prioritizing original magnitude/depth."""

        required = [
            "Magnitudo", "Kedalaman_km", # Diganti di baris berikutnya
            "rolling_mag_mean", "rolling_event_count",
            "Pheromone_Score",
            "Temporal_Risk_Factor",
            "Proporsi_Grid_Terdampak_CNN"
        ]

        # ✅ FIX KRITIS: Prioritaskan kolom Original (Data ASLI/Mentah)
        self.logger.debug("Memprioritaskan Magnitudo/Kedalaman_Original.")
        df['Magnitudo_Feat'] = self._safe_column(df, 'Magnitudo_Original', 0.0)
        df['Kedalaman_km_Feat'] = self._safe_column(df, 'Kedalaman_Original', 999.0)

        # Ganti required columns dengan versi Feat (yang sudah ter-safe_column)
        required_feat = [
             "Magnitudo_Feat", "Kedalaman_km_Feat", 
             "rolling_mag_mean", "rolling_event_count",
             "Pheromone_Score",
             "Temporal_Risk_Factor",
             "Proporsi_Grid_Terdampak_CNN"
        ]

        for col in required_feat:
            self._safe_column(df, col, 0.0)

        X = df[required_feat].copy()
        X = X.fillna(0.0)

        return X

    # ---------------------------------------------------------
    # RISK LABELING (GROUND TRUTH) (Fungsi Penting)
    # ---------------------------------------------------------

    def _label_rule_based(self, row):
        """Physics-informed label (Ground Truth). TIDAK BOLEH menggunakan hasil model lain."""
        
        # ✅ FIX KRITIS: Hapus Ketergantungan pada Pheromone_Score (Data Leakage Fix)
        M = float(row.get("Magnitudo_Original", row.get("Magnitudo", 0)))
        D = float(row.get("Kedalaman_Original", row.get("Kedalaman_km", 999)))

        # Hapus M_eff = M + (0.4 if P > 0.7 else 0)
        M_eff = M 

        # Logika pelabelan berbasis Magnitudo dan Kedalaman mentah (Tanpa Pheromone)
        if D <= 60:
            if M_eff >= 6.0: return "PARAH"
            if M_eff >= 4.5: return "SEDANG"
            return "RINGAN"

        if D <= 300:
            if M_eff >= 6.5: return "PARAH"
            if M_eff >= 5.5: return "SEDANG"
            return "RINGAN"

        return "RINGAN"

    # ---------------------------------------------------------
    # MAIN EXECUTION (Fungsi Penting)
    # ---------------------------------------------------------

    def run(self, df: pd.DataFrame, train_idx, test_idx):

        self.logger.info("=== SAFE EVALUATION STARTED ===")

        # ------------------------------
        # SANITY: Reset index sehingga .iloc/.loc numerik konsisten
        # ------------------------------
        try:
            df = df.reset_index(drop=True)
        except Exception:
            # Kalau gagal (sangat jarang), tetap lanjut tapi log exception
            self.logger.exception("Gagal melakukan reset_index pada df — melanjutkan dengan index asli.")

        # Pastikan train_idx / test_idx menjadi numpy int arrays dan valid
        try:
            train_idx = np.array(train_idx, dtype=int) if train_idx is not None else np.array([], dtype=int)
        except Exception:
            train_idx = np.array([], dtype=int)

        try:
            test_idx = np.array(test_idx, dtype=int) if test_idx is not None else np.array([], dtype=int)
        except Exception:
            test_idx = np.array([], dtype=int)

        # Filter indices yang berada di rentang valid [0, len(df)-1]
        if len(df) > 0:
            train_idx = train_idx[(train_idx >= 0) & (train_idx < len(df))]
            test_idx = test_idx[(test_idx >= 0) & (test_idx < len(df))]
        else:
            train_idx = np.array([], dtype=int)
            test_idx = np.array([], dtype=int)

        # ------------------------------
        # 1-3. DATA PREP & SPLIT (SAFE)
        # ------------------------------
        labels = df.apply(self._label_rule_based, axis=1)
        y_encoded = self.encoder.transform(labels)
        X_raw = self._extract_features(df)

        # Gunakan iloc (posisi) — lebih eksplisit dan aman
        X_train = X_raw.iloc[train_idx].values if len(train_idx) > 0 else np.empty((0, X_raw.shape[1]))
        X_test = X_raw.iloc[test_idx].values if len(test_idx) > 0 else np.empty((0, X_raw.shape[1]))
        y_train = y_encoded[train_idx] if len(train_idx) > 0 else np.array([], dtype=int)
        y_test = y_encoded[test_idx] if len(test_idx) > 0 else np.array([], dtype=int)

        # ------------------------------
        # 4. STANDARDIZATION & SYNTHETIC FIX (TRAIN PATH)
        # ------------------------------
        if len(X_train) == 0:
            self.logger.critical("[SAFE EVAL] Training data kosong. Gagal melatih model.")
            df["Final_Prob_Impact"] = 0.0
            return df

        # Standarisasi fitur (FIT HANYA DI TRAIN)
        try:
            self.scaler.fit(X_train)
            X_train_std = self.scaler.transform(X_train)
        except Exception as e:
            self.logger.exception(f"Scaler fit/transform gagal: {e}")
            X_train_std = X_train.copy()

        # Inject synthetic samples jika single-class
        X_train_std, y_train = self._inject_synthetic_samples(X_train_std, y_train)

        # ------------------------------
        # 5. ROBUST MODEL TRAINING (Menciptakan successful_classifier)
        # ------------------------------
        successful_classifier = None
        is_single_sample_train = len(X_train) == 1 
    
        self.logger.info(f"Mencoba melatih Ensemble ({len(X_train_std)} sampel)...")
    
        # --- Coba Latih Ensemble/VotingClassifier ---
        try:
            if is_single_sample_train:
                # Bypass ensemble untuk data tunggal/sangat sedikit agar tidak crash Naive Bayes.
                raise ValueError("Bypass ensemble for single sample to enforce stable fallback.")
            
            self.ensemble.fit(X_train_std, y_train)
            successful_classifier = self.ensemble
            self.logger.info("Ensemble Training BERHASIL.")
        
        except Exception as e:
            self.logger.error(f"[MODEL ERROR] Ensemble/NaiveBayes GAGAL: {e}. Menggunakan FALLBACK: RandomForest.")
        
            # --- FALLBACK ke RandomForest Saja ---
            try:
                self.rf.fit(X_train_std, y_train)
            
                # Kalibrasikan RF untuk predict_proba jika ada lebih dari 1 kelas unik
                if len(np.unique(y_train)) > 1:
                    from sklearn.calibration import CalibratedClassifierCV
                    # Tentukan CV yang aman
                    cv_folds = min(len(X_train_std) // 2, 5)
                    calib_rf = CalibratedClassifierCV(self.rf, method='sigmoid', cv=max(2, cv_folds))
                    calib_rf.fit(X_train_std, y_train)
                    successful_classifier = calib_rf
                else:
                    successful_classifier = self.rf
                 
                self.logger.info("RandomForest Fallback Training BERHASIL.")
            except Exception as rf_e:
                self.logger.critical(f"[MODEL ERROR] RandomForest GAGAL juga: {rf_e}. Gagal Melatih.")
                return df # Jika RF gagal, siklus dihentikan.

        # Gunakan classifier yang berhasil (ensemble/rf) untuk save.
        self.ensemble = successful_classifier 
    
    
        # ------------------------------
        # 6. PENANGANAN DATA TEST KOSONG (REAL-TIME TUNGGAL)
        # ------------------------------
        if len(X_test) == 0:
            self.logger.warning("[SAFE EVAL] Test set kosong. Melewati Inference/Metrics. Menyimpan model...")
            df["Final_Prob_Impact"] = 0.0 # Pastikan default 0.0
        
            # --- Simpan Model (Langkah 9 - dipindahkan ke sini) ---
            try:
                 with open(self.paths["model"], "wb") as f:
                     pickle.dump({
                         "model": self.ensemble,
                         "scaler": self.scaler,
                         "encoder": self.encoder
                     }, f)
                 self.logger.info("Model FALLBACK BERHASIL disimpan.")
            except Exception as e:
                self.logger.error(f"MODEL SAVE FAILED: {e}")
            
            return df
    
        # ------------------------------
        # 7. INFERENCE (JALUR NORMAL: len(X_test) > 0)
        # ------------------------------
    
        # Transform X_test
        X_test_std = self.scaler.transform(X_test)

        # Prediksi menggunakan classifier yang berhasil
        y_pred = self.ensemble.predict(X_test_std)

        # Probabilitas (digunakan untuk Final_Prob_Impact)
        try:
            prob = self.ensemble.predict_proba(X_test_std)
        except:
            self.logger.warning("Predict_proba gagal, menggunakan probabilitas default 0.")
            prob = np.zeros((len(y_test), 3))

        # Extract only PARAH probability safely
        try:
            class_index = list(self.encoder.classes_).index("PARAH")
            prob_parah = prob[:, class_index]
        except:
            prob_parah = np.zeros(len(prob))

        df["Final_Prob_Impact"] = 0.0

        # SAFE assignment by POSITION (bukan label index)
        if len(test_idx) > 0:
            df.iloc[test_idx, df.columns.get_loc("Final_Prob_Impact")] = prob_parah
        # ------------------------------
        # 8. METRICS + LOGGING
        # ------------------------------
    
        # Labels numerik yang harus dicari (0, 1, 2)
        all_possible_numeric_labels = list(self.encoder.transform(self.fixed_labels)) 
    
        # Perbaiki y_test dan y_pred, pastikan hanya data valid yang digunakan
        y_test_clean = y_test[np.isfinite(y_test)]
        y_pred_clean = y_pred[:len(y_test_clean)]

        try:
            metrics = classification_report(
                y_test_clean, y_pred_clean,
                labels=all_possible_numeric_labels, 
                target_names=self.fixed_labels,
                output_dict=True,
                zero_division=0
            )
        except Exception as e:
            self.logger.error(f"Classification Report Gagal: {e}")
            metrics = {}
        
        try:
            cm = confusion_matrix(
                y_test_clean, y_pred_clean,
                labels=all_possible_numeric_labels
            ).tolist()
        except Exception as e:
            self.logger.error(f"Confusion Matrix Gagal: {e}")
            cm = [[0,0,0], [0,0,0], [0,0,0]] 

        data_out = {
            "timestamp": str(datetime.now()),
            "labels": self.fixed_labels,
            "metrics": metrics,
            "confusion_matrix": cm
        }

        with open(self.paths["metrics"], "w") as f:
            json.dump(data_out, f, indent=4)

        # ------------------------------
        # 9. SAVE MODEL (JALUR NORMAL)
        # ------------------------------
        try:
             with open(self.paths["model"], "wb") as f:
                 pickle.dump({
                     "model": self.ensemble,
                     "scaler": self.scaler,
                     "encoder": self.encoder
                 }, f)
        except Exception as e:
            self.logger.error(f"MODEL SAVE FAILED: {e}")

        return df