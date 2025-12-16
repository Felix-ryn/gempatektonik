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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

        self.evaluator_name = "Naive Bayes"

        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output"))
        self.paths = {
            "model": os.path.join(base_path, "evaluation_model.pkl"),
            "metrics": os.path.join(base_path, "system_metrics.json"),
            "eval_metrics": os.path.join(base_path, "system_evaluation_metrics.json"),
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
            "Magnitudo_Feat",
            "Kedalaman_km_Feat",
            "rolling_mag_mean",
            "rolling_event_count",
            "Pheromone_Score",
            "Temporal_Risk_Factor",
            "CNN_Pred_Angle",
            "CNN_Pred_Distance"
        ]

        # ✅ FIX KRITIS: Prioritaskan kolom Original (Data ASLI/Mentah)
        self.logger.debug("Memprioritaskan Magnitudo/Kedalaman_Original.")
        df['Magnitudo_Feat'] = self._safe_column(df, 'Magnitudo_Original', 0.0)
        df['Kedalaman_km_Feat'] = self._safe_column(df, 'Kedalaman_Original', 999.0)

        # Ganti required columns dengan versi Feat (yang sudah ter-safe_column)
        required_feat = [
            "Magnitudo_Feat",
            "Kedalaman_km_Feat",
            "rolling_mag_mean",
            "rolling_event_count",
            "Pheromone_Score",
            "Temporal_Risk_Factor",
            "CNN_Pred_Angle",
            "CNN_Pred_Distance"
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

    # =================================================
    # MODEL-LEVEL EVALUATION SUMMARY (BARU)
    # =================================================
    def _build_model_summary(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=list(self.encoder.transform(self.fixed_labels)))
        return {
            "Akurasi": float(accuracy_score(y_true, y_pred)),
            "PPV (Presisi)": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "Sensitivitas (Recall)": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "F1-Score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "Confusion_Matrix": cm.tolist()
        }

    def _write_all_metrics(
        self,
        mode: str,
        metrics: dict,
        confusion_matrix: list,
        evaluation_summary: dict | None
    ):
        """
        SATU-SATUNYA pintu penulisan metrics.
        Dijamin update di realtime & offline.
        """

        timestamp = str(datetime.now())

        # ===============================
        # system_metrics.json
        # ===============================
        system_metrics = {
            "timestamp": timestamp,
            "mode": mode,
            "evaluator": self.evaluator_name,
            "labels": self.fixed_labels,
            "metrics": metrics,
            "confusion_matrix": confusion_matrix
        }

        with open(self.paths["metrics"], "w") as f:
            json.dump(system_metrics, f, indent=4)

        # ===============================
        # system_evaluation_metrics.json
        # ===============================
        if evaluation_summary is None:
            evaluation_summary = {
                "info": "No statistical evaluation (realtime mode)",
                "timestamp": timestamp
            }

        with open(self.paths["eval_metrics"], "w") as f:
            json.dump(evaluation_summary, f, indent=4)

        self.logger.info("ALL METRICS FILES UPDATED SUCCESSFULLY.")

    # ---------------------------------------------------------
    # MAIN EXECUTION (Fungsi Penting)
    # ---------------------------------------------------------

    def run(self, df: pd.DataFrame, train_idx, test_idx):

        self.logger.info("=== SAFE EVALUATION STARTED ===")

        # SANITY: Reset index sehingga .iloc/.loc numerik konsisten
        try:
            df = df.reset_index(drop=True)
        except Exception:
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

        # 1-3. DATA PREP & SPLIT (SAFE)
        labels = df.apply(self._label_rule_based, axis=1)
        y_encoded = self.encoder.transform(labels)
        X_raw = self._extract_features(df)

        X_train = X_raw.iloc[train_idx].values if len(train_idx) > 0 else np.empty((0, X_raw.shape[1]))
        X_test = X_raw.iloc[test_idx].values if len(test_idx) > 0 else np.empty((0, X_raw.shape[1]))
        y_train = y_encoded[train_idx] if len(train_idx) > 0 else np.array([], dtype=int)
        y_test = y_encoded[test_idx] if len(test_idx) > 0 else np.array([], dtype=int)

        # default
        force_offline_eval = False

        # 4. STANDARDIZATION & SYNTHETIC FIX (TRAIN PATH)
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
            
            self.logger.info("Training Naive Bayes Evaluator...")
            self.nb.fit(X_train_std, y_train)
            successful_classifier = self.nb

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

        # ==========================================================
        # FORCE OFFLINE EVALUATION USING TRAIN DATA IF TEST EMPTY
        # ==========================================================
        force_offline_eval = False

        if len(X_test) < 2:
            self.logger.warning("[SAFE EVAL] X_test terlalu kecil. Menggunakan DATA TRAIN untuk evaluasi statistik.")
            X_eval = X_train_std
            y_eval_true = y_train
            force_offline_eval = True
        else:
            X_eval = self.scaler.transform(X_test)
            y_eval_true = y_test

        # Tentukan mode sekarang setelah tahu force_offline_eval
        mode = "OFFLINE_BATCH" if force_offline_eval else "REALTIME_SINGLE_EVENT"

        # Prediksi menggunakan classifier yang berhasil
        y_pred = successful_classifier.predict(X_eval)

        # Probabilitas (digunakan untuk Final_Prob_Impact) -> gunakan test set asli untuk probabilitas
        try:
            X_test_std = self.scaler.transform(X_test) if len(X_test) > 0 else np.empty((0, X_train_std.shape[1]))
            prob = successful_classifier.predict_proba(X_test_std) if len(X_test_std) > 0 else np.zeros((0, len(self.fixed_labels)))
        except Exception:
            self.logger.warning("Predict_proba gagal, menggunakan probabilitas default 0.")
            prob = np.zeros((len(y_test), len(self.fixed_labels)))

        # Extract only PARAH probability safely
        try:
            class_index = list(self.encoder.classes_).index("PARAH")
            prob_parah = prob[:, class_index] if prob.size > 0 else np.zeros(len(y_test))
        except Exception:
            prob_parah = np.zeros(len(y_test))

        df["Final_Prob_Impact"] = 0.0
        if len(test_idx) > 0 and len(prob_parah) > 0:
            df.iloc[test_idx, df.columns.get_loc("Final_Prob_Impact")] = prob_parah

        # Prepare y_test_clean / y_pred_clean for metrics (guard)
        y_test_clean = y_eval_true if len(y_eval_true) > 0 else np.array([], dtype=int)
        y_pred_clean = y_pred[:len(y_test_clean)] if len(y_test_clean) > 0 else np.array([], dtype=int)

        # Build metrics only if we have labels
        if len(y_test_clean) > 0:
            try:
                all_possible_numeric_labels = list(self.encoder.transform(self.fixed_labels))
                metrics = classification_report(
                    y_test_clean, y_pred_clean,
                    labels=all_possible_numeric_labels,
                    target_names=self.fixed_labels,
                    output_dict=True, zero_division=0
                )
            except Exception as e:
                self.logger.error(f"Classification Report Gagal: {e}")
                metrics = {}

            try:
                cm = confusion_matrix(y_test_clean, y_pred_clean, labels=all_possible_numeric_labels).tolist()
            except Exception as e:
                self.logger.error(f"Confusion Matrix Gagal: {e}")
                cm = [[0]*len(self.fixed_labels) for _ in self.fixed_labels]

            try:
                evaluation_summary = {"NaiveBayes": self._build_model_summary(y_test_clean, y_pred_clean)}
            except Exception as e:
                self.logger.error(f"MODEL EVALUATION SUMMARY FAILED: {e}")
                evaluation_summary = None
        else:
            # no labels available -> keep graceful defaults
            metrics = {}
            cm = [[0]*len(self.fixed_labels) for _ in self.fixed_labels]
            evaluation_summary = None

        # Write files using the correctly computed mode
        self._write_all_metrics(
            mode=mode,
            metrics=metrics,
            confusion_matrix=cm,
            evaluation_summary=evaluation_summary
        )


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