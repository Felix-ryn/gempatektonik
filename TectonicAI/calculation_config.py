import numpy as np
import math

# --- DEFINISI VARIABEL GLOBAL (Wajib Ada) ---
CALCULATION_CONFIG = {
    # === 1. FAKTOR SCALING (PEMBOBOTAN) ===
    "scaling": {
        "magnitude_scaling": 2.0,
        "depth_scaling": 0.5
    },

    # === 2. FORMULA METRIK ===
    "metrics": {
        # A. Ground Truth Radius
        "R_true_formula": "10**(0.5 * (MagScaled / 2.0) - 1.8)",

        # B. Heuristik Semut (ACO)
        "aco_heuristic_formula": "(MagScaled**2) / (DepthScaled + 1.0)",

        # C. Kepadatan Kejadian (LSTM)
        "event_density_formula": "rolling_event_count / (rolling_mag_mean + 1e-6)"
    },

    # === 3. KONFIGURASI LAIN ===
    "clustering": {
        "method": "kmeans",
        "n_clusters": 5,
        "random_state": 42
    },

    "noise": {
        "mean": 0.0,
        "std": 0.02
    },
    
    "flags": {
        "save_intermediate": True,
        "log_metrics": True
    }
}

def eval_formula(formula: str, variables: dict):
    """Eksekusi aman string rumus matematika."""
    context = {"np": np, "math": math}
    context.update(variables)
    try:
        return eval(formula, context)
    except Exception:
        return 0.0