import logging
import requests
import pandas as pd
import os
from datetime import datetime, timedelta


# ============================================================
#  GLOBAL LOGGER
# ============================================================

logger = logging.getLogger("BMKG_Sensor")
logger.setLevel(logging.INFO)


# ============================================================
#  MAIN CLASS: INA TEWS CLIENT
# ============================================================

class InaTEWSApiClient:
    """
    Sensor Real-time: Mengambil data gempa BMKG terbaru
    + filter hanya Jawa Timur
    """

    def __init__(self):
        self.api_url = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"

        self.jatim_keywords = [
            "JATIM","JAWA TIMUR","SURABAYA","SIDOARJO","GRESIK","LAMONGAN","TUBAN",
            "BOJONEGORO","NGANJUK","MADIUN","MAGETAN","NGAWI","PONOROGO","PACITAN",
            "TRENGGALEK","TULUNGAGUNG","BLITAR","KEDIRI","MALANG","LUMAJANG","JEMBER",
            "BONDOWOSO","SITUBONDO","PROBOLINGGO","PASURUAN","MOJOKERTO","JOMBANG",
            "BANGKALAN","SAMPANG","PAMEKASAN","SUMENEP","MADURA"
        ]

    # ------------------------------------------------------------
    def fetch_recent_data(self) -> pd.DataFrame:
        """Fetch BMKG realtime + filter Jawa Timur"""

        logger.info("Menghubungi Server BMKG (Filter Jawa Timur)...")

        try:
            r = requests.get(self.api_url, timeout=10)
            r.raise_for_status()

            data = r.json()
            infogempa = data.get("Infogempa", {})
            gempa_list = infogempa.get("gempa", [])

            cleaned = []
            jatim_count = 0

            for g in gempa_list:
                wilayah = g["Wilayah"].upper()

                if not any(k in wilayah for k in self.jatim_keywords):
                    continue

                jatim_count += 1

                lat, lon = g["Coordinates"].split(",")
                dt_str = f"{g['Tanggal']} {g['Jam'].split(' ')[0]}"

                cleaned.append({
                    "Tanggal": pd.to_datetime(dt_str, dayfirst=True, errors="coerce"),
                    "Lintang": float(lat),
                    "Bujur": float(lon),
                    "Magnitudo": float(g["Magnitude"]),
                    "Kedalaman_km": float(g["Kedalaman"].replace(" km", "")),
                    "Lokasi": g["Wilayah"],
                    "Sumber": "BMKG_Realtime"
                })

            df = pd.DataFrame(cleaned)

            logger.info(f"BMKG: {jatim_count} event Jatim ditemukan.")
            return df

        except Exception as e:
            logger.error(f"BMKG API ERROR: {e}")
            return pd.DataFrame()


# ============================================================
#  ARCHIVE ENGINE (90 HARI)
# ============================================================

def manage_data_archiving(df: pd.DataFrame, base_dir: str):
    """Menyimpan data >90 hari ke folder archive"""
    archive_dir = os.path.join(base_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    if "Tanggal" not in df.columns:
        return df

    cutoff = datetime.now() - timedelta(days=90)

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    mask_old = df["Tanggal"] < cutoff

    df_old = df[mask_old]
    df_recent = df[~mask_old].copy()

    if not df_old.empty:
        archive_file = os.path.join(archive_dir, "tectonic_history_archive.csv")
        df_old.to_csv(archive_file, mode="a", index=False, header=not os.path.exists(archive_file))

        logger.info(f"Archive: {len(df_old)} record dipindahkan (>90 hari)")

    return df_recent.reset_index(drop=True)


# ============================================================
#  MERGE ENGINE (RAW + BMKG + INJECTION)
# ============================================================

def update_dataset_with_realtime(df_old, path_to_save, df_injection=None):
    """
    Menggabungkan RAW + BMKG + Injection
    dan melakukan archive otomatis (>90 hari)
    """

    # ----------------------
    # BMKG Realtime
    # ----------------------
    try:
        api = InaTEWSApiClient()
        df_new = api.fetch_recent_data()
    except Exception as e:
        logger.error(f"BMKG fetch error: {e}")
        df_new = pd.DataFrame()

    # ----------------------
    # Normalisasi tanggal
    # ----------------------
    for df in [df_old, df_new, df_injection]:
        if df is not None and "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

    # ----------------------
    # Combine dataset
    # ----------------------
    df_list = [d for d in [df_old, df_new, df_injection] if d is not None and not d.empty]
    df_combined = pd.concat(df_list, ignore_index=True)

    df_combined.drop_duplicates(
        subset=["Tanggal", "Lintang", "Bujur"], keep="last", inplace=True
    )
    df_combined.sort_values("Tanggal", inplace=True)

    # ----------------------
    # Save DB utama
    # ----------------------
    try:
        if path_to_save.endswith(".csv"):
            df_combined.to_csv(path_to_save, index=False)
        else:
            df_combined.to_excel(path_to_save, index=False)

        logger.info(f"SAVED: Database realtime → {path_to_save}")

        # Archive >90 hari
        base_dir = os.path.dirname(path_to_save)
        df_clean = manage_data_archiving(df_combined, base_dir)

    except Exception as e:
        logger.error(f"Gagal menyimpan database: {e}")
        df_clean = df_combined

    return df_clean
