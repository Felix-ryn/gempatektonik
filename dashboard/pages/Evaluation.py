import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import json
import os

st.set_page_config(page_title="System Evaluation", layout="wide", page_icon="✅")

# =========================================================
# LOAD METRICS JSON (EVALUATION ENGINE)
# =========================================================
def load_real_metrics():
    """
    Membaca file JSON hasil output dari evaluation_engine.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "..", "output", "system_metrics.json")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"Gagal membaca file metrics: {e}")
            return None
    return None


# =========================================================
# LOAD VALIDASI CNN (CSV)
# =========================================================
def load_cnn_validation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        current_dir,
        "..", "..", "output", "cnn_results",
        "cnn_next_earthquake_prediction.csv"
    )

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            st.error(f"Gagal membaca CSV validasi CNN: {e}")
            return None
    return None


# =========================================================
# MAIN DASHBOARD
# =========================================================
def main():
    st.title("✅ Evaluasi Sistem (Real-Time Metrics)")

    # =====================================================
    # LOAD DATA
    # =====================================================
    data = load_real_metrics()
    df_val = load_cnn_validation()

    # =====================================================
    # INFO SKEMA VALIDASI (CLIENT / DOSEN)
    # =====================================================
    st.markdown("### 🧪 Skema Validasi Model")

    if df_val is not None:
        st.info(
            """
            🔍 **Metode Validasi Arah Gempa**
            - Data latih: **2022 – 2024**
            - Validasi:
              - Prioritas: **Data gempa aktual BMKG 2025**
              - Fallback: **Backtesting historis (2024)**
            - Fokus evaluasi: **arah & sudut pergerakan**, bukan lokasi absolut
            """
        )
    else:
        st.warning("⚠️ Data validasi CNN belum tersedia.")

    st.markdown("---")

    # =====================================================
    # DEFAULT VALUE (FALLBACK)
    # =====================================================
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    labels = ['Rendah', 'Sedang', 'Tinggi']

    if data:
        st.success(f"📌 Data Evaluasi Terakhir: {data.get('timestamp', 'Unknown')}")

        metrics = data.get("metrics", {})
        accuracy = metrics.get("accuracy", 0.0)

        avg_metrics = metrics.get("weighted avg", {})
        precision = avg_metrics.get("precision", 0.0)
        recall = avg_metrics.get("recall", 0.0)

        cm_data = data.get("confusion_matrix", [])
        if cm_data:
            z = cm_data

        if "labels" in data:
            labels = data["labels"]
    else:
        st.warning("⚠️ File system_metrics.json belum ditemukan.")

    x_labels = [f'Prediksi {l}' for l in labels]
    y_labels = [f'Aktual {l}' for l in labels]

    # =====================================================
    # KPI UTAMA
    # =====================================================
    c1, c2, c3 = st.columns(3)
    c1.metric("Akurasi Model", f"{accuracy*100:.1f}%")
    c2.metric("Presisi (Weighted)", f"{precision*100:.1f}%")
    c3.metric("Recall (Weighted)", f"{recall*100:.1f}%")

    st.markdown("---")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("🧩 Confusion Matrix")

        if len(z) == len(x_labels):
            fig = ff.create_annotated_heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale='Viridis',
                showscale=True
            )
            fig.update_layout(title_text="Perbandingan Prediksi vs Aktual")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Dimensi confusion matrix tidak sesuai.")

        st.caption("Data diambil dari hasil training terakhir.")

    with col_r:
        st.subheader("📈 Performa Per Kelas")

        if data and "metrics" in data:
            class_metrics = []
            for k, v in data["metrics"].items():
                if k in labels:
                    class_metrics.append({
                        "Kelas": k,
                        "F1-Score": v.get("f1-score", 0)
                    })

            if class_metrics:
                df_metrics = pd.DataFrame(class_metrics)
                fig2 = px.bar(
                    df_metrics,
                    x='Kelas',
                    y='F1-Score',
                    color='Kelas',
                    range_y=[0, 1],
                    title="F1-Score per Kelas"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Menunggu data klasifikasi per kelas...")
        else:
            dummy = pd.DataFrame({
                "Kelas": labels,
                "Probabilitas": [0.0] * len(labels)
            })
            fig2 = px.bar(dummy, x="Kelas", y="Probabilitas")
            st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # VALIDASI CNN – ARAH & SUDUT
    # =====================================================
    if df_val is not None and not df_val.empty:
        latest = df_val.iloc[-1]

        st.markdown("---")
        st.markdown("### 🧭 Hasil Validasi Prediksi Arah")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Arah Prediksi", latest["arah_prediksi"])
        k2.metric("Sudut Prediksi", f"{latest['arah_derajat']:.1f}°")
        k3.metric("Selisih Sudut", f"{latest['selisih_sudut']:.1f}°")
        k4.metric("Status Validasi", latest["status_validasi"])

        # STATUS RELEVANSI (NON-TEKNIS)
        if latest["status_validasi"] == "RELEVAN":
            st.success(
                f"""
                ✅ **Prediksi RELEVAN**
                Gempa aktual terjadi pada arah **{latest['dir_inferred_from_angle']}**
                dengan selisih sudut **{latest['selisih_sudut']:.1f}°**,
                masih dalam batas toleransi arah wilayah.
                """
            )
        else:
            st.error("❌ Prediksi tidak relevan terhadap kejadian aktual.")

        # GRAFIK SELISIH SUDUT
        st.markdown("### 📐 Tren Selisih Sudut Prediksi vs Aktual")
        st.line_chart(
            df_val.set_index("timestamp")["selisih_sudut"],
            height=300
        )

        # INTERPRETASI OTOMATIS
        st.markdown("### 🧠 Interpretasi Sistem")
        st.info(
            f"""
            Model CNN memprediksi arah gempa dominan ke **{latest['arah_prediksi']}**
            dengan sudut **{latest['arah_derajat']:.1f}°**.
            Validasi menunjukkan selisih sudut **{latest['selisih_sudut']:.1f}°**,
            sehingga prediksi dinilai **{latest['status_validasi']}**
            berdasarkan pendekatan kesesuaian arah wilayah.
            """
        )

        # DETAIL TEKNIS
        with st.expander("📄 Detail Teknis Validasi CNN"):
            st.dataframe(
                df_val[
                    [
                        "timestamp",
                        "arah_prediksi",
                        "arah_derajat",
                        "alt_sampling_1_angle",
                        "selisih_sudut",
                        "confidence_scalar",
                        "akurasi_prediksi_persen",
                        "validasi_note"
                    ]
                ],
                use_container_width=True
            )


if __name__ == "__main__":
    main()
