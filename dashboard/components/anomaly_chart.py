from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def _resolve_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def render_anomaly_chart(
    df: pd.DataFrame,
    date_col_candidates=('Tanggal', 'date', 'Date', 'timestamp'),
    risk_col_candidates=('Temporal_Risk_Factor', 'Risk', 'temporal_risk'),
    anomaly_col_candidates=('is_Anomaly', 'anomaly', 'Is_Anomaly'),
    relation_col_candidates=('ACO_GA_Relation_Score', 'Relation_Score', 'aco_ga_score'),
    figsize=(12, 4),
    title: Optional[str] = "Temporal Risk & Anomalies (LSTM)"
) -> None:
    """
    Render a time-series chart showing temporal risk with anomalies highlighted.
    If relation score column exists it will be plotted on a secondary axis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series.
    date_col_candidates : tuple
        Candidate names for date column (will pick first match).
    risk_col_candidates : tuple
        Candidate names for risk column.
    anomaly_col_candidates : tuple
        Candidate names for anomaly flag column (boolean / 0-1).
    relation_col_candidates : tuple
        Candidate names for relation score to overlay.
    figsize : tuple
        Matplotlib figure size.
    title : str
        Plot title.
    """
    if df is None or df.shape[0] == 0:
        st.info("Tidak ada data untuk ditampilkan pada anomaly chart.")
        return

    date_col = _resolve_column(df, date_col_candidates)
    risk_col = _resolve_column(df, risk_col_candidates)
    anomaly_col = _resolve_column(df, anomaly_col_candidates)
    relation_col = _resolve_column(df, relation_col_candidates)

    if date_col is None:
        st.error("Kolom tanggal tidak ditemukan. Harap sertakan kolom tanggal (e.g. 'Tanggal').")
        return

    if risk_col is None:
        st.error("Kolom risiko temporal tidak ditemukan. Harap sertakan kolom 'Temporal_Risk_Factor'.")
        return

    # Ensure datetime
    series = df.copy()
    series[date_col] = pd.to_datetime(series[date_col], errors='coerce')
    series = series.sort_values(date_col).reset_index(drop=True)

    x = series[date_col]
    y = series[risk_col].astype(float).fillna(0.0)

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, label="Temporal Risk", linewidth=1.5, alpha=0.9)

    # highlight anomalies
    if anomaly_col is not None:
        mask = series[anomaly_col].astype(bool)
        if mask.any():
            ax.scatter(x[mask], y[mask], s=40, marker='o', label='Anomaly', zorder=5, color='red')
    else:
        # PERBAIKAN DI SINI: Menggunakan tanda strip biasa (-)
        st.caption("Kolom anomaly tidak ditemukan - hanya menampilkan risk trend.")

    # overlay relation score on secondary axis if available
    if relation_col is not None:
        ax2 = ax.twinx()
        rel = series[relation_col].astype(float).fillna(0.0)
        ax2.plot(x, rel, linestyle='--', linewidth=1, alpha=0.9, label='Relation Score', color='orange')
        ax2.set_ylabel("Relation Score")
        # combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax.legend(loc='upper left')

    ax.set_title(title)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Temporal Risk")
    ax.grid(alpha=0.2)

    st.pyplot(fig)