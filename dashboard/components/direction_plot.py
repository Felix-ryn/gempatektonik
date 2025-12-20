# -*- coding: utf-8 -*-

# components/direction_plot.py
"""
Direction plot component.

Visualizes predicted directions/angles as a polar histogram or a single vector marker.
Designed to accept either:
- A DataFrame with angle column (degrees), or
- A single direction + angle pair.

Functions
---------
render_direction_distribution(...)
render_direction_single(...)
"""
from typing import Optional, Iterable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def _resolve_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def render_direction_distribution(
    df: pd.DataFrame,
    angle_col_candidates=('Pred_Angle', 'Angle', 'GA_Angle_Pred', 'GA_Angle', 'Arah_Angle'),
    bins: int = 18,
    title: Optional[str] = "Direction / Angle Distribution (degrees)"
) -> None:
    """
    Render a polar histogram of direction/angle values in degrees.

    Parameters
    ----------
    df : pd.DataFrame
    angle_col_candidates : tuple
        candidate column names for angle in degrees (0-360)
    bins : int
        number of bins for polar histogram
    """
    if df is None or df.shape[0] == 0:
        st.info("Tidak ada data untuk memvisualisasikan arah.")
        return

    angle_col = _resolve_column(df, angle_col_candidates)
    if angle_col is None:
        st.error("Kolom angle tidak ditemukan. Contoh nama: 'Pred_Angle' atau 'GA_Angle_Pred'.")
        return

    angles = df[angle_col].dropna().astype(float).values
    if len(angles) == 0:
        st.info("Tidak ada nilai sudut untuk divisualisasikan.")
        return

    # convert degrees to radians
    theta = np.deg2rad(angles % 360)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    counts, bin_edges = np.histogram(theta, bins=bins)
    widths = bin_edges[1:] - bin_edges[:-1]
    ax.bar(bin_edges[:-1], counts, width=widths, bottom=0.0, align='edge', alpha=0.7)
    ax.set_title(title)
    st.pyplot(fig)


def render_direction_single(
    direction_text: Optional[str] = None,
    angle_deg: Optional[float] = None,
    title: Optional[str] = "Predicted Direction & Angle"
) -> None:
    """
    Render a single-vector indicator in a small polar chart and show textual label.

    Parameters
    ----------
    direction_text : Optional[str]
        human-readable direction (e.g., "Timur Laut")
    angle_deg : Optional[float]
        angle in degrees (0 = North, clockwise OR define convention)
    """
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")  # 0 deg at top (North)
    ax.set_theta_direction(-1)  # clockwise positive
    ax.set_ylim(0, 1)

    if angle_deg is not None:
        theta = np.deg2rad(angle_deg % 360)
        ax.arrow(theta, 0, 0, 0.8, width=0.03, length_includes_head=True, head_width=0.08, color='C0')
        
        # PERBAIKAN DI SINI: Menggunakan \u00B0 menggantikan simbol derajat literal
        label = f"{direction_text or ''} ({angle_deg:.1f}\u00B0)"
    else:
        label = direction_text or "No prediction available"

    ax.set_title(title)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_yticklabels([])

    st.pyplot(fig)
    st.write("**Interpretation:**", label)