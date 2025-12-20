# dashboard/components/map_view.py
import pandas as pd
import pydeck as pdk
import streamlit as st

def render_map(
    df: pd.DataFrame,
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    pher_col: str = 'Risk_Pheromone',
    rad_col: str = 'Risk_Radius_KM',
    map_height: int = 500
):
    """
    Render map menggunakan PyDeck dengan pewarnaan dinamis berbasis Python.
    """
    if df is None or df.empty:
        st.info("Data spatial tidak tersedia.")
        return

    # Validasi Kolom
    required = [lat_col, lon_col]
    if not all(col in df.columns for col in required):
        st.warning(f"Kolom koordinat ({lat_col}, {lon_col}) tidak ditemukan.")
        return

    # Prepare Data Copy
    data = df.copy()
    data = data.dropna(subset=[lat_col, lon_col])

    # Normalisasi Pheromone untuk Warna (0-255)
    # Merah makin pekat jika pheromone tinggi
    max_pher = data[pher_col].max() if pher_col in data.columns else 1
    if max_pher == 0: max_pher = 1
    
    def get_color(val):
        if pher_col not in data.columns: return [80, 180, 250, 160] # Biru default
        intensity = int((val / max_pher) * 255)
        return [intensity, 50, 50, 160] # Gradasi Merah

    data['color'] = data[pher_col].apply(get_color) if pher_col in data.columns else [[80, 180, 250, 160]] * len(data)
    data['radius_meter'] = data[rad_col] * 1000 if rad_col in data.columns else 1000

    # View State
    mid_lat = data[lat_col].mean()
    mid_lon = data[lon_col].mean()
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=5, pitch=0)

    # Layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=[lon_col, lat_col],
        get_fill_color='color',
        get_radius='radius_meter',
        radius_min_pixels=3,
        radius_max_pixels=50,
        pickable=True
    )

    # Tooltip
    tooltip = {
        "html": f"<b>Lat:</b> {{{lat_col}}} <br/><b>Lon:</b> {{{lon_col}}} <br/><b>Risk:</b> {{{pher_col}}}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    ))