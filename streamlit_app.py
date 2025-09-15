import streamlit as st
import pandas as pd
import folium
import os
import geopandas as gpd
import rarfile
import tempfile

from folium.plugins import MarkerCluster
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import IsolationForest
from shapely.geometry import Point
from shapely import wkt
from streamlit_folium import st_folium
from rtree import index

st.title("AAAVisor de múltiples KML en Streamlit 🌍")

# Subida de múltiples archivos KML
uploaded_files = st.file_uploader("Sube tus archivos KML", type=["kml"], accept_multiple_files=True)

if uploaded_files:
    m = folium.Map(location=[0, 0], zoom_start=2)  # mapa base
    
    for uploaded_file in uploaded_files:
        try:
            # Leer cada archivo KML
            gdf = gpd.read_file(uploaded_file, driver="KML")
            
            # Calcular centro aproximado
            center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
            m.location = center  # ajustar centro en la última capa cargada

            # Agregar al mapa
            folium.GeoJson(
                gdf,
                name=uploaded_file.name,
                tooltip=folium.GeoJsonTooltip(fields=gdf.columns, aliases=gdf.columns.tolist())
            ).add_to(m)
        
        except Exception as e:
            st.error(f"No se pudo leer {uploaded_file.name}: {e}")
    
    # Control de capas
    folium.LayerControl().add_to(m)

    # Mostrar mapa
    st_folium(m, width=800, height=600)
