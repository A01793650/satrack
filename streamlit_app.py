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

st.title("Visor de GeoJSON en Streamlit 🌍")

# Subir archivo GeoJSON
uploaded_file = st.file_uploader("Sube tu archivo GeoJSON", type=["geojson", "json"])

if uploaded_file is not None:
    # Leer GeoJSON con GeoPandas
    gdf = gpd.read_file(uploaded_file)

    # Calcular el centro de la geometría
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]

    # Crear mapa
    m = folium.Map(location=center, zoom_start=10)

    # Agregar capa GeoJSON
    folium.GeoJson(
        gdf,
        name="Capa GeoJSON",
        tooltip=folium.GeoJsonTooltip(fields=gdf.columns, aliases=gdf.columns.tolist())
    ).add_to(m)

    # Agregar control de capas
    folium.LayerControl().add_to(m)

    # Mostrar mapa en Streamlit
    st_folium(m, width=800, height=600)
    folium.LayerControl().add_to(m)

    # Mostrar mapa
    st_folium(m, width=800, height=600)
