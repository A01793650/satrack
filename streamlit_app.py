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

# Lista de GeoJSON desde GitHub (raw links)
geojson_urls = [
    "https://raw.githubusercontent.com/tu_usuario/tu_repo/main/archivo1.geojson",
    "https://raw.githubusercontent.com/tu_usuario/tu_repo/main/archivo2.geojson"
]

# Crear mapa base
m = folium.Map(location=[0, 0], zoom_start=2)

# Cargar cada GeoJSON
for url in geojson_urls:
    gdf = gpd.read_file(url)
    
    # Calcular centro aproximado (última capa cargada)
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m.location = center
    
    # Agregar capa con tooltip si hay atributos
    atributos = [col for col in gdf.columns if col != "geometry"]
    if atributos:
        folium.GeoJson(
            gdf,
            name=url.split("/")[-1],  # nombre del archivo
            tooltip=folium.GeoJsonTooltip(fields=atributos, aliases=atributos)
        ).add_to(m)
    else:
        folium.GeoJson(gdf, name=url.split("/")[-1]).add_to(m)

# Control de capas
folium.LayerControl().add_to(m)

# Exportar a HTML
m.save("mapa_geojson.html")
print("✅ Mapa exportado como mapa_geojson.html")
