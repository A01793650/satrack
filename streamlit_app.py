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
from rtree import index

st.set_page_config(layout="wide")

st.title("Visor de Shapefile en HTML")

# Ruta al rar (puede ser local o URL RAW en GitHub)
rar_path = 'https://github.com/A01793650/satrack/blob/main/shp.rar'

with tempfile.TemporaryDirectory() as tmpdir:
    rf = rarfile.RarFile(rar_path)
    rf.extractall(tmpdir)

    m = folium.Map(location=[0,0], zoom_start=2, tiles="OpenStreetMap")

    for file in os.listdir(tmpdir):
        if file.endswith(".shp"):
            gdf = gpd.read_file(os.path.join(tmpdir, file))
            folium.GeoJson(gdf, name=file).add_to(m)

    folium.LayerControl().add_to(m)
    m.save("mapa.html")
