import streamlit as st
import pandas as pd
import folium
import os
import geopandas as gpd

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

# Ruta al zip (puede ser local o URL RAW en GitHub)
zip_path = "mi_shapes.zip"  
# Si está en GitHub: "https://github.com/usuario/repositorio/raw/main/mi_shapes.zip"

# Crear carpeta temporal y extraer ZIP
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir)

    # Crear mapa base
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

    # Recorrer todos los .shp dentro del zip
    for file in os.listdir(tmpdir):
        if file.endswith(".shp"):
            shp_path = os.path.join(tmpdir, file)
            gdf = gpd.read_file(shp_path)

            # Calcular centro aproximado para el primer shape
            if len(gdf) > 0 and m.location == [0, 0]:
                m.location = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
                m.zoom_start = 6

            # Agregar cada shapefile como capa
            folium.GeoJson(gdf, name=file).add_to(m)

    # Agregar control de capas
    folium.LayerControl().add_to(m)

    # Guardar mapa
    m.save("index.html")
