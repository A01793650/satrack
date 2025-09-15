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
    "https://raw.githubusercontent.com/A01793650/satrack/main/break.geojson",
    "https://raw.githubusercontent.com/A01793650/satrack/main/bus.geojson",
    "https://raw.githubusercontent.com/A01793650/satrack/main/conn_line.geojson",
    "https://raw.githubusercontent.com/A01793650/satrack/main/diststat.geojson",
    "https://raw.githubusercontent.com/A01793650/satrack/main/line_cable.geojson",
    "https://raw.githubusercontent.com/A01793650/satrack/main/line_jk.geojson",
    "https://raw.githubusercontent.com/A01793650/satrack/main/station.geojson"         
]
st.title("Visor de GeoJSON desde GitHub 🌍")

# Crear mapa base
m = folium.Map(location=[0, 0], zoom_start=2)

for url in geojson_urls:
    try:
        gdf = gpd.read_file(url)

        # Calcular centro (última capa cargada)
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
        m.location = center

        # Verificar atributos
        atributos = [col for col in gdf.columns if col != "geometry"]

        if atributos:
            folium.GeoJson(
                gdf,
                name=url.split("/")[-1],
                tooltip=folium.GeoJsonTooltip(fields=atributos, aliases=atributos)
            ).add_to(m)
        else:
            folium.GeoJson(gdf, name=url.split("/")[-1]).add_to(m)

    except Exception as e:
        st.error(f"No se pudo cargar {url}: {e}")

# Control de capas
folium.LayerControl().add_to(m)

# Mostrar en Streamlit
st_folium(m, width=800, height=600)

# Exportar a HTML
m.save("mapa_geojson.html")
st.success("✅ Se exportó también como mapa_geojson.html")
    
# Ruta al archivo HTML generado por Folium o Plotly
archivo_html = 'mapa_geojson.html'

# Verificar si el archivo existe
if os.path.isfile(archivo_html):
    # Mostrar un mensaje o título
    st.title('Descargar Mapa de Análisis')

    # Mostrar el botón de descarga
    def descargar_html():
        with open(archivo_html, 'rb') as f:
            contenido = f.read()
        return contenido

    # Botón de descarga
    if st.button('Descargar Mapa'):
        contenido_archivo = descargar_html()
        st.download_button(label='Haz clic para descargar', data=contenido_archivo, file_name='Mapa_Analisis.html', mime='text/html')
