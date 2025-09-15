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

uploaded_file = st.file_uploader("Sube un ZIP con tu GeoJSON", type=["zip"])

if uploaded_file:
    gdf = gpd.read_file(f"zip://{uploaded_file}")

    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    folium.GeoJson(gdf).add_to(m)
    st_folium(m, width=800, height=600)
