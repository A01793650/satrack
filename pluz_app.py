import streamlit as st
import geopandas as gpd
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

st.title("Visor de Shapefile en Streamlit")

uploaded_file = st.file_uploader("Sube tu shapefile (.zip o .geojson)", type=["zip", "geojson"])

if uploaded_file:
    # Detecta formato
    if uploaded_file.name.endswith(".zip"):
        gdf = gpd.read_file(f"zip://{uploaded_file}")
    else:
        gdf = gpd.read_file(uploaded_file)
    
    st.write("📌 Vista previa de atributos")
    st.dataframe(gdf.head())

    # Mostrar en mapa
    m = leafmap.Map(center=[0, 0], zoom=2)
    m.add_gdf(gdf, layer_name="Capa cargada")
    m.to_streamlit(height=600)
else:
    st.info("👉 Sube un archivo para visualizarlo en el mapa.")
