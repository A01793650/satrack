import streamlit as st
import pandas as pd

# Título de la aplicación
st.title('Análisis de Rutas - Fuente: SATRACK')

# Instrucciones para el usuario
st.write('Cargue el archivo de Detalle del Recorrido.')

# Widget de carga de archivo
archivo = st.file_uploader("Cargar archivo XLSX", type=['xlsx'])

# Verificar si se ha cargado un archivo
if archivo is not None:
    # Leer el archivo XLSX
    df = pd.read_excel(archivo)  # Usamos 'openpyxl' como motor para leer XLSX

    # Mostrar el DataFrame
    st.write('**Datos del archivo XLSX:**')
    st.write(df)

    # Opcional: Mostrar información adicional
    st.write(f"**Número total de filas:** {len(df)}")
    st.write(f"**Columnas:** {df.columns.tolist()}")

    # Opcional: Mostrar gráfico u otras visualizaciones
    # Ejemplo: st.bar_chart(df['columna'])

else:
    st.write('Aún no se ha cargado ningún archivo.')

