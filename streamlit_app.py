import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

from folium.plugins import MarkerCluster
#from sklearn.base import TransformerMixin, BaseEstimator
#from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
#from sklearn.pipeline import Pipeline, make_pipeline
#from sklearn.ensemble import IsolationForest

# Importamos los transformadores
#from transformadores import CustomCleaner, DropColumnsAndRows, TimeTransformer, DropOnlyColums, RowDropper, CustomLabelEncoder, CustomStandardScaler, CustomMinMaxScaler, OneHotEncoderConcat
#from transformadoresCaracteristicas import DateTimeUnifier, CoordenadasMerger, DuracionEstadoMinutos, UltimoRegistroPorEstado, RangoTiempoEvento, Horario
from preprocesamiento import pipeline_preprocesamiento


#######################
# Page configuration
st.set_page_config(
    page_title="Análisis de rutas",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

#######################
# Variables Globales
#global resgistros_totales
resgistros_totales = None
resgistros_filtrados = None
total_vehiculos = None
lista_vehiculos = None

recorrido = None
df_recorrido_trans = None

#######################
# Sidebar
with st.sidebar:
    st.title('🗺️ Análisis de Rutas - Fuente: SATRACK')
    # Título de la aplicación

    my_list = ['Sin información aún']

    st.header('Filtros')
    
    if lista_vehiculos is None:
        st.multiselect(
            "Filtre por la placa del vehículo:", 
            my_list #my_list, 
            #["China", "United States of America"]
        ),
    else:
        st.multiselect(
            "Filtre por la placa del vehículo:", 
            lista_vehiculos #my_list, 
            #["China", "United States of America"]
        )

    # Add a slider to the sidebar:
    add_slider = st.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )

    st.button('Filtrar')

#########################
# Gráficos
#def generar_graficos(df_filtered_estado_apagado):
#    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 fila, 2 columnas

    # Gráfico de 'Sentido'
    #df_filtered_estado_apagado.groupby('Sentido').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'), ax=axs[0])
    #axs[0].spines[['top', 'right']].set_visible(False)
    #axs[0].set_title('Distribución por Sentido. DF sin Estado Apagado')

    # Gráfico de 'Horario'
    #df_filtered_estado_apagado.groupby('Horario').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'), ax=axs[1])
    #axs[1].spines[['top', 'right']].set_visible(False)
    #axs[1].set_title('Distribución por Horario. DF sin Estado Apagado')

    #return plt.show()
#def generar_graficos(df_filtered_estado_apagado):
    #plt.figure(figsize=(10, 5))  # Tamaño de la figura

    # Gráfico de 'Sentido'
    #df_filtered_estado_apagado.groupby('Sentido').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
    #plt.gca().spines[['top', 'right']].set_visible(False)
    #plt.title('Distribución por Sentido. DF sin Estado Apagado')

    #return plt.show()

def generar_graficos(df_filtered_estado_apagado):
    st.title('Distribución por Sentido. DF sin Estado Apagado')

    # Gráfico de 'Sentido'
    sentido_counts = df_filtered_estado_apagado.groupby('Sentido').size()
    st.bar_chart(sentido_counts)


#########################
# Funciones principales del Dashboard
def procesar_archivo():
    global resgistros_totales
    global resgistros_filtrados
    global total_vehiculos
    global lista_vehiculos
    global df_recorrido_trans

    # Verificar si se ha cargado un archivo
    if recorrido is not None:
        # Leer el archivo CSV
        df_recorrido = pd.read_csv(recorrido)

        # Mostrar el DataFrame
        #st.write('**Datos del archivo CSV:**')
        #st.write(df_recorrido)

        # Opcional: Mostrar información adicional
        #st.write('**INFORMACIÓN DE DATOS CARGADOS**')
        #st.write(f"**Número total de filas:** {len(df_recorrido)}")
        #st.write(f"**Columnas:** {df_recorrido.columns.tolist()}")

        # Copia del DF original
        df_copia = df_recorrido.copy()
        
        # Selección de características relevantes
        #features = ['Estado', 'Tipo de Evento', 'Sentido', 'Velocidad (km/h)', 'Hora', 'Día de la semana', 'Es fin de semana']
        #X = df[features]
        
        pipeline_preprocesamiento.fit(df_copia)
        # Transformamos los datos
        df_recorrido_trans = pipeline_preprocesamiento.transform(df_copia)   

        # Llenamos los indicadores
        resgistros_totales = str(round(len(df_copia) / 1000, 1)) + " K"
        resgistros_filtrados = str(round(len(df_recorrido_trans) / 1000, 1)) + " K"
        total_vehiculos = len(df_recorrido_trans["Vehículo"].unique())
        lista_vehiculos = list(df_recorrido_trans["Vehículo"].unique())

        # resgistros_filtrados = len(df_recorrido_trans)

        # Función para verificar si un objeto es un DataFrame
        def es_dataframe(obj):
            return isinstance(obj, pd.DataFrame)
        
        if es_dataframe(df_recorrido_trans):
                #st.success("El objeto es un DataFrame.")
                st.success("Archivo cargado con éxito.")
                #st.write("Las primeras filas del DataFrame son:")
                #st.table(df_recorrido_trans[cols])
                #st.table(df_recorrido_trans.head(10))
    else:
        st.write('Aún no se ha cargado ningún archivo.')

#  Función para descargar archivo y mapa
def descargar_archivo_mapa():
    # Función para descargar el DataFrame como archivo CSV
    def descargar_csv(df):
        try:
            # Crear un buffer de BytesIO para almacenar temporalmente el texto
            buffer = BytesIO()
            # Convertir el DataFrame a una cadena de texto (tabulado en este ejemplo)
            text_data = df.to_csv(index=False, sep='\t')
            # Escribir la cadena de texto en el buffer
            buffer.write(text_data.encode())
            # Obtener los bytes del buffer
            buffer.seek(0)
            return buffer
        except Exception as e:
            st.error(f"Error al exportar a TXT: {str(e)}")
    
    # Ejemplo de uso en Streamlit
    def main():
        
        # Verificar si el DataFrame no está vacío
        if not df_recorrido_trans.empty:
            st.write('**INFORMACIÓN DE DATOS FILTRADOS**')
            st.write(f"**Número total de filas:** {len(df_recorrido_trans)}")
            st.write(f"**Columnas:** {df_recorrido_trans.columns.tolist()}")
            st.write(df_recorrido_trans.head())
            
            # Botón de descarga CSV
            if st.button('Descargar CSV'):
                archivo_csv = descargar_csv(df_recorrido_trans)
                if archivo_csv:
                    st.download_button(label='Haz clic para descargar', data=archivo_csv, file_name='datos.csv', mime='text/csv')
    
        else:
            st.error('El DataFrame está vacío. No hay datos para mostrar.')
    
    if __name__ == "__main__":
        main()
        

    # Mapa centrado en una ubicación promedio
    map_center = [ df_recorrido_trans['Latitud'].mean(),  df_recorrido_trans['Longitud'].mean()]
    mapa = folium.Map(location=map_center, zoom_start=6)
    
    # Agrupar marcadores
    marker_cluster = MarkerCluster().add_to(mapa)
    
    # Añadir marcadores al grupo
    for _, row in  df_recorrido_trans.iterrows():
        folium.Marker(
            location=[row['Latitud'], row['Longitud']],
            popup=f"Vehículo: {row['Vehículo']}<br>Estado: {row['Estado']}<br>Duración: {row['DuracionEstadoMin']} min <br>Coordenadas: {row['Latitud']} {row['Longitud']}<br>Fecha: {row['datetime GPS']}",
            icon=folium.Icon(color='blue' if row['Estado'] == 'Apagado' else 'green' if row['Estado'] == 'Detenido' else 'red')
        ).add_to(marker_cluster)
    
    # Mostrar mapa
    mapa.save('Mapa_Analisis.html')
    
    # Ruta al archivo HTML generado por Folium o Plotly
    archivo_html = 'Mapa_Analisis.html'
    
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
    else:
        st.error('El archivo HTML generado no se encontró. Por favor, genera el mapa primero.')

#######################
# Dashboard procesamiento y gráfico
col_archivo = st.columns((6, 2), gap='medium')
with col_archivo[0]:
    st.markdown('#### Procesamiento, métricas y gráficos:')
    recorrido = st.file_uploader("Cargar archivo CSV", type=['csv'])
    if recorrido is not None:
        procesar_archivo()
    
    indi = st.columns(5)
    with indi[0]:
        st.metric(label="Registros totales", value=resgistros_totales) #, delta="1.2 K")
    with indi[1]:
        st.metric(label="Registros filtrados", value=resgistros_filtrados)
    with indi[2]:
        st.metric(label="Total de vehículos", value=total_vehiculos)
    
    if recorrido is not None:
        # Llama a la función con tu DataFrame df_filtered_estado_apagado
        generar_graficos(df_recorrido_trans)
        st.write("Las primeras filas del DataFrame son:")
        st.write(df_recorrido_trans.head())

with col_archivo[1]:
    st.markdown('#### Acerca de la aplicación:')
    # Instrucciones para el usuario
    with st.expander('Acerca de la aplicación', expanded=True):
        st.write('''
            En esta aplicación podrás:
            - Identificar rápidamente los lugares donde uno o varios vehículos paran y por cuanto tiempo lo hicieron.
            - Extraer el archivo identificando si estuvo o estuvieron en puntos autorizados.
            - Descargar el mapa con la información sobre los puntos geográficos de los vehículos.
            ''')

#######################
# Indicadores principales
#st.metric(label="Registros totales", value=resgistros_totales) #, delta="1.2 K")
#st.metric(label="Registros totales", value=resgistros_filtrados)
#indi = st.columns(5)
#with indi[0]:
#    st.metric(label="Registros totales", value=resgistros_totales) #, delta="1.2 K")
#with indi[1]:
#    st.metric(label="Registros filtrados", value=resgistros_filtrados)
#with indi[2]:
#    st.metric(label="Total de vehículos", value=total_vehiculos)


#######################
# Dashboard generación Mapa
col = st.columns((6, 2), gap='medium')

with col[0]:
    st.markdown('#### Generación de mapa')

    # Widget de carga de archivo
   # recorrido = st.file_uploader("Cargar archivo CSV", type=['csv'])
    #if recorrido is not None:   
        #descargar_archivo_mapa()
        
    #resgistros_totales = len(recorrido)

    # You can use a column just like st.sidebar:
    #st.button('Press me!')
    # Or even better, call Streamlit functions inside a "with" block:
    #chosen = st.radio(
    #    'Sorting hat',
    #    ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    #st.write(f"You are in {chosen} house!")

#with col[1]:
#    st.markdown('#### Acerca de la aplicación:')
#    # Instrucciones para el usuario
#    with st.expander('Acerca de la aplicación', expanded=True):
#        st.write('''
#            En esta aplicación podrás:
#            - Identificar rápidamente los lugares donde uno o varios vehículos paran y por cuanto tiempo lo hicieron.
#            - Extraer el archivo identificando si estuvo o estuvieron en puntos autorizados.
#            - Descargar el mapa con la información sobre los puntos geográficos de los vehículos.
#            ''')

