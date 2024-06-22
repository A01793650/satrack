import streamlit as st
import pandas as pd
import folium
import os

from folium.plugins import MarkerCluster
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import IsolationForest

# Transformador para la limpieza
class CustomCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, map_dict):
        self.map_dict = map_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:, 'Tipo de Evento'] = X['Tipo de Evento'].str.lower()
        X.loc[:, 'Tipo de Evento'] = X['Tipo de Evento'].str.replace('-', '')
        X.loc[:, 'Tipo de Evento'] = X['Tipo de Evento'].str.replace('[^\w\s]', '')
        X.loc[:, 'Tipo de Evento'] = X['Tipo de Evento'].str.strip()
        X.loc[:, 'Tipo de Evento'] = X['Tipo de Evento'].replace(self.map_dict, regex=True)
        return X

# Transformador para eliminar variables y registros
class DropColumnsAndRows(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop, column_condition, condition_value):
        self.columns_to_drop = columns_to_drop
        self.column_condition = column_condition
        self.condition_value = condition_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(self.columns_to_drop, axis=1)
        X_copy = X.copy()
        X_copy = X_copy[X_copy[self.column_condition] != self.condition_value]
        return X_copy

# Transformador para las transformaciones de horas
class TimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Hora GPS'] = X['Hora GPS'].str.replace(' AM', '').str.replace(' PM', '')
        X['Hora Sistema'] = X['Hora Sistema'].str.replace(' AM', '').str.replace(' PM', '')
        X['Hora GPS'] = pd.to_datetime(X['Hora GPS'], format='%H:%M:%S').dt.time
        X['Hora Sistema'] = pd.to_datetime(X['Hora Sistema'], format='%H:%M:%S').dt.time
        #X['Hora Sistema'] = X['Hora Sistema'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second) # Nuevo, probar
        #X['Hora GPS'] = X['Hora GPS'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second) # Nuevo, probar
        return X

# Transformador para eliminar columnas #y hacer otra transformación de hora
class DropOnlyColums(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(self.columns_to_drop, axis=1)
        return X

# Transformador para eliminar filas, tiene la opción para eliminar o conservar filas
class RowDropper(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_filter, categories_to_erase=None, categories_to_conserve=None):
        self.column_to_filter = column_to_filter
        self.categories_to_erase = categories_to_erase
        self.categories_to_conserve = categories_to_conserve

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.categories_to_erase:
            X = X[~X[self.column_to_filter].isin(self.categories_to_erase)]
        if self.categories_to_conserve:
            X = X[X[self.column_to_filter].isin(self.categories_to_conserve)]
        return X
    # Ejemplo de uso
    # column_to_filter = 'Tipo de Evento'
    # categories_to_erase = ['movimiento', 'exceso de velocidad']
    # categories_to_conserve = ['parado', 'otra categoría']
    # row_dropper = RowDropper(column_to_filter, categories_to_erase, categories_to_conserve)
    # X_transformed = row_dropper.transform(X)

# Transformador para LabelEncoder
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, keep_original=True):
        """
        Custom transformer to encode specified columns and optionally keep or drop original columns.

        Parameters:
        - columns: List of column names to encode.
        - keep_original: If True, keep the original columns; if False, drop them (default is True).
        """
        self.columns = columns
        self.keep_original = keep_original
        self.encoders = {}  # Store LabelEncoders for each column

    def fit(self, X, y=None):
        for col in self.columns:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        X_encoded = X.copy()

        for col in self.columns:
            encoded_col_name = f"{col}_LabelEncoded"
            X_encoded[encoded_col_name] = self.encoders[col].transform(X_encoded[col])

        if not self.keep_original:
            X_encoded.drop(columns=self.columns, inplace=True)

        return X_encoded

# Transformador para StandardScaler
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, keep_original=True):
        """
        Custom transformer to scale specified columns using Standard Scaler.

        Parameters:
        - columns: List of column names to scale.
        - keep_original: If True, keep the original columns; if False, drop them (default is True).
        """
        self.columns = columns
        self.keep_original = keep_original
        self.scalers = {}  # Store StandardScalers for each column

    def fit(self, X, y=None):
        for col in self.columns:
            scaler = StandardScaler()
            scaler.fit(X[col].values.reshape(-1, 1))  # Reshape to 2D array
            self.scalers[col] = scaler
        return self

    def transform(self, X):
        X_scaled = X.copy()

        for col in self.columns:
            scaled_col_name = f"{col}_standardScaled"
            X_scaled[scaled_col_name] = self.scalers[col].transform(X_scaled[col].values.reshape(-1, 1))

        if not self.keep_original:
            X_scaled.drop(columns=self.columns, inplace=True)

        return X_scaled

# Transformador para MinMaxScaler
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, keep_original=True):
        """
        Custom transformer to scale specified columns using MinMax Scaler.

        Parameters:
        - columns: List of column names to scale.
        - keep_original: If True, keep the original columns; if False, drop them (default is True).
        """
        self.columns = columns
        self.keep_original = keep_original
        self.scalers = {}  # Store MinMaxScalers for each column

    def fit(self, X, y=None):
        for col in self.columns:
            scaler = MinMaxScaler()
            scaler.fit(X[col].values.reshape(-1, 1))  # Reshape to 2D array
            self.scalers[col] = scaler
        return self

    def transform(self, X):
        X_scaled = X.copy()

        for col in self.columns:
            scaled_col_name = f"{col}_minMaxScaled"
            X_scaled[scaled_col_name] = self.scalers[col].transform(X_scaled[col].values.reshape(-1, 1))

        if not self.keep_original:
            X_scaled.drop(columns=self.columns, inplace=True)

        return X_scaled

# Transformador OneHotEncoder que concatena el resultado al mismo DF
class OneHotEncoderConcat(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Obtener One-Hot Encoding para la columna especificada
        tipo_evento_encoded = pd.get_dummies(X[self.column], prefix=self.column)

        tipo_evento_encoded = tipo_evento_encoded.astype(int)

        # Unir las nuevas columnas al DataFrame original
        X = pd.concat([X, tipo_evento_encoded], axis=1)
        return X

# Transformador para el Modelo InsolationForest
class CustomIsolationForest(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        self.model_ = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, contamination=self.contamination, random_state=self.random_state)
        self.model_.fit(X)
        return self

    def transform(self, X):
        scores = self.model_.decision_function(X)
        return scores

# Transformador para unificar fecha y hora en la misma columna
class DateTimeUnifier(BaseEstimator, TransformerMixin):
    #def __init__(self, date_col='fecha', time_col='hora', datetime_col='datetime'):
    def __init__(self, date_col, time_col, datetime_col='datetime'):
        self.date_col = date_col
        self.time_col = time_col
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #X = X.copy()
        X[self.date_col] = X[self.date_col].astype(str)  # Convertir a string
        X[self.time_col] = X[self.time_col].astype(str)  # Convertir a string
        X[self.datetime_col] = pd.to_datetime(X[self.date_col] + ' ' + X[self.time_col])
        # Unir las nuevas columnas al DataFrame original
        #X = pd.concat([X, X[self.datetime_col]], axis=1)
        return X

# Transformador que permite unificar etiqueta de las coordenadas de otro DF con la información
class CoordenadasMerger(BaseEstimator, TransformerMixin):
    def __init__(self, df_referencia, columna_etiqueta, tol=0.0001, etiqueta_no_autorizado='Sitio no autorizado'):
        self.df_referencia = df_referencia
        self.columna_etiqueta = columna_etiqueta
        self.tol = tol
        self.etiqueta_no_autorizado = etiqueta_no_autorizado

    def fit(self, X, y=None):
        # No es necesario entrenar nada para este transformador
        return self

    def transform(self, X):
        # Asegurarse de que las coordenadas estén en el mismo formato
        X = X.copy()
        df_referencia_indexed = self.df_referencia.set_index(['LATITUD', 'LONGITUD'])

        # Función para buscar la etiqueta más cercana dentro de la tolerancia
        def etiqueta_cercana(lat, lon, df_referencia_indexed, columna_etiqueta, tol, etiqueta_no_autorizado):
            # Buscar en un cuadrado alrededor del punto
            for lat_shift in np.arange(-tol, tol + tol/10, tol/10):
                for lon_shift in np.arange(-tol, tol + tol/10, tol/10):
                    try:
                        return df_referencia_indexed.loc[(lat + lat_shift, lon + lon_shift), columna_etiqueta]
                    except KeyError:
                        continue
            return etiqueta_no_autorizado  # Si no encuentra coincidencia, retorna la etiqueta de sitio no autorizado

        # Aplicar la función a cada fila de X
        X[self.columna_etiqueta] = X.apply(
            lambda row: etiqueta_cercana(
                row['Latitud'], row['Longitud'], df_referencia_indexed, self.columna_etiqueta, self.tol, self.etiqueta_no_autorizado
            ), axis=1
        )

        return X

# Transformador que permite calcular el tiempo de demora en cada estado
class DuracionEstadoMinutos(BaseEstimator, TransformerMixin):
    def __init__(self, columna_vehiculo='Vehículo', columna_datetime='datetime GPS', columna_estado='Estado'):
        self.columna_vehiculo = columna_vehiculo
        self.columna_datetime = columna_datetime
        self.columna_estado = columna_estado

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Asegurarse de que las columnas estén en el formato correcto
        X[self.columna_datetime] = pd.to_datetime(X[self.columna_datetime])
        X = X.sort_values(by=[self.columna_vehiculo, self.columna_datetime])

        # Calcular la diferencia de tiempo entre filas consecutivas
        X['Diferencia'] = X.groupby(self.columna_vehiculo)[self.columna_datetime].diff().shift(-1)

        # Identificar los cambios de estado
        X['CambioEstado'] = X[self.columna_estado] != X[self.columna_estado].shift(1)

        # Calcular la duración del estado
        X['DuracionEstado'] = X.groupby([self.columna_vehiculo, (X['CambioEstado']).cumsum()])['Diferencia'].transform('sum')

        # Eliminar columnas temporales
        X.drop(['Diferencia', 'CambioEstado'], axis=1, inplace=True)

        # Convertir la columna a tipo timedelta
        X['DuracionEstado'] = pd.to_timedelta(X['DuracionEstado'])

        # Convertir a minutos
        #X['DuracionEstado_Minutos'] = X['DuracionEstado'].dt.total_seconds() / 60

        # LLevar a minutos
        X['DuracionEstadoMin'] = (X['DuracionEstado'].dt.total_seconds() / 60).round(2)

        return X

# Transformador para conservar el último valor con los minutos de duración por Estado
class UltimoRegistroPorEstado(BaseEstimator, TransformerMixin):
    def __init__(self, columnas=['Vehículo', 'Estado', 'DuracionEstadoMin']):
        self.columnas = columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Identificar cambios en las columnas especificadas
        X['group'] = (X[self.columnas].shift() != X[self.columnas]).any(axis=1).cumsum()

        # Seleccionar el último registro de cada grupo
        X_resultado = X.groupby('group').tail(1)

        # Eliminar la columna 'group' usada para el cálculo
        X_resultado = X_resultado.drop(columns=['group'])
        return X_resultado

# Transformador que permite estructurar en rangos el tiempo de demora de los Eventos
class RangoTiempoEvento(BaseEstimator, TransformerMixin):
    def __init__(self, columna_duracion='DuracionEstadoMin'):
        self.columna_duracion = columna_duracion

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Definir una función para categorizar la duración
        def categorizar_duracion(duracion):
            if duracion <= 15:
                return 'Menos de 15 min'
            elif 15 < duracion <= 30:
                return '15 y 30 min'
            elif 30 < duracion <= 60:
                return '30 y 60 min'
            else:
                return 'Más de 60 min'

        # Aplicar la función a la columna de duración para crear la nueva característica
        X['RangoTiempoEvento'] = X[self.columna_duracion].apply(categorizar_duracion)
        return X

# Transformador para crear la característica horario: día o noche
class Horario(BaseEstimator, TransformerMixin):
    def __init__(self, columna_datetime='datetime GPS'):
        self.columna_datetime = columna_datetime

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convertir la columna datetime a tipo datetime si aún no lo es
        X[self.columna_datetime] = pd.to_datetime(X[self.columna_datetime])

        # Definir una función para determinar si es día o noche
        def dia_o_noche(hora):
            if 6 <= hora.hour < 18:
                return 'Día'
            else:
                return 'Noche'

        # Aplicar la función a la columna de datetime para crear la nueva característica
        X['Horario'] = X[self.columna_datetime].apply(lambda x: dia_o_noche(x))
        return X

# Construímos el pipeline para el preprocesamiento y creación de nuevas características
pipeline_preprocesamiento = Pipeline([
    ('Eliminar filas de Estado del vehículo', RowDropper(column_to_filter='Estado',
                                                         categories_to_erase=['Movimiento'],
                                                         categories_to_conserve=None)),
    #('Agregar etiqueta de sitios autorizados', CoordenadasMerger(df_referencia=df_paradas, columna_etiqueta='JUSTIFICACION EVENTO', tol=0.0001)),
    ('Trf1: cleaning', CustomCleaner(map_dict={'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'})),
    ('Trf2: drop_columns_and_rows', DropColumnsAndRows(columns_to_drop=['Edad del dato', 'Nivel de Batería',
                                                                        'Temperatura', 'Estado de la puerta'],
                                                       column_condition='Sentido', condition_value='-')),
    ('time_transformer', TimeTransformer()),
    ('Unificar columna fecha y hora GPS', DateTimeUnifier(date_col='Fecha GPS', time_col='Hora GPS', datetime_col='datetime GPS')),
    ('Borrar columnas de horas y fechas', DropOnlyColums(columns_to_drop=['Fecha GPS', 'Hora GPS', 'Fecha Sistema',
                                                                          'Hora Sistema'])),
    ('Calcular duración de cada estado', DuracionEstadoMinutos()),
    ('Dejar único valor por duración en cada estado', UltimoRegistroPorEstado()),
    ('Crear característica de Rango de tiempo por evento', RangoTiempoEvento()),
    ('Crear característica de Horario: día o noche', Horario()),
])

# Título de la aplicación
st.title('Análisis de Rutas - Fuente: SATRACK')

# Instrucciones para el usuario
st.write('En esta aplicación podrás:')
st.write('1. Identificar rápidamente los lugares donde uno o varios vehículos paran y por cuanto tiempo lo hicieron')
st.write('2. Extraer el archivo identificando si estuvo o estuvieron en puntos autorizados')
st.write(' ')         
st.write('Cargue el archivo de Detalle del Recorrido.')
st.write('')

# Widget de carga de archivo
recorrido = st.file_uploader("Cargar archivo CSV", type=['csv'])

# Verificar si se ha cargado un archivo
if recorrido is not None:
    # Leer el archivo CSV
    df_recorrido = pd.read_csv(recorrido)

    # Mostrar el DataFrame
    #st.write('**Datos del archivo CSV:**')
    #st.write(df_recorrido)

    # Opcional: Mostrar información adicional
    st.write('**INFORMACIÓN DE DATOS CARGADOS**')
    st.write(f"**Número total de filas:** {len(df_recorrido)}")
    st.write(f"**Columnas:** {df_recorrido.columns.tolist()}")

    # Copia del DF original
    df_copia = df_recorrido.copy()
    
    # Selección de características relevantes
    #features = ['Estado', 'Tipo de Evento', 'Sentido', 'Velocidad (km/h)', 'Hora', 'Día de la semana', 'Es fin de semana']
    #X = df[features]
    
    pipeline_preprocesamiento.fit(df_copia)
    # Transformamos los datos
    df_recorrido_trans = pipeline_preprocesamiento.transform(df_copia) 
    #df['your_duration_column'] = df['your_duration_column'].astype(str)
    #df_recorrido_trans = df_recorrido_trans [['Vehículo', 'Estado', 'Tipo de Evento', 'Ubicación', 'Velocidad (km/h)', 'Odómetro', 'Longitud', 'Latitud', 'Sentido', 'datetime GPS', 'DuracionEstadoMin', 'RangoTiempoEvento', 'Horario']]

    # Función para descargar el DataFrame como archivo CSV
    def descargar_csv(df):
        try:
            # Crear un buffer de BytesIO para almacenar temporalmente el texto
            buffer = BytesIO()
            # Convertir el DataFrame a una cadena de texto (tabulado en este ejemplo)
            text_data = df.to_csv()
            # Escribir la cadena de texto en el buffer
            buffer.write(text_data.encode())
            # Obtener los bytes del buffer
            buffer.seek(0)
            return buffer
        except Exception as e:
            st.error(f"Error al exportar a CSV: {str(e)}")
    
    def main():
        # Verificar si el DataFrame no está vacío
        if not df_recorrido_trans.empty:
            st.write('**INFORMACIÓN DE DATOS FILTRADOS**')
            st.write(f"**Número total de filas:** {len(df_recorrido_trans)}")
            st.write(f"**Columnas:** {df_recorrido_trans.columns.tolist()}")
            st.text(df_recorrido_trans.dtypes)
    
            # Botón de descarga TXT
            if st.button('Descargar CSV'):
                archivo_txt = descargar_csv(df_recorrido_trans)
                if archivo_txt:
                    st.download_button(label='Haz clic para descargar', data=archivo_txt, file_name='datos.csv')
    
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

else:
    st.write('Aún no se ha cargado ningún archivo.')




