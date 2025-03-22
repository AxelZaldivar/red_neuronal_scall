import random
import requests
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
# from tensorflow.keras.callbacks import EarlyStopping
from keras.src.callbacks import EarlyStopping # Evitar el sobreentrenamiento.


API_LLUVIA = "a1aaac909741453c8c1145326212101"
API_IMECA = "9bd1bbe77b6bf8318d15575eea661467"
CIUDAD = "Guadalajara"

URL_API_CLIMA = f"https://api.weatherapi.com/v1/forecast.json?key={API_LLUVIA}&q={CIUDAD}&days=1"

# Obtener pronóstico del clima.
respuesta_clima = requests.get(URL_API_CLIMA)
if respuesta_clima.status_code == 200:
    datos_clima = respuesta_clima.json()

    # Extraer coordenadas directamente del JSON
    LAT = datos_clima['location']['lat']
    LON = datos_clima['location']['lon']
    URL_API_IMECA = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_IMECA}"

    # Extraer información del clima
    lluvia_mm = datos_clima['forecast']['forecastday'][0]['day'].get('totalprecip_mm', 0)
    humedad = datos_clima['forecast']['forecastday'][0]['day'].get('avghumidity', 50)
    viento = datos_clima['forecast']['forecastday'][0]['day'].get('maxwind_kph', 10)
    print(f"Lluvia: {lluvia_mm} mm, Humedad: {humedad}%, Viento: {viento} kph")
else:
    lluvia_mm, humedad, viento = 0, 50, 10
    print("No se pudo obtener el pronóstico del clima. Se usarán valores predeterminados.")

# Obtener la calidad del aire.
respuesta_imeca = requests.get(URL_API_IMECA)
if respuesta_imeca.status_code == 200:
    datos_imeca = respuesta_imeca.json()
    imeca_actual = datos_imeca.get('list', [{}])[0].get('main', {}).get('aqi', 2) * 50
    print(f"Puntos IMECA: {imeca_actual}")
else:
    imeca_actual = 100
    print("No se pudo obtener la calidad del aire. Se usará valor predeterminado.")

# Comienza la simulación Monte Carlo.
num_simulaciones = 1000
max_dias_sin_lluvia = 10
max_imeca = 500
dias_sequia, imeca_values, humedad_values, viento_values, porcentaje_tiro = [], [], [], [], []

for _ in range(num_simulaciones):
    dias = random.randint(0, max_dias_sin_lluvia)
    imeca = random.randint(50, max_imeca)
    humedad_sim = random.randint(30, 90)
    viento_sim = random.randint(5, 40)
    
    if dias == 0:
        porcentaje_tiro.append(0)
    elif lluvia_mm > 2:
        porcentaje_tiro.append(5)
    else:
        porcentaje_tiro.append(min(50, 5 + 1.5 * dias))
    
    dias_sequia.append(dias)
    imeca_values.append(imeca)
    humedad_values.append(humedad_sim)
    viento_values.append(viento_sim)

X = np.stack((dias_sequia, imeca_values, humedad_values, viento_values), axis=1)
Y = np.array(porcentaje_tiro, dtype=float)

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, input_shape=[4], activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')

# Entrenamiento con Early Stopping.
early_stop = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
print("Entrenando modelo...")
modelo.fit(X, Y, epochs=1000, verbose=False, callbacks=[early_stop])
print("Entrenamiento finalizado.")

# Visualización de pérdida (borrar la gráfica).
#plt.plot(modelo.history.history['loss'])
#plt.title("Pérdida durante el entrenamiento.")
#plt.xlabel("Épocas.")
#plt.ylabel("Magnitud de pérdida.")
#plt.show()

# Comienza la predicción.
X_pred = np.array([[max_dias_sin_lluvia, imeca_actual, humedad, viento]])
resultado = modelo.predict(X_pred)
print(f"Para {max_dias_sin_lluvia} días sin lluvia, {imeca_actual} puntos IMECA, {humedad}% de húmedad y {viento} kph de viento, se debe tirar aproximadamente {resultado[0][0]:.2f}% del agua.")

# Mostrar el gráfico 3D (borrar la gráfica).
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(dias_sequia, imeca_values, porcentaje_tiro, c=porcentaje_tiro, cmap='viridis')
#ax.set_xlabel("Días sin lluvia.")
#ax.set_ylabel("Puntos IMECA.")
#ax.set_zlabel("Porcentaje de descarte.")
#plt.show()