from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
import numpy as np
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes cambiar "*" por ["http://localhost:8080"] si deseas limitarlo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KEYS
API_LLUVIA = "a1aaac909741453c8c1145326212101"
API_IMECA = "9bd1bbe77b6bf8318d15575eea661467"

# Ruta del modelo
MODEL_PATH = "modelo_entrenado.keras"

# Si el modelo no existe, se entrena y se guarda
if not os.path.exists(MODEL_PATH):
    import random
    from keras.src.callbacks import EarlyStopping

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

    early_stop = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    modelo.fit(X, Y, epochs=1000, verbose=False, callbacks=[early_stop])
    modelo.save(MODEL_PATH)
else:
    modelo = tf.keras.models.load_model(MODEL_PATH)

@app.get("/predecir")
def predecir_porcentaje_tiro(ciudad: str = Query(...)):
    # Obtener datos de clima
    url_clima = f"https://api.weatherapi.com/v1/forecast.json?key={API_LLUVIA}&q={ciudad}&days=1"
    respuesta_clima = requests.get(url_clima)

    if respuesta_clima.status_code != 200:
        return {"error": "No se pudo obtener el clima"}

    datos_clima = respuesta_clima.json()
    lluvia_mm = datos_clima['forecast']['forecastday'][0]['day'].get('totalprecip_mm', 0)
    humedad = datos_clima['forecast']['forecastday'][0]['day'].get('avghumidity', 50)
    viento = datos_clima['forecast']['forecastday'][0]['day'].get('maxwind_kph', 10)
    lat = datos_clima['location']['lat']
    lon = datos_clima['location']['lon']

    # Obtener IMECA
    url_imeca = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_IMECA}"
    respuesta_imeca = requests.get(url_imeca)
    if respuesta_imeca.status_code == 200:
        datos_imeca = respuesta_imeca.json()
        imeca_actual = datos_imeca.get('list', [{}])[0].get('main', {}).get('aqi', 2) * 50
    else:
        imeca_actual = 100

    dias_sin_lluvia = 10  # puedes cambiar esto por una lógica más avanzada

    X_pred = np.array([[dias_sin_lluvia, imeca_actual, humedad, viento]])
    resultado = modelo.predict(X_pred)[0][0]
    return {
        "ciudad": ciudad,
        "dias_sin_lluvia": dias_sin_lluvia,
        "imeca": imeca_actual,
        "humedad": humedad,
        "viento": viento,
        "lluvia_mm": lluvia_mm,
        "porcentaje_tiro": round(float(resultado), 2)
    }
