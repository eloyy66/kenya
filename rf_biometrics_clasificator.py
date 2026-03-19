import requests
import time
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. Preparar el modelo de clasificación ---
# Clases: Relajado, Estrés, Enojo, Ansiedad, Fatiga
data = {
    'heart_rate': [72, 71, 60, 58, 85, 90, 110, 115, 65, 66],
    'rr_interval': [833, 845, 1000, 1034, 705, 666, 545, 521, 923, 909],
    'spo2': [98, 98, 97, 97, 99, 98, 96, 95, 99, 98],
    'skin_temp': [34.5, 34.5, 34.2, 34.1, 35.0, 35.2, 35.8, 36.0, 34.0, 34.1],
    'acc_x': [120, 115, 10, 5, 850, 900, 1200, 1500, 50, 45],
    'acc_y': [-45, -40, 5, -2, 1200, 1100, 800, 950, -10, -5],
    'acc_z': [980, 975, 1010, 1005, -300, -250, 400, 350, 990, 995],
    'label': ['Relajado', 'Relajado', 'Fatiga', 'Fatiga', 'Estrés', 'Estrés', 'Enojo', 'Enojo', 'Relajado', 'Relajado']
}

# Convertir a DataFrame y entrenar el modelo
df_entrenamiento = pd.DataFrame(data)
X = df_entrenamiento[['heart_rate', 'rr_interval', 'spo2', 'skin_temp', 'acc_x', 'acc_y', 'acc_z']]
y = df_entrenamiento['label']

print("Entrenando el modelo de clasificación...")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X, y)
print("Modelo entrenado y listo.\n")

# La IP de tu computadora
API_URL = "http://192.168.101.6:8000/latest"

def obtener_datos_del_reloj():
    try:
        response = requests.get(API_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data is None: # Si el servidor manda None, el reloj se detuvo
                return None
            return data
        else:
            return None
    except Exception:
        return None

print(" Iniciando Clasificador en Tiempo Real...")
print(" Esperando conexión del reloj...")

try:
    reloj_activo = False
    
    while True:
        datos = obtener_datos_del_reloj()
        
        if datos:
            if not reloj_activo:
                print("\n Reloj conectado. Analizando datos en tiempo real...\n")
                reloj_activo = True
            
            # Preparar los datos leídos del reloj para el modelo. 
            # Si el reloj no envía todos los datos, usamos valores por defecto similares a un estado de reposo para evitar errores.
            nueva_lectura = pd.DataFrame([{
                'heart_rate': datos.get('heart_rate', 70),
                'rr_interval': datos.get('rr_interval', 800),
                'spo2': datos.get('spo2', 98),
                'skin_temp': datos.get('skin_temp', 34.5),
                'acc_x': datos.get('acc_x', 0),
                'acc_y': datos.get('acc_y', 0),
                'acc_z': datos.get('acc_z', 980)
            }])
            
            # Realizar predicción
            emocion_detectada = modelo_rf.predict(nueva_lectura)[0]
            probabilidades = modelo_rf.predict_proba(nueva_lectura)[0]
            max_prob = max(probabilidades) * 100
            
            # Formatear la salida para que sea una sola línea que se actualiza
            hr = datos.get('heart_rate', '--')
            rr = datos.get('rr_interval', '--')
            spo2 = datos.get('spo2', '--')
            temp = datos.get('skin_temp', '--')
            acc_x = datos.get('acc_x', '--')
            acc_y = datos.get('acc_y', '--')
            acc_z = datos.get('acc_z', '--')
            
            estado_str = f"ESTADO: {emocion_detectada.upper()} ({max_prob:.1f}%)"
            datos_str = f"HR:{hr}bpm | RR:{rr}ms | SpO2:{spo2}% | Temp:{temp}°C | Acc:({acc_x},{acc_y},{acc_z})"
            
            sys.stdout.write(f"\r{datos_str}  ||  {estado_str}          ")
            sys.stdout.flush()

        else:
            if reloj_activo:
                print("\n\nConexión perdida. Esperando al reloj...")
                reloj_activo = False

        time.sleep(1) 

except KeyboardInterrupt:
    print("\n\n Programa cerrado por el usuario.")