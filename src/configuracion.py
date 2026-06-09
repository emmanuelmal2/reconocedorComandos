"""
Configuracion central del proyecto de reconocimiento de comandos de voz.

Asistente con activacion por voz:
    1. Detectar intencion "activacion" (ej. "oye computadora")
    2. Escuchar comando (listar, memoria, disco, red, procesos)
    3. Ejecutar accion Bash solo para intenciones ejecutables
"""

from pathlib import Path

# Rutas del proyecto
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_DATASET = RAIZ_PROYECTO / "dataset"
CARPETA_MODELOS = RAIZ_PROYECTO / "models"
RUTA_ESCALADOR = CARPETA_MODELOS / "escalador.pkl"
NOMBRE_ESCALADOR = "escalador"

# Intenciones que entrenan y clasifican HMM
INTENCIONES = ["activacion", "listar", "memoria", "disco", "red", "procesos"]

# Intenciones que ejecutan comando Bash (activacion NO ejecuta nada)
INTENCIONES_EJECUTABLES = ["listar", "memoria", "disco", "red", "procesos"]

# Frases de entrenamiento por intencion (5 frases x 5 repeticiones = 25 audios por intencion)
FRASES = {
    "activacion": [
        "oye computadora",
        "hola computadora",
        "escucha computadora",
        "activar asistente",
        "modo comando",
    ],
    "listar": [
        "listar archivos",
        "muestra los archivos",
        "lista el directorio",
        "ver archivos",
        "ensename los archivos",
    ],
    "memoria": [
        "muestra la memoria",
        "ver memoria",
        "uso de memoria",
        "estado de memoria",
        "cuanta memoria hay",
    ],
    "disco": [
        "muestra el disco",
        "ver espacio en disco",
        "uso de disco",
        "espacio disponible",
        "cuanto espacio queda",
    ],
    "red": [
        "muestra la red",
        "ver interfaces de red",
        "estado de red",
        "muestra mi ip",
        "ver configuracion de red",
    ],
    "procesos": [
        "muestra los procesos",
        "ver procesos",
        "lista los procesos",
        "procesos activos",
        "que procesos hay",
    ],
}

# Parametros de grabacion
FRECUENCIA_MUESTREO = 16000       # Hz, mono
DURACION_GRABACION = 3            # segundos por toma
REPETICIONES_POR_FRASE = 5
FRASES_POR_INTENCION = 5

# Hablantes del dataset (cada uno graba el mismo set de frases)
HABLANTES = ["emmanuel", "elioth"]
HABLANTE_PREDETERMINADO = "emmanuel"

# Parametros de extraccion LPC (practica de clase)
ALFA_PREENFASIS = 0.95
TAMANIO_BLOQUE = 512
SALTO_BLOQUE = 170
ORDEN_LPC = 12
UMBRAL_SILENCIO = 0.1

# Parametros del GaussianHMM
HMM_N_COMPONENTS = 4
HMM_COVARIANCE_TYPE = "diag"
HMM_N_ITER = 100
HMM_RANDOM_STATE = 42

# Confianza en prediccion (diferencia de log-likelihood entre 1.º y 2.º)
# Si el margen es menor, el asistente pide repetir la frase.
MARGEN_MINIMO_ACTIVACION = 25
MARGEN_MINIMO_COMANDO = 35
MAX_REINTENTOS_COMANDO = 2
