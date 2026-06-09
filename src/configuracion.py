"""
Configuracion central del proyecto de reconocimiento de comandos de voz.

Asistente con activacion por voz:
    1. Detectar intencion "activacion" (ej. "oye computadora")
    2. Escuchar comando (listar, memoria, procesos)
    3. Ejecutar accion Bash solo para intenciones ejecutables

Se usan 3 comandos con una frase fija cada uno (sin disco/red) para evitar
confusiones entre hablantes.
"""

from pathlib import Path

# Rutas del proyecto
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_DATASET = RAIZ_PROYECTO / "dataset"
CARPETA_MODELOS = RAIZ_PROYECTO / "models"
RUTA_ESCALADOR = CARPETA_MODELOS / "escalador.pkl"
NOMBRE_ESCALADOR = "escalador"


def ruta_modelo_hablante(intencion: str, hablante: str) -> Path:
    """Ruta del HMM por intencion y hablante: models/memoria_emmanuel.pkl"""
    return CARPETA_MODELOS / f"{intencion}_{hablante}.pkl"

# Intenciones activas (activacion + 3 comandos ejecutables)
INTENCIONES = ["activacion", "listar", "memoria", "procesos"]

# Intenciones que ejecutan comando Bash (activacion NO ejecuta nada)
INTENCIONES_EJECUTABLES = ["listar", "memoria", "procesos"]

# Una frase fija por intencion (la que se usa en entrenamiento, demo y en vivo)
FRASES = {
    "activacion": ["oye computadora"],
    "listar": ["listar archivos"],
    "memoria": ["muestra la memoria"],
    "procesos": ["ver procesos"],
}

# Indice de esa frase en el dataset grabado (frase01, frase02, ...)
# Coincide con el orden original del set de 5 frases por intencion.
FRASE_INDICE_POR_INTENCION: dict[str, int] = {
    "activacion": 1,
    "listar": 1,
    "memoria": 1,
    "procesos": 2,
}

# Misma frase unica para la demo
FRASES_DEMO = FRASES


def filtrar_audios_frase_activa(archivos: list[Path], intencion: str) -> list[Path]:
    """Conserva solo los WAV de la frase activa (ej. frase01 para listar archivos)."""
    numero = FRASE_INDICE_POR_INTENCION[intencion]
    marcador = f"_frase{numero:02d}_"
    return [ruta for ruta in archivos if marcador in ruta.name]

# Parametros de grabacion
FRECUENCIA_MUESTREO = 16000       # Hz, mono
DURACION_GRABACION = 3            # segundos por toma
REPETICIONES_POR_FRASE = 5
FRASES_POR_INTENCION = 1

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

# Confianza en prediccion (diferencia de log-likelihood entre candidatos).
# En vivo el microfono de la VM suele dar margenes mas bajos que los WAV del dataset.
MARGEN_MINIMO_ACTIVACION = 25
MARGEN_MINIMO_COMANDO = 12
# Si 1.º y 2.º estan muy cerca, aceptar si el 3.º queda claramente peor (ej. listar vs memoria).
MARGEN_MINIMO_COMANDO_TERCERO = 22
MAX_REINTENTOS_COMANDO = 3

# Evaluacion holdout (train/test split estratificado)
HOLDOUT_TEST_SIZE = 0.2
HOLDOUT_RANDOM_STATE = 42

# Intenciones sugeridas para re-grabar audios mal clasificados
INTENCIONES_REGRABAR_SUGERIDAS = ["memoria", "listar", "procesos"]
