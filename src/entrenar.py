"""
Entrenamiento de GaussianHMM por intencion de voz.

Flujo:
    1. Extraer LPC de todos los audios del dataset
    2. Entrenar StandardScaler global con todos los bloques LPC
    3. Guardar escalador en models/escalador.pkl
    4. Por cada intencion: escalar LPC, concatenar secuencias y entrenar HMM
    5. Guardar cada modelo en models/<intencion>.pkl
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.caracteristicas import extraer_secuencia_lpc
from src.configuracion import (
    CARPETA_DATASET,
    CARPETA_MODELOS,
    HMM_COVARIANCE_TYPE,
    HMM_N_COMPONENTS,
    HMM_N_ITER,
    HMM_RANDOM_STATE,
    INTENCIONES,
    NOMBRE_ESCALADOR,
    ORDEN_LPC,
    RUTA_ESCALADOR,
)


def _listar_archivos_wav(carpeta_intencion: Path) -> list[Path]:
    """Lista archivos .wav de una carpeta de intencion, ordenados por nombre."""
    if not carpeta_intencion.is_dir():
        return []
    return sorted(carpeta_intencion.glob("*.wav"))


def _recolectar_lpc_global(
    intenciones: list[str],
    carpeta_dataset: Path = CARPETA_DATASET,
    verbose: bool = False,
) -> np.ndarray:
    """Extrae LPC de todos los audios para entrenar el escalador global."""
    secuencias = []

    for intencion in intenciones:
        carpeta = carpeta_dataset / intencion
        for ruta_wav in _listar_archivos_wav(carpeta):
            secuencia = extraer_secuencia_lpc(ruta_wav, verbose=verbose)
            if len(secuencia) > 0:
                secuencias.append(secuencia)

    if not secuencias:
        return np.zeros((0, ORDEN_LPC), dtype=np.float64)

    return np.vstack(secuencias)


def entrenar_escalador_global(
    intenciones: list[str] | None = None,
    carpeta_dataset: Path = CARPETA_DATASET,
    verbose: bool = False,
) -> StandardScaler:
    """Entrena StandardScaler con todos los LPC y lo guarda en models/escalador.pkl."""
    if intenciones is None:
        intenciones = INTENCIONES

    print("\n=== Escalador global (StandardScaler) ===")
    X_todos = _recolectar_lpc_global(intenciones, carpeta_dataset, verbose=verbose)

    if len(X_todos) == 0:
        raise ValueError(
            "No hay bloques LPC en el dataset. "
            "Graba audios con: python -m src.grabar"
        )

    escalador = StandardScaler()
    escalador.fit(X_todos)

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(escalador, RUTA_ESCALADOR)
    print(f"  Bloques usados: {X_todos.shape[0]}")
    print(f"  Escalador guardado: {RUTA_ESCALADOR}")

    return escalador


def cargar_escalador(carpeta_modelos: Path = CARPETA_MODELOS) -> StandardScaler:
    """Carga el escalador entrenado. No crea uno nuevo."""
    ruta = carpeta_modelos / f"{NOMBRE_ESCALADOR}.pkl"

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el escalador: {ruta}. "
            "Ejecuta primero: python -m src.entrenar"
        )

    return joblib.load(ruta)


def construir_datos_entrenamiento(
    archivos_wav: list[Path],
    escalador: StandardScaler,
    verbose: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """
    Extrae LPC, escala y prepara X y longitudes para model.fit().

    Returns:
        X: (total_bloques, 12) escalado
        longitudes: bloques por audio
    """
    secuencias = []
    longitudes = []

    for ruta_wav in archivos_wav:
        secuencia_lpc = extraer_secuencia_lpc(ruta_wav, verbose=verbose)

        if len(secuencia_lpc) == 0:
            print(f"  [aviso] Sin bloques validos: {ruta_wav.name}, se omite")
            continue

        secuencias.append(secuencia_lpc)
        longitudes.append(len(secuencia_lpc))

    if not secuencias:
        return np.zeros((0, ORDEN_LPC), dtype=np.float64), []

    X = np.vstack(secuencias)
    X = escalador.transform(X)
    return X, longitudes


def entrenar_modelo_intencion(
    intencion: str,
    escalador: StandardScaler,
    carpeta_dataset: Path = CARPETA_DATASET,
    verbose: bool = False,
) -> GaussianHMM | None:
    """
    Entrena un HMM para una intencion con modelo.fit(X, longitudes).
    """
    carpeta = carpeta_dataset / intencion
    archivos_wav = _listar_archivos_wav(carpeta)

    print(f"\n=== Intencion: {intencion} ===")
    print(f"  Audios encontrados: {len(archivos_wav)}")

    if not archivos_wav:
        print(f"  [error] No hay archivos WAV en dataset/{intencion}/")
        return None

    X, longitudes = construir_datos_entrenamiento(archivos_wav, escalador, verbose=verbose)
    print(f"  Bloques totales: {X.shape[0]}, secuencias: {len(longitudes)}")

    if len(longitudes) == 0:
        print("  [error] Ningun audio produjo bloques LPC validos.")
        return None

    modelo = GaussianHMM(
        n_components=HMM_N_COMPONENTS,
        covariance_type=HMM_COVARIANCE_TYPE,
        n_iter=HMM_N_ITER,
        random_state=HMM_RANDOM_STATE,
    )

    modelo.fit(X, longitudes)

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = CARPETA_MODELOS / f"{intencion}.pkl"
    joblib.dump(modelo, ruta_modelo)
    print(f"  Modelo guardado: {ruta_modelo}")

    return modelo


def entrenar_todas_intenciones(
    intenciones: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, GaussianHMM]:
    """Entrena escalador global y un HMM por cada intencion."""
    if intenciones is None:
        intenciones = INTENCIONES

    escalador = entrenar_escalador_global(intenciones, verbose=verbose)

    modelos = {}
    for intencion in intenciones:
        modelo = entrenar_modelo_intencion(intencion, escalador, verbose=verbose)
        if modelo is not None:
            modelos[intencion] = modelo

    return modelos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrena escalador y modelos HMM.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra detalle de extraccion LPC por audio",
    )
    args = parser.parse_args()

    print("Entrenando escalador y modelos HMM...")
    modelos_entrenados = entrenar_todas_intenciones(verbose=args.verbose)
    print(
        f"\nEntrenamiento completado: {len(modelos_entrenados)} "
        f"modelos guardados en {CARPETA_MODELOS}"
    )
