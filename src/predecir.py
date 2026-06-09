"""
Prediccion de intenciones de voz con modelos HMM entrenados.

Flujo:
    1. Cargar escalador (solo transform)
    2. Extraer y escalar secuencia LPC
    3. Calcular modelo.score(X, [len(X)]) por intencion
    4. Elegir la intencion con mayor log-likelihood
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.caracteristicas import extraer_secuencia_lpc, procesar_senal_lpc
from src.configuracion import CARPETA_MODELOS, INTENCIONES
from src.entrenar import cargar_escalador


def cargar_modelos(carpeta_modelos: Path = CARPETA_MODELOS) -> dict[str, GaussianHMM]:
    """
    Carga modelos HMM (.pkl) solo para intenciones vigentes en configuracion.

    Ignora el escalador y modelos obsoletos (fecha, ruta, salir, etc.).
    """
    modelos: dict[str, GaussianHMM] = {}

    if not carpeta_modelos.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de modelos: {carpeta_modelos}. "
            "Ejecuta primero: python -m src.entrenar"
        )

    for intencion in INTENCIONES:
        ruta_pkl = carpeta_modelos / f"{intencion}.pkl"
        if ruta_pkl.exists():
            modelos[intencion] = joblib.load(ruta_pkl)

    return modelos


def _filtrar_modelos(
    modelos: dict[str, GaussianHMM],
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
) -> dict[str, GaussianHMM]:
    """Filtra modelos por lista blanca o negra de intenciones."""
    filtrados = dict(modelos)

    if intenciones_permitidas is not None:
        permitidas = set(intenciones_permitidas)
        filtrados = {k: v for k, v in filtrados.items() if k in permitidas}

    if intenciones_excluidas is not None:
        excluidas = set(intenciones_excluidas)
        filtrados = {k: v for k, v in filtrados.items() if k not in excluidas}

    return filtrados


def calcular_puntajes_modelos(
    X: np.ndarray,
    modelos: dict[str, GaussianHMM],
) -> dict[str, float]:
    """
    Calcula log-likelihood con modelo.score(X, [len(X)]).
    """
    if len(X) == 0:
        return {intencion: float("-inf") for intencion in modelos}

    longitudes = [len(X)]
    puntajes = {}

    for intencion, modelo in modelos.items():
        try:
            puntajes[intencion] = float(modelo.score(X, longitudes))
        except Exception:
            puntajes[intencion] = float("-inf")

    return puntajes


def calcular_margen_confianza(puntajes: dict[str, float]) -> tuple[float, str, str]:
    """
    Calcula margen entre la 1.ª y 2.ª intencion (log-likelihood).

    Margen alto = clasificacion mas clara.
    Margen bajo = posible confusion (ej. red vs listar).
    """
    if len(puntajes) < 2:
        unica = next(iter(puntajes.keys()))
        return float("inf"), unica, unica

    ordenados = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)
    primero, puntaje_1 = ordenados[0]
    segundo, puntaje_2 = ordenados[1]
    margen = puntaje_1 - puntaje_2
    return margen, primero, segundo


def es_prediccion_confiable(
    puntajes: dict[str, float],
    intencion_esperada: str | None = None,
    margen_minimo: float = 35.0,
) -> tuple[bool, float, str, str]:
    """
    Indica si la prediccion es confiable segun el margen entre 1.º y 2.º.
    """
    margen, primero, segundo = calcular_margen_confianza(puntajes)
    confiable = margen >= margen_minimo
    if intencion_esperada is not None:
        confiable = confiable and primero == intencion_esperada
    return confiable, margen, primero, segundo


def predecir_intencion_desde_lpc_con_modelos(
    secuencia_lpc: np.ndarray,
    escalador: StandardScaler,
    modelos: dict[str, GaussianHMM],
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
) -> tuple[str, dict[str, float]]:
    """Predice intencion con escalador y modelos ya cargados en memoria."""
    modelos = _filtrar_modelos(modelos, intenciones_permitidas, intenciones_excluidas)

    if not modelos:
        raise RuntimeError("No hay modelos HMM disponibles para la prediccion.")

    if len(secuencia_lpc) == 0:
        raise ValueError("La secuencia LPC esta vacia.")

    X = escalador.transform(secuencia_lpc)
    puntajes = calcular_puntajes_modelos(X, modelos)
    intencion_predicha = max(puntajes, key=puntajes.get)
    return intencion_predicha, puntajes


def predecir_intencion_con_modelos(
    ruta_audio: str | Path,
    escalador: StandardScaler,
    modelos: dict[str, GaussianHMM],
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    verbose: bool = False,
) -> tuple[str, dict[str, float]]:
    """Predice intencion de un WAV usando modelos en memoria (sin leer models/*.pkl)."""
    secuencia_lpc = extraer_secuencia_lpc(ruta_audio, verbose=verbose)
    if len(secuencia_lpc) == 0:
        raise ValueError(f"No se extrajeron bloques LPC validos de {ruta_audio}.")
    return predecir_intencion_desde_lpc_con_modelos(
        secuencia_lpc,
        escalador,
        modelos,
        intenciones_permitidas=intenciones_permitidas,
        intenciones_excluidas=intenciones_excluidas,
    )


def predecir_intencion_desde_lpc(
    secuencia_lpc: np.ndarray,
    carpeta_modelos: Path = CARPETA_MODELOS,
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    verbose: bool = False,
) -> tuple[str, dict[str, float]]:
    """
    Predice intencion a partir de una matriz LPC (n_bloques, 12).
    """
    escalador = cargar_escalador(carpeta_modelos)
    modelos = cargar_modelos(carpeta_modelos)
    return predecir_intencion_desde_lpc_con_modelos(
        secuencia_lpc,
        escalador,
        modelos,
        intenciones_permitidas=intenciones_permitidas,
        intenciones_excluidas=intenciones_excluidas,
    )


def predecir_intencion(
    ruta_audio: str | Path,
    carpeta_modelos: Path = CARPETA_MODELOS,
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    verbose: bool = False,
) -> tuple[str, dict[str, float]]:
    """
    Predice la intencion mas probable para un archivo WAV.
    """
    secuencia_lpc = extraer_secuencia_lpc(ruta_audio, verbose=verbose)

    if len(secuencia_lpc) == 0:
        raise ValueError(
            f"No se extrajeron bloques LPC validos de {ruta_audio}. "
            "Verifica que el audio contenga voz."
        )

    return predecir_intencion_desde_lpc(
        secuencia_lpc,
        carpeta_modelos=carpeta_modelos,
        intenciones_permitidas=intenciones_permitidas,
        intenciones_excluidas=intenciones_excluidas,
        verbose=verbose,
    )


def predecir_intencion_desde_senal(
    senal: np.ndarray,
    carpeta_modelos: Path = CARPETA_MODELOS,
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    verbose: bool = False,
) -> tuple[str, dict[str, float]]:
    """Predice intencion a partir de una senal de audio grabada en memoria."""
    secuencia_lpc = procesar_senal_lpc(senal, verbose=verbose, etiqueta="microfono")

    if len(secuencia_lpc) == 0:
        raise ValueError("No se extrajeron bloques LPC validos del audio grabado.")

    return predecir_intencion_desde_lpc(
        secuencia_lpc,
        carpeta_modelos=carpeta_modelos,
        intenciones_permitidas=intenciones_permitidas,
        intenciones_excluidas=intenciones_excluidas,
        verbose=verbose,
    )


def mostrar_puntajes(puntajes: dict[str, float], intencion_predicha: str) -> None:
    """Muestra puntajes ordenados de mayor a menor log-likelihood."""
    print("\n--- Puntajes (log-likelihood) ---")
    puntajes_ordenados = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)

    for posicion, (intencion, puntaje) in enumerate(puntajes_ordenados, start=1):
        marca = " <-- prediccion" if intencion == intencion_predicha else ""
        print(f"  {posicion}. {intencion:12s}  {puntaje:+.4f}{marca}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predice una intencion de voz a partir de un archivo WAV."
    )
    parser.add_argument("audio", type=Path, help="Ruta al archivo WAV de entrada")
    parser.add_argument(
        "--carpeta-modelos",
        type=Path,
        default=CARPETA_MODELOS,
        help="Carpeta con modelos .pkl y escalador.pkl",
    )
    parser.add_argument(
        "--excluir",
        nargs="+",
        help="Intenciones a excluir de la prediccion (ej. activacion)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra detalle de extraccion LPC y puntajes",
    )
    args = parser.parse_args()

    intencion_predicha, puntajes = predecir_intencion(
        args.audio,
        carpeta_modelos=args.carpeta_modelos,
        intenciones_excluidas=args.excluir,
        verbose=args.verbose,
    )

    print(f"\nArchivo: {args.audio}")
    print(f"Intencion predicha: {intencion_predicha}")
    if not args.verbose:
        mostrar_puntajes(puntajes, intencion_predicha)


if __name__ == "__main__":
    main()
