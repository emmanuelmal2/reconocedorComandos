"""
Prediccion de intenciones de voz con modelos HMM entrenados.

Flujo:
    1. Cargar escalador (solo transform)
    2. Extraer y escalar secuencia LPC
    3. Calcular modelo.score(X, [len(X)]) por intencion y hablante
    4. Por intencion, tomar el mejor score entre hablantes (max-pooling)
    5. Elegir la intencion con mayor log-likelihood
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.caracteristicas import extraer_secuencia_lpc, procesar_senal_lpc
from src.configuracion import CARPETA_MODELOS, HABLANTES, INTENCIONES, NOMBRE_ESCALADOR
from src.entrenar import cargar_escalador

# intencion -> hablante -> HMM
ModelosPorIntencion = dict[str, dict[str, GaussianHMM]]


def cargar_modelos(carpeta_modelos: Path = CARPETA_MODELOS) -> ModelosPorIntencion:
    """
    Carga HMMs por intencion y hablante (models/<intencion>_<hablante>.pkl).

    Soporta modelos legacy models/<intencion>.pkl como hablante "_legacy".
    """
    modelos: ModelosPorIntencion = {}

    if not carpeta_modelos.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de modelos: {carpeta_modelos}. "
            "Ejecuta primero: python -m src.entrenar"
        )

    for ruta_pkl in sorted(carpeta_modelos.glob("*.pkl")):
        nombre = ruta_pkl.stem
        if nombre == NOMBRE_ESCALADOR:
            continue

        cargado = False
        for intencion in INTENCIONES:
            for hablante in HABLANTES:
                if nombre == f"{intencion}_{hablante}":
                    modelos.setdefault(intencion, {})[hablante] = joblib.load(ruta_pkl)
                    cargado = True
                    break
            if cargado:
                break

        if not cargado and nombre in INTENCIONES:
            modelos.setdefault(nombre, {})["_legacy"] = joblib.load(ruta_pkl)

    return {k: v for k, v in modelos.items() if v}


def _filtrar_modelos(
    modelos: ModelosPorIntencion,
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    hablante: str | None = None,
) -> ModelosPorIntencion:
    """Filtra modelos por intencion y, opcionalmente, por un solo hablante."""
    filtrados = dict(modelos)

    if intenciones_permitidas is not None:
        permitidas = set(intenciones_permitidas)
        filtrados = {k: v for k, v in filtrados.items() if k in permitidas}

    if intenciones_excluidas is not None:
        excluidas = set(intenciones_excluidas)
        filtrados = {k: v for k, v in filtrados.items() if k not in excluidas}

    if hablante is not None:
        filtrados = {
            intencion: {hablante: modelo}
            for intencion, modelos_hablante in filtrados.items()
            if (modelo := modelos_hablante.get(hablante)) is not None
        }

    return filtrados


def _score_modelo(modelo: GaussianHMM, X: np.ndarray) -> float:
    try:
        return float(modelo.score(X, [len(X)]))
    except Exception:
        return float("-inf")


def calcular_puntajes_modelos(
    X: np.ndarray,
    modelos: ModelosPorIntencion,
) -> dict[str, float]:
    """
    Calcula log-likelihood por intencion.

    Con varios hablantes, usa el mejor score de cada intencion (max-pooling):
    asi funciona emmanuel y elioth sin identificar al hablante antes.
    """
    if len(X) == 0:
        return {intencion: float("-inf") for intencion in modelos}

    puntajes: dict[str, float] = {}
    for intencion, modelos_hablante in modelos.items():
        scores = [_score_modelo(modelo, X) for modelo in modelos_hablante.values()]
        puntajes[intencion] = max(scores) if scores else float("-inf")

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


def calcular_margenes_comando(
    puntajes: dict[str, float],
) -> tuple[float, float, str, str, str | None]:
    """
    Margenes 1.º-2.º y 1.º-3.º para decidir confianza en fase de comando (3 clases).
    """
    ordenados = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)
    primero, puntaje_1 = ordenados[0]
    segundo, puntaje_2 = ordenados[1]
    margen_12 = puntaje_1 - puntaje_2
    if len(ordenados) >= 3:
        tercero, puntaje_3 = ordenados[2]
        margen_13 = puntaje_1 - puntaje_3
        return margen_12, margen_13, primero, segundo, tercero
    return margen_12, float("inf"), primero, segundo, None


def es_comando_confiable(
    puntajes: dict[str, float],
    margen_minimo: float,
    margen_tercero_minimo: float,
) -> tuple[bool, float, str, str]:
    """
    True si el comando es confiable por separacion 1.º-2.º o 1.º-3.º.

    En vivo, listar/memoria pueden quedar muy cerca pero lejos de procesos;
    en ese caso el margen hasta el 3.º lugar basta para aceptar.
    """
    margen_12, margen_13, primero, segundo, _ = calcular_margenes_comando(puntajes)
    if margen_12 >= margen_minimo:
        return True, margen_12, primero, segundo
    if margen_13 >= margen_tercero_minimo:
        return True, margen_13, primero, segundo
    return False, margen_12, primero, segundo


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
    modelos: ModelosPorIntencion,
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    hablante: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Predice intencion con escalador y modelos ya cargados en memoria."""
    modelos = _filtrar_modelos(
        modelos,
        intenciones_permitidas,
        intenciones_excluidas,
        hablante=hablante,
    )

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
    modelos: ModelosPorIntencion,
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
    hablante: str | None = None,
    verbose: bool = False,
) -> tuple[str, dict[str, float]]:
    """
    Predice intencion a partir de una matriz LPC (n_bloques, 12).
    """
    del verbose  # reservado para compatibilidad
    escalador = cargar_escalador(carpeta_modelos)
    modelos = cargar_modelos(carpeta_modelos)
    return predecir_intencion_desde_lpc_con_modelos(
        secuencia_lpc,
        escalador,
        modelos,
        intenciones_permitidas=intenciones_permitidas,
        intenciones_excluidas=intenciones_excluidas,
        hablante=hablante,
    )


def predecir_intencion(
    ruta_audio: str | Path,
    carpeta_modelos: Path = CARPETA_MODELOS,
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    hablante: str | None = None,
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
        hablante=hablante,
    )


def predecir_intencion_desde_senal(
    senal: np.ndarray,
    carpeta_modelos: Path = CARPETA_MODELOS,
    intenciones_permitidas: list[str] | None = None,
    intenciones_excluidas: list[str] | None = None,
    hablante: str | None = None,
    verbose: bool = False,
    *,
    escalador: StandardScaler | None = None,
    modelos: ModelosPorIntencion | None = None,
) -> tuple[str, dict[str, float]]:
    """Predice intencion a partir de una senal de audio grabada en memoria."""
    secuencia_lpc = procesar_senal_lpc(senal, verbose=verbose, etiqueta="microfono")

    if len(secuencia_lpc) == 0:
        raise ValueError("No se extrajeron bloques LPC validos del audio grabado.")

    if escalador is not None and modelos is not None:
        return predecir_intencion_desde_lpc_con_modelos(
            secuencia_lpc,
            escalador,
            modelos,
            intenciones_permitidas=intenciones_permitidas,
            intenciones_excluidas=intenciones_excluidas,
            hablante=hablante,
        )

    return predecir_intencion_desde_lpc(
        secuencia_lpc,
        carpeta_modelos=carpeta_modelos,
        intenciones_permitidas=intenciones_permitidas,
        intenciones_excluidas=intenciones_excluidas,
        hablante=hablante,
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
        "--hablante",
        choices=HABLANTES,
        help="Usar solo modelos de un hablante (util en la VM con tu voz)",
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
        hablante=args.hablante,
        verbose=args.verbose,
    )

    print(f"\nArchivo: {args.audio}")
    print(f"Intencion predicha: {intencion_predicha}")
    if not args.verbose:
        mostrar_puntajes(puntajes, intencion_predicha)


if __name__ == "__main__":
    main()
