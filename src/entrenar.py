"""
Entrenamiento de GaussianHMM por intencion de voz.

Flujo:
    1. Extraer LPC de todos los audios del dataset
    2. Entrenar StandardScaler global con todos los bloques LPC
    3. Guardar escalador en models/escalador.pkl
    4. Por cada intencion y hablante: entrenar un HMM propio
    5. Guardar models/<intencion>_<hablante>.pkl

    En prediccion se usa el mejor score entre hablantes por intencion,
    asi el sistema funciona con emmanuel y elioth sin identificar la voz antes.
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
    FRASES,
    HABLANTES,
    HMM_COVARIANCE_TYPE,
    HMM_N_COMPONENTS,
    HMM_N_ITER,
    HMM_RANDOM_STATE,
    INTENCIONES,
    NOMBRE_ESCALADOR,
    ORDEN_LPC,
    RUTA_ESCALADOR,
    filtrar_audios_frase_activa,
    ruta_modelo_hablante,
)


def _listar_archivos_wav(
    carpeta_intencion: Path,
    hablante: str | None = None,
) -> list[Path]:
    """Lista archivos .wav de una carpeta de intencion, ordenados por nombre."""
    if not carpeta_intencion.is_dir():
        return []
    archivos = sorted(carpeta_intencion.glob("*.wav"))
    if hablante:
        marcador = f"_{hablante}_"
        archivos = [ruta for ruta in archivos if marcador in ruta.name]
    return archivos


def _recolectar_lpc_de_archivos(
    archivos_wav: list[Path],
    verbose: bool = False,
) -> np.ndarray:
    """Extrae LPC de una lista explicita de archivos WAV."""
    secuencias = []

    for ruta_wav in archivos_wav:
        secuencia = extraer_secuencia_lpc(ruta_wav, verbose=verbose)
        if len(secuencia) > 0:
            secuencias.append(secuencia)

    if not secuencias:
        return np.zeros((0, ORDEN_LPC), dtype=np.float64)

    return np.vstack(secuencias)


def _recolectar_lpc_global(
    intenciones: list[str],
    carpeta_dataset: Path = CARPETA_DATASET,
    hablante: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Extrae LPC de todos los audios para entrenar el escalador global."""
    archivos: list[Path] = []
    for intencion in intenciones:
        carpeta = carpeta_dataset / intencion
        wavs = _listar_archivos_wav(carpeta, hablante=hablante)
        archivos.extend(filtrar_audios_frase_activa(wavs, intencion))
    return _recolectar_lpc_de_archivos(archivos, verbose=verbose)


def entrenar_escalador_global(
    intenciones: list[str] | None = None,
    carpeta_dataset: Path = CARPETA_DATASET,
    hablante: str | None = None,
    verbose: bool = False,
) -> StandardScaler:
    """Entrena StandardScaler con todos los LPC y lo guarda en models/escalador.pkl."""
    if intenciones is None:
        intenciones = INTENCIONES

    print("\n=== Escalador global (StandardScaler) ===")
    if hablante:
        print(f"  Hablante filtrado: {hablante}")
    X_todos = _recolectar_lpc_global(
        intenciones, carpeta_dataset, hablante=hablante, verbose=verbose
    )

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


def _hablantes_con_audios(
    intencion: str,
    carpeta_dataset: Path = CARPETA_DATASET,
    hablantes: list[str] | None = None,
) -> list[str]:
    """Lista hablantes que tienen al menos un WAV para la intencion."""
    if hablantes is None:
        hablantes = HABLANTES
    carpeta = carpeta_dataset / intencion
    encontrados = []
    for hablante in hablantes:
        if _listar_archivos_wav(carpeta, hablante=hablante):
            encontrados.append(hablante)
    return encontrados


def _limpiar_modelos_obsoletos(hablantes: list[str] | None = None) -> None:
    """
    Elimina modelos .pkl que ya no corresponden a INTENCIONES vigentes
    o al esquema antiguo (un solo HMM por intencion).
    """
    if hablantes is None:
        hablantes = HABLANTES

    validos = {
        NOMBRE_ESCALADOR,
        *(f"{intencion}_{hablante}" for intencion in INTENCIONES for hablante in hablantes),
        *INTENCIONES,
    }

    for ruta in CARPETA_MODELOS.glob("*.pkl"):
        if ruta.stem not in validos:
            ruta.unlink()
            print(f"  [aviso] Eliminado modelo obsoleto: {ruta.name}")

    for png in ("matriz_confusion.png", "matriz_confusion_holdout.png"):
        ruta_png = CARPETA_MODELOS / png
        if ruta_png.exists():
            ruta_png.unlink()
            print(f"  [aviso] Eliminada grafica antigua: {png}")


def _entrenar_hmm(
    archivos_wav: list[Path],
    escalador: StandardScaler,
    verbose: bool = False,
) -> GaussianHMM | None:
    """Entrena un HMM con los archivos indicados."""
    X, longitudes = construir_datos_entrenamiento(archivos_wav, escalador, verbose=verbose)
    if len(longitudes) == 0:
        return None

    modelo = GaussianHMM(
        n_components=HMM_N_COMPONENTS,
        covariance_type=HMM_COVARIANCE_TYPE,
        n_iter=HMM_N_ITER,
        random_state=HMM_RANDOM_STATE,
    )
    modelo.fit(X, longitudes)
    return modelo


def entrenar_modelo_intencion_hablante(
    intencion: str,
    hablante: str,
    escalador: StandardScaler,
    carpeta_dataset: Path = CARPETA_DATASET,
    verbose: bool = False,
) -> GaussianHMM | None:
    """Entrena y guarda un HMM para una intencion y un hablante."""
    carpeta = carpeta_dataset / intencion
    archivos_wav = _listar_archivos_wav(carpeta, hablante=hablante)
    archivos_wav = filtrar_audios_frase_activa(archivos_wav, intencion)

    print(f"\n=== Intencion: {intencion} | Hablante: {hablante} ===")
    print(f'  Frase activa      : "{FRASES[intencion][0]}"')
    print(f"  Audios encontrados: {len(archivos_wav)}")

    if not archivos_wav:
        print("  [aviso] Sin audios, se omite.")
        return None

    modelo = _entrenar_hmm(archivos_wav, escalador, verbose=verbose)
    if modelo is None:
        print("  [error] Ningun audio produjo bloques LPC validos.")
        return None

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = ruta_modelo_hablante(intencion, hablante)
    joblib.dump(modelo, ruta_modelo)
    print(f"  Modelo guardado: {ruta_modelo}")
    return modelo


def _agrupar_por_hablante(archivos: list[Path]) -> dict[str, list[Path]]:
    """Agrupa rutas WAV por hablante segun el nombre del archivo."""
    por_hablante: dict[str, list[Path]] = {}
    for ruta in archivos:
        for hablante in HABLANTES:
            if f"_{hablante}_" in ruta.name:
                por_hablante.setdefault(hablante, []).append(ruta)
                break
    return por_hablante


def entrenar_modelos_en_memoria(
    archivos_por_intencion: dict[str, list[Path]],
    intenciones: list[str] | None = None,
    verbose: bool = False,
) -> tuple[StandardScaler, dict[str, dict[str, GaussianHMM]]]:
    """
    Entrena escalador y HMMs en memoria sin guardar en disco.

    Util para evaluacion holdout (train/test split) sin pisar models/*.pkl.
    """
    if intenciones is None:
        intenciones = INTENCIONES

    archivos_entrenamiento: list[Path] = []
    for intencion in intenciones:
        archivos_entrenamiento.extend(archivos_por_intencion.get(intencion, []))

    X_todos = _recolectar_lpc_de_archivos(archivos_entrenamiento, verbose=verbose)
    if len(X_todos) == 0:
        raise ValueError("No hay bloques LPC en la particion de entrenamiento.")

    escalador = StandardScaler()
    escalador.fit(X_todos)

    modelos: dict[str, dict[str, GaussianHMM]] = {}
    for intencion in intenciones:
        archivos = archivos_por_intencion.get(intencion, [])
        if not archivos:
            continue

        archivos = filtrar_audios_frase_activa(archivos, intencion)
        por_hablante = _agrupar_por_hablante(archivos)
        modelos[intencion] = {}
        for hablante, archivos_hablante in por_hablante.items():
            modelo = _entrenar_hmm(archivos_hablante, escalador, verbose=verbose)
            if modelo is not None:
                modelos[intencion][hablante] = modelo

    return escalador, modelos


def entrenar_todas_intenciones(
    intenciones: list[str] | None = None,
    hablante: str | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, GaussianHMM]]:
    """Entrena escalador global y un HMM por cada (intencion, hablante)."""
    if intenciones is None:
        intenciones = INTENCIONES

    hablantes_entrenar = [hablante] if hablante else HABLANTES

    escalador = entrenar_escalador_global(intenciones, hablante=hablante, verbose=verbose)
    _limpiar_modelos_obsoletos(hablantes=hablantes_entrenar)

    modelos: dict[str, dict[str, GaussianHMM]] = {}
    total_guardados = 0

    for intencion in intenciones:
        modelos[intencion] = {}
        for hab in _hablantes_con_audios(intencion, hablantes=hablantes_entrenar):
            modelo = entrenar_modelo_intencion_hablante(
                intencion, hab, escalador, verbose=verbose
            )
            if modelo is not None:
                modelos[intencion][hab] = modelo
                total_guardados += 1

    print(f"\n  Modelos por hablante guardados: {total_guardados}")
    return modelos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrena escalador y modelos HMM.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra detalle de extraccion LPC por audio",
    )
    parser.add_argument(
        "--hablante",
        choices=HABLANTES,
        help="Entrenar solo modelos de un hablante (por defecto: todos los del dataset)",
    )
    args = parser.parse_args()

    print("Entrenando escalador y modelos HMM (uno por intencion y hablante)...")
    modelos_entrenados = entrenar_todas_intenciones(
        hablante=args.hablante,
        verbose=args.verbose,
    )
    total = sum(len(por_h) for por_h in modelos_entrenados.values())
    print(f"\nEntrenamiento completado: {total} modelos guardados en {CARPETA_MODELOS}")
