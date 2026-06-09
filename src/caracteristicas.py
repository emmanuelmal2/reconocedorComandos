"""
Extraccion de caracteristicas LPC a partir de archivos WAV.

Pipeline (igual que en la practica de clase):
    1. Cargar audio mono a 16 kHz
    2. Preenfasis: y[n] = x[n] - 0.95 * x[n-1]
    3. Ventanas Hamming de 512 muestras, salto 170
    4. Energia por bloque y eliminacion de silencio
    5. LPC de orden 12 por bloque valido

La salida es una matriz (n_bloques_validos, 12) que representa la secuencia
temporal de coeficientes LPC [a1..a12]. Esa secuencia alimenta el HMM.

Codigo adaptado desde practicas_pasadas/:
    - practica_2_RP: preenfasis, Hamming 512/170, energia por bloque
    - P3_PAT/src/utils/filtroWiener.py: autocorrelacion y LPC (version corregida)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample

from src.configuracion import (
    ALFA_PREENFASIS,
    FRECUENCIA_MUESTREO,
    ORDEN_LPC,
    SALTO_BLOQUE,
    TAMANIO_BLOQUE,
    UMBRAL_SILENCIO,
)


def cargar_audio(ruta: str | Path, frecuencia_objetivo: int = FRECUENCIA_MUESTREO) -> np.ndarray:
    """
    Carga un archivo WAV y lo devuelve como senal mono a la frecuencia deseada.

    Args:
        ruta: Ruta al archivo .wav
        frecuencia_objetivo: Frecuencia de muestreo objetivo (16 kHz por defecto)

    Returns:
        Senal 1-D en float64.
    """
    ruta = Path(ruta)
    audio, frecuencia = sf.read(ruta, dtype="float64", always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if frecuencia != frecuencia_objetivo:
        n_muestras = int(round(len(audio) * frecuencia_objetivo / frecuencia))
        audio = resample(audio, n_muestras)

    return np.asarray(audio, dtype=np.float64)


def aplicar_preenfasis(audio: np.ndarray, alfa: float = ALFA_PREENFASIS) -> np.ndarray:
    """
    Aplica filtro de preenfasis para realzar frecuencias altas.

    Formula de la practica:
        y[n] = x[n] - alfa * x[n-1]
    """
    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        return audio.copy()

    preenfatizado = np.zeros_like(audio)
    preenfatizado[0] = audio[0]
    preenfatizado[1:] = audio[1:] - alfa * audio[:-1]
    return preenfatizado


def dividir_en_bloques(
    audio: np.ndarray,
    tamanio_bloque: int = TAMANIO_BLOQUE,
    salto_bloque: int = SALTO_BLOQUE,
) -> np.ndarray:
    """
    Divide la senal en bloques sin ventana.

    Returns:
        Matriz (n_bloques, tamanio_bloque)
    """
    audio = np.asarray(audio, dtype=np.float64)

    if tamanio_bloque <= 0 or salto_bloque <= 0:
        raise ValueError("tamanio_bloque y salto_bloque deben ser mayores que cero.")

    if len(audio) < tamanio_bloque:
        audio = np.pad(audio, (0, tamanio_bloque - len(audio)))

    bloques = []
    for inicio in range(0, len(audio) - tamanio_bloque + 1, salto_bloque):
        bloques.append(audio[inicio : inicio + tamanio_bloque])

    return np.asarray(bloques, dtype=np.float64)


def aplicar_hamming(bloques: np.ndarray) -> np.ndarray:
    """
    Multiplica cada bloque por una ventana de Hamming.

    Ventana periodica:
        w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
    """
    bloques = np.asarray(bloques, dtype=np.float64)

    if len(bloques) == 0:
        return bloques.copy()

    tamanio = bloques.shape[1]
    n = np.arange(tamanio, dtype=np.float64)
    ventana = 0.54 - 0.46 * np.cos((2.0 * np.pi * n) / (tamanio - 1))

    return bloques * ventana


def calcular_energia(bloques: np.ndarray) -> np.ndarray:
    """Calcula la potencia (energia media) de cada bloque: E = (1/N) * sum(x[i]^2)."""
    bloques = np.asarray(bloques, dtype=np.float64)

    if len(bloques) == 0:
        return np.asarray([], dtype=np.float64)

    return np.mean(bloques ** 2, axis=1)


def eliminar_silencio(
    bloques: np.ndarray,
    umbral_relativo: float = UMBRAL_SILENCIO,
) -> np.ndarray:
    """
    Elimina bloques cuya energia esta por debajo de un umbral relativo.

    Umbral = umbral_relativo * max(energia de todos los bloques)
    """
    bloques = np.asarray(bloques, dtype=np.float64)

    if len(bloques) == 0:
        return bloques.copy()

    energias = calcular_energia(bloques)
    energia_maxima = float(np.max(energias))

    if energia_maxima <= 1e-14:
        return bloques.copy()

    umbral = umbral_relativo * energia_maxima
    mascara = energias >= umbral
    return bloques[mascara]


def _autocorrelacion_en_desfase(senal: np.ndarray, desfase: int) -> float:
    """Autocorrelacion R(desfase) = (1/N) * sum_i x[i] * x[i + desfase]."""
    x = np.asarray(senal, dtype=np.float64)
    n = len(x)

    if n <= 0 or abs(desfase) >= n:
        return 0.0

    if desfase >= 0:
        return float(np.sum(x[: n - desfase] * x[desfase:n]) / n)

    k = -desfase
    return float(np.sum(x[k:n] * x[: n - k]) / n)


def calcular_autocorrelacion(bloque: np.ndarray, orden: int = ORDEN_LPC) -> np.ndarray:
    """Calcula el vector [R(0), R(1), ..., R(orden)]."""
    bloque = np.asarray(bloque, dtype=np.float64)
    return np.array(
        [_autocorrelacion_en_desfase(bloque, k) for k in range(orden + 1)],
        dtype=np.float64,
    )


def _construir_sistema_toeplitz(r: np.ndarray, orden: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construye matriz Toeplitz M y vector b para el sistema LPC."""
    matriz = np.zeros((orden, orden), dtype=np.float64)
    for i in range(orden):
        for j in range(orden):
            matriz[i, j] = r[abs(i - j)]

    vector = r[1 : orden + 1].copy()
    return matriz, vector


def _normalizar_coeficientes_lpc(coeficientes: np.ndarray, orden: int = ORDEN_LPC) -> np.ndarray:
    """
    Garantiza exactamente `orden` coeficientes: a1, a2, ..., a_orden.

    Si la salida es [a0, a1, ..., a_orden] con a0 = 1, elimina a0.
    """
    coeficientes = np.asarray(coeficientes, dtype=np.float64).reshape(-1)

    if len(coeficientes) == orden + 1 and np.isclose(coeficientes[0], 1.0, atol=1e-6):
        coeficientes = coeficientes[1:]

    if len(coeficientes) != orden:
        raise ValueError(
            f"Se esperaban {orden} coeficientes LPC (a1..a{orden}), "
            f"pero se obtuvieron {len(coeficientes)}."
        )

    return coeficientes


def calcular_lpc(bloque: np.ndarray, orden: int = ORDEN_LPC) -> np.ndarray:
    """
    Calcula coeficientes LPC [a1, a2, ..., a_orden] mediante Yule-Walker.

    Si el metodo devuelve [a0=1, a1, ..., a_orden], elimina a0.
    Siempre retorna forma (orden,).
    """
    bloque = np.asarray(bloque, dtype=np.float64)

    if len(bloque) <= orden or float(np.mean(bloque ** 2)) <= 1e-14:
        return np.zeros(orden, dtype=np.float64)

    r = calcular_autocorrelacion(bloque, orden=orden)
    matriz, vector = _construir_sistema_toeplitz(r, orden)

    try:
        coeficientes = np.linalg.solve(matriz, vector)
    except np.linalg.LinAlgError:
        regularizacion = 1e-8 * np.eye(orden)
        coeficientes = np.linalg.solve(matriz + regularizacion, vector)

    return _normalizar_coeficientes_lpc(coeficientes, orden=orden)


def procesar_senal_lpc(
    audio: np.ndarray,
    verbose: bool = False,
    etiqueta: str = "",
) -> np.ndarray:
    """
    Pipeline LPC sobre una senal ya cargada: preenfasis -> bloques -> LPC.

    Returns:
        Matriz (n_bloques_validos, 12).
    """
    audio = aplicar_preenfasis(np.asarray(audio, dtype=np.float64))
    bloques = dividir_en_bloques(audio)
    bloques = aplicar_hamming(bloques)

    cantidad_antes = len(bloques)
    bloques = eliminar_silencio(bloques)
    cantidad_despues = len(bloques)

    if cantidad_despues == 0:
        matriz_lpc = np.zeros((0, ORDEN_LPC), dtype=np.float64)
    else:
        filas_lpc = [calcular_lpc(bloque) for bloque in bloques]
        matriz_lpc = np.vstack(filas_lpc)

    if matriz_lpc.ndim != 2 or matriz_lpc.shape[1] != ORDEN_LPC:
        raise ValueError(
            f"Forma LPC invalida: {matriz_lpc.shape}. "
            f"Se esperaba (n_bloques, {ORDEN_LPC})."
        )
    if matriz_lpc.shape[0] != cantidad_despues:
        raise ValueError(
            f"Inconsistencia: matriz tiene {matriz_lpc.shape[0]} filas "
            f"pero quedaron {cantidad_despues} bloques tras eliminar silencio."
        )

    if verbose:
        if etiqueta:
            print(f"  Origen: {etiqueta}")
        print(f"  Bloques antes de eliminar silencio: {cantidad_antes}")
        print(f"  Bloques despues de eliminar silencio: {cantidad_despues}")
        print(f"  Forma final matriz LPC: {matriz_lpc.shape}")

    return matriz_lpc


def extraer_secuencia_lpc(ruta: str | Path, verbose: bool = False) -> np.ndarray:
    """
    Pipeline completo: WAV -> matriz LPC de forma (n_bloques_validos, 12).

    Returns:
        Matriz numpy (n_bloques_validos, 12) con [a1..a12] por bloque activo.
    """
    ruta = Path(ruta)
    audio = cargar_audio(ruta)

    if verbose:
        print(f"  Ruta del audio: {ruta}")

    return procesar_senal_lpc(audio, verbose=verbose)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m src.caracteristicas <archivo.wav>")
        sys.exit(1)

    secuencia = extraer_secuencia_lpc(sys.argv[1], verbose=True)
    print(f"Secuencia LPC: forma {secuencia.shape}")
    if len(secuencia) > 0:
        print(f"Primer bloque (a1..a12): {secuencia[0]}")
