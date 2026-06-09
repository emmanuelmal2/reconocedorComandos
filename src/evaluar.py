"""
Evaluacion del clasificador HMM sobre el dataset de entrenamiento.

Calcula precision por intencion y matriz de confusion.
Util para detectar intenciones debiles (ej. red, memoria en vivo).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.configuracion import CARPETA_DATASET, INTENCIONES
from src.predecir import calcular_margen_confianza, predecir_intencion


def _listar_audios_por_intencion(
    carpeta_dataset: Path,
    intenciones: list[str],
) -> list[tuple[str, Path]]:
    """Devuelve pares (intencion_real, ruta_wav)."""
    muestras = []
    for intencion in intenciones:
        carpeta = carpeta_dataset / intencion
        if not carpeta.is_dir():
            continue
        for ruta in sorted(carpeta.glob("*.wav")):
            muestras.append((intencion, ruta))
    return muestras


def evaluar_dataset(
    carpeta_dataset: Path = CARPETA_DATASET,
    intenciones: list[str] | None = None,
) -> dict:
    """
    Evalua todos los WAV del dataset.

    Returns:
        Diccionario con etiquetas, predicciones, margenes y metricas.
    """
    if intenciones is None:
        intenciones = INTENCIONES

    muestras = _listar_audios_por_intencion(carpeta_dataset, intenciones)
    if not muestras:
        raise ValueError(f"No hay audios en {carpeta_dataset}")

    reales = []
    predichas = []
    margenes = []
    rutas = []

    for intencion_real, ruta in muestras:
        intencion_pred, puntajes = predecir_intencion(ruta)
        margen, _, _ = calcular_margen_confianza(puntajes)

        reales.append(intencion_real)
        predichas.append(intencion_pred)
        margenes.append(margen)
        rutas.append(str(ruta))

    reales_arr = np.array(reales)
    predichas_arr = np.array(predichas)

    return {
        "intenciones": intenciones,
        "reales": reales,
        "predichas": predichas,
        "margenes": margenes,
        "rutas": rutas,
        "reales_arr": reales_arr,
        "predichas_arr": predichas_arr,
    }


def mostrar_reporte(resultado: dict) -> None:
    """Imprime precision por intencion, errores y matriz de confusion."""
    intenciones = resultado["intenciones"]
    reales = resultado["reales"]
    predichas = resultado["predichas"]
    margenes = resultado["margenes"]
    rutas = resultado["rutas"]

    print("\n" + "=" * 60)
    print("  EVALUACION DEL DATASET")
    print("=" * 60)
    print(f"  Audios evaluados: {len(reales)}")

    # Precision por intencion
    print("\n--- Precision por intencion ---")
    for intencion in intenciones:
        indices = [i for i, r in enumerate(reales) if r == intencion]
        if not indices:
            print(f"  {intencion:12s}  (sin audios)")
            continue
        aciertos = sum(1 for i in indices if predichas[i] == intencion)
        total = len(indices)
        print(f"  {intencion:12s}  {aciertos:3d}/{total:3d}  ({100 * aciertos / total:.0f}%)")

    # Errores mas frecuentes
    print("\n--- Errores de clasificacion ---")
    errores = [
        (rutas[i], reales[i], predichas[i], margenes[i])
        for i in range(len(reales))
        if reales[i] != predichas[i]
    ]
    if not errores:
        print("  Ningun error.")
    else:
        for ruta, real, pred, margen in errores:
            nombre = Path(ruta).name
            print(f"  {nombre}: {real} -> {pred}  (margen={margen:.1f})")

    # Matriz de confusion
    print("\n--- Matriz de confusion ---")
    matriz = confusion_matrix(reales, predichas, labels=intenciones)
    encabezado = "              " + "  ".join(f"{i[:6]:>6s}" for i in intenciones)
    print(encabezado)
    for idx, intencion in enumerate(intenciones):
        fila = "  ".join(f"{v:6d}" for v in matriz[idx])
        print(f"  {intencion:12s}  {fila}")

    print("\n--- Reporte sklearn ---")
    print(
        classification_report(
            reales,
            predichas,
            labels=intenciones,
            zero_division=0,
        )
    )

    # Margen promedio cuando acierta vs falla
    aciertos_m = [margenes[i] for i in range(len(reales)) if reales[i] == predichas[i]]
    fallos_m = [margenes[i] for i in range(len(reales)) if reales[i] != predichas[i]]
    if aciertos_m:
        print(f"  Margen promedio (aciertos): {np.mean(aciertos_m):.1f}")
    if fallos_m:
        print(f"  Margen promedio (errores):  {np.mean(fallos_m):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalua el clasificador HMM sobre dataset/."
    )
    parser.add_argument(
        "--carpeta-dataset",
        type=Path,
        default=CARPETA_DATASET,
        help="Carpeta raiz del dataset",
    )
    args = parser.parse_args()

    resultado = evaluar_dataset(carpeta_dataset=args.carpeta_dataset)
    mostrar_reporte(resultado)


if __name__ == "__main__":
    main()
