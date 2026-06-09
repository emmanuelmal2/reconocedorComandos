"""
Evaluacion del clasificador HMM sobre el dataset de entrenamiento.

Calcula precision por intencion y matriz de confusion.
Util para detectar intenciones debiles (ej. red, memoria en vivo).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.configuracion import (
    CARPETA_DATASET,
    CARPETA_MODELOS,
    HOLDOUT_RANDOM_STATE,
    HOLDOUT_TEST_SIZE,
    INTENCIONES,
)
from src.entrenar import entrenar_modelos_en_memoria
from src.predecir import (
    calcular_margen_confianza,
    predecir_intencion,
    predecir_intencion_con_modelos,
)


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


def dividir_train_test(
    muestras: list[tuple[str, Path]],
    test_size: float = HOLDOUT_TEST_SIZE,
    random_state: int = HOLDOUT_RANDOM_STATE,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """
    Divide muestras (intencion, ruta) en train/test de forma estratificada.
    """
    if len(muestras) < 2:
        raise ValueError("Se necesitan al menos 2 audios para dividir train/test.")

    etiquetas = [intencion for intencion, _ in muestras]
    rutas = [ruta for _, ruta in muestras]

    train_rutas, test_rutas, train_y, test_y = train_test_split(
        rutas,
        etiquetas,
        test_size=test_size,
        random_state=random_state,
        stratify=etiquetas,
    )

    train = list(zip(train_y, train_rutas))
    test = list(zip(test_y, test_rutas))
    return train, test


def _muestras_a_particion(muestras: list[tuple[str, Path]]) -> dict[str, list[Path]]:
    """Agrupa rutas WAV por intencion."""
    particion: dict[str, list[Path]] = {}
    for intencion, ruta in muestras:
        particion.setdefault(intencion, []).append(ruta)
    return particion


def evaluar_holdout(
    carpeta_dataset: Path = CARPETA_DATASET,
    intenciones: list[str] | None = None,
    test_size: float = HOLDOUT_TEST_SIZE,
    random_state: int = HOLDOUT_RANDOM_STATE,
    verbose: bool = False,
) -> dict:
    """
    Entrena en memoria con train (1-test_size) y evalua solo en test.

    Metrica mas honesta que evaluar sobre el mismo set de entrenamiento.
    """
    if intenciones is None:
        intenciones = INTENCIONES

    muestras = _listar_audios_por_intencion(carpeta_dataset, intenciones)
    if not muestras:
        raise ValueError(f"No hay audios en {carpeta_dataset}")

    train, test = dividir_train_test(muestras, test_size=test_size, random_state=random_state)
    archivos_train = _muestras_a_particion(train)

    print(
        f"\n  Particion holdout: train={len(train)} audios, "
        f"test={len(test)} audios ({100 * test_size:.0f}% test)"
    )

    escalador, modelos = entrenar_modelos_en_memoria(
        archivos_train,
        intenciones=intenciones,
        verbose=verbose,
    )

    reales = []
    predichas = []
    margenes = []
    rutas = []

    for intencion_real, ruta in test:
        intencion_pred, puntajes = predecir_intencion_con_modelos(
            ruta,
            escalador,
            modelos,
            verbose=verbose,
        )
        margen, _, _ = calcular_margen_confianza(puntajes)

        reales.append(intencion_real)
        predichas.append(intencion_pred)
        margenes.append(margen)
        rutas.append(str(ruta))

    return {
        "intenciones": intenciones,
        "reales": reales,
        "predichas": predichas,
        "margenes": margenes,
        "rutas": rutas,
        "reales_arr": np.array(reales),
        "predichas_arr": np.array(predichas),
        "holdout": True,
        "train_size": len(train),
        "test_size": len(test),
    }


def listar_audios_fallidos(resultado: dict) -> list[dict]:
    """
    Devuelve audios mal clasificados con metadatos para re-grabar.

    Cada item: ruta, real, predicho, margen, nombre.
    """
    fallos = []
    for i in range(len(resultado["reales"])):
        if resultado["reales"][i] == resultado["predichas"][i]:
            continue
        fallos.append(
            {
                "ruta": Path(resultado["rutas"][i]),
                "real": resultado["reales"][i],
                "predicho": resultado["predichas"][i],
                "margen": resultado["margenes"][i],
                "nombre": Path(resultado["rutas"][i]).name,
            }
        )
    return fallos


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
        "holdout": False,
    }


def calcular_accuracy(resultado: dict) -> float:
    """Fraccion de aciertos sobre el total de audios evaluados."""
    reales = resultado["reales"]
    predichas = resultado["predichas"]
    if not reales:
        return 0.0
    aciertos = sum(1 for real, pred in zip(reales, predichas) if real == pred)
    return aciertos / len(reales)


def imprimir_matriz_confusion(
    reales: list[str],
    predichas: list[str],
    intenciones: list[str],
) -> np.ndarray:
    """Imprime la matriz de confusion alineada con las etiquetas del proyecto."""
    matriz = confusion_matrix(reales, predichas, labels=intenciones)
    encabezado = "              " + "  ".join(f"{i[:6]:>6s}" for i in intenciones)
    print(encabezado)
    for idx, intencion in enumerate(intenciones):
        fila = "  ".join(f"{v:6d}" for v in matriz[idx])
        print(f"  {intencion:12s}  {fila}")
    return matriz


def mostrar_precision_por_intencion(resultado: dict) -> None:
    """Imprime aciertos/total por intencion."""
    intenciones = resultado["intenciones"]
    reales = resultado["reales"]
    predichas = resultado["predichas"]

    print("\n--- Precision por intencion ---")
    for intencion in intenciones:
        indices = [i for i, r in enumerate(reales) if r == intencion]
        if not indices:
            print(f"  {intencion:12s}  (sin audios)")
            continue
        aciertos = sum(1 for i in indices if predichas[i] == intencion)
        total = len(indices)
        print(f"  {intencion:12s}  {aciertos:3d}/{total:3d}  ({100 * aciertos / total:.0f}%)")


def _tiene_entorno_grafico() -> bool:
    """
    True si matplotlib puede abrir una ventana.

    En macOS no existe DISPLAY (usa Cocoa). En Linux hace falta X11/Wayland.
    """
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def graficar_matriz_confusion(
    reales: list[str],
    predichas: list[str],
    intenciones: list[str],
    accuracy: float,
    *,
    mostrar_ventana: bool = True,
    ruta_salida: Path | None = None,
) -> Path:
    """
    Dibuja la matriz de confusion con matplotlib (heatmap + valores en celdas).
    Guarda PNG y, si hay GUI, abre una ventana hasta que el usuario la cierre.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "matplotlib no esta instalado. Activa el venv y ejecuta:\n"
            "  pip install -r requirements.txt"
        ) from error

    matriz = confusion_matrix(reales, predichas, labels=intenciones)
    ruta_png = ruta_salida or (CARPETA_MODELOS / "matriz_confusion.png")
    ruta_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    imagen = ax.imshow(matriz, cmap="YlGnBu", interpolation="nearest")

    ax.set_xticks(range(len(intenciones)))
    ax.set_yticks(range(len(intenciones)))
    ax.set_xticklabels(intenciones, rotation=35, ha="right")
    ax.set_yticklabels(intenciones)
    ax.set_xlabel("Prediccion del modelo", fontsize=11)
    ax.set_ylabel("Etiqueta real (dataset)", fontsize=11)
    ax.set_title(
        f"Matriz de confusion — LPC + GaussianHMM\n"
        f"Accuracy global: {accuracy:.1%}  |  Audios: {len(reales)}",
        fontsize=13,
        fontweight="bold",
    )

    umbral_texto = matriz.max() / 2 if matriz.max() > 0 else 0
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            valor = int(matriz[fila, columna])
            color_texto = "white" if valor > umbral_texto else "black"
            ax.text(
                columna,
                fila,
                str(valor),
                ha="center",
                va="center",
                color=color_texto,
                fontsize=12,
                fontweight="bold",
            )

    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("Cantidad de audios", rotation=270, labelpad=18)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=160, bbox_inches="tight")

    if mostrar_ventana and _tiene_entorno_grafico():
        print("\n  Abriendo grafica... (cierra la ventana para continuar)")
        plt.show(block=True)
    else:
        print("\n  Sin entorno grafico: solo se guardo la imagen PNG.")

    plt.close(fig)
    return ruta_png


def mostrar_fallos(resultado: dict, intenciones_filtrar: list[str] | None = None) -> None:
    """Imprime audios mal clasificados (opcionalmente filtrados por intencion real)."""
    fallos = listar_audios_fallidos(resultado)
    if intenciones_filtrar is not None:
        permitidas = set(intenciones_filtrar)
        fallos = [f for f in fallos if f["real"] in permitidas]

    print("\n--- Audios mal clasificados ---")
    if not fallos:
        print("  Ninguno.")
        return

    for item in fallos:
        print(
            f"  {item['nombre']}: {item['real']} -> {item['predicho']}  "
            f"(margen={item['margen']:.1f})"
        )


def mostrar_reporte_demo(
    resultado: dict,
    *,
    mostrar_grafica: bool = True,
    titulo: str = "EVALUACION OFFLINE DEL MODELO",
) -> Path | None:
    """
    Resumen corto para la presentacion: precision, accuracy y matriz grafica.
    Sin listado de errores ni reporte sklearn (mas legible en demo).
    """
    intenciones = resultado["intenciones"]
    reales = resultado["reales"]
    predichas = resultado["predichas"]
    accuracy = calcular_accuracy(resultado)

    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)
    if resultado.get("holdout"):
        print(
            f"  Train: {resultado.get('train_size', '?')} audios  |  "
            f"Test: {resultado.get('test_size', len(reales))} audios"
        )
    print(f"  Audios evaluados: {len(reales)}")
    print(f"  Accuracy global : {100 * accuracy:.1f}%")

    mostrar_precision_por_intencion(resultado)

    if mostrar_grafica:
        print("\n--- Matriz de confusion (matplotlib) ---")
        ruta = graficar_matriz_confusion(
            reales,
            predichas,
            intenciones,
            accuracy,
        )
        print(f"  Imagen guardada en: {ruta}")
        return ruta

    print("\n--- Matriz de confusion (texto) ---")
    print("  (filas = etiqueta real, columnas = prediccion)")
    imprimir_matriz_confusion(reales, predichas, intenciones)
    return None


def mostrar_reporte(
    resultado: dict,
    mostrar_grafica: bool = False,
    titulo: str | None = None,
) -> None:
    """Imprime precision por intencion, errores y matriz de confusion."""
    intenciones = resultado["intenciones"]
    reales = resultado["reales"]
    predichas = resultado["predichas"]
    margenes = resultado["margenes"]
    rutas = resultado["rutas"]

    if titulo is None:
        titulo = (
            "EVALUACION HOLDOUT (train/test)"
            if resultado.get("holdout")
            else "EVALUACION DEL DATASET"
        )

    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)
    if resultado.get("holdout"):
        print(
            f"  Train: {resultado.get('train_size', '?')} audios  |  "
            f"Test: {resultado.get('test_size', len(reales))} audios"
        )
    print(f"  Audios evaluados: {len(reales)}")

    mostrar_precision_por_intencion(resultado)

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

    if mostrar_grafica:
        accuracy = calcular_accuracy(resultado)
        print("\n--- Matriz de confusion (matplotlib) ---")
        ruta = graficar_matriz_confusion(
            reales,
            predichas,
            intenciones,
            accuracy,
        )
        print(f"  Imagen guardada en: {ruta}")
    else:
        print("\n--- Matriz de confusion ---")
        imprimir_matriz_confusion(reales, predichas, intenciones)

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
    parser.add_argument(
        "--grafica",
        action="store_true",
        help="Muestra la matriz de confusion con matplotlib",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help=(
            "Evalua con particion 80/20: entrena en memoria con train "
            "y mide accuracy en test (mas honesto academicamente)"
        ),
    )
    parser.add_argument(
        "--solo-fallos",
        action="store_true",
        help="Solo lista audios mal clasificados (sin reporte completo)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detalle LPC al evaluar holdout",
    )
    args = parser.parse_args()

    if args.holdout:
        resultado = evaluar_holdout(
            carpeta_dataset=args.carpeta_dataset,
            verbose=args.verbose,
        )
        ruta_grafica = CARPETA_MODELOS / "matriz_confusion_holdout.png"
    else:
        resultado = evaluar_dataset(carpeta_dataset=args.carpeta_dataset)
        ruta_grafica = CARPETA_MODELOS / "matriz_confusion.png"

    if args.solo_fallos:
        mostrar_fallos(resultado)
        return

    if args.grafica and args.holdout:
        mostrar_reporte(resultado, mostrar_grafica=False)
        accuracy = calcular_accuracy(resultado)
        print("\n--- Matriz de confusion holdout (matplotlib) ---")
        graficar_matriz_confusion(
            resultado["reales"],
            resultado["predichas"],
            resultado["intenciones"],
            accuracy,
            ruta_salida=ruta_grafica,
        )
        print(f"  Imagen guardada en: {ruta_grafica}")
    else:
        mostrar_reporte(resultado, mostrar_grafica=args.grafica)


if __name__ == "__main__":
    main()
