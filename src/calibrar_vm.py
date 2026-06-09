"""
Calibracion del microfono en la VM (o cualquier maquina de la demo).

El dataset del repo se grabo en otro equipo/microfono. En vivo los comandos
(listar, memoria, procesos) se confunden si no re-grabas con EL MISMO micro
que usaras en la presentacion.

Este script:
    1. Graba las 4 frases activas x N repeticiones para un hablante
    2. Re-entrena los modelos
    3. Muestra accuracy offline

Uso en la VM (5 minutos antes de la expo):
    python3 -m src.calibrar_vm --hablante emmanuel
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.configuracion import (
    FRASES,
    FRASE_INDICE_POR_INTENCION,
    HABLANTES,
    HABLANTE_PREDETERMINADO,
    INTENCIONES,
    REPETICIONES_POR_FRASE,
)
from src.grabar import grabar_repeticion


def calibrar(
    hablante: str,
    repeticiones: int = REPETICIONES_POR_FRASE,
    *,
    entrenar: bool = True,
    evaluar: bool = True,
) -> None:
    """Graba frases activas en este microfono y opcionalmente re-entrena."""
    total = len(INTENCIONES) * repeticiones
    print("\n" + "=" * 60)
    print("  CALIBRACION DE MICROFONO PARA LA DEMO")
    print("=" * 60)
    print(f"  Hablante    : {hablante}")
    print(f"  Intenciones : {', '.join(INTENCIONES)}")
    print(f"  Repeticiones: {repeticiones} por frase")
    print(f"  Total WAV   : {total}")
    print()
    print("  IMPORTANTE: usa el mismo microfono y distancia que en la expo.")
    print("  Di cada frase EXACTA cuando suene el beep.")
    print("=" * 60)

    if not _confirmar("¿Listo para empezar?"):
        print("  Calibracion cancelada.")
        return

    for intencion in INTENCIONES:
        numero_frase = FRASE_INDICE_POR_INTENCION[intencion]
        frase = FRASES[intencion][0]
        print(f"\n>>> Intencion: {intencion} — di: \"{frase}\"")
        for repeticion in range(1, repeticiones + 1):
            grabar_repeticion(hablante, intencion, numero_frase, frase, repeticion)

    if entrenar:
        print("\n  Re-entrenando modelos con tus audios de esta maquina...")
        subprocess.run([sys.executable, "-m", "src.entrenar"], check=True)

    if evaluar:
        print("\n  Evaluacion offline (deberia quedar cerca de 97%):")
        subprocess.run([sys.executable, "-m", "src.evaluar"], check=False)

    print("\n" + "=" * 60)
    print("  CALIBRACION LISTA")
    print("=" * 60)
    print(f"  Arranca la demo con TU voz y modelos de {hablante}:")
    print(f"    python3 -m src.asistente --demo --hablante {hablante}")
    print("=" * 60)


def _confirmar(mensaje: str) -> bool:
    while True:
        respuesta = input(f"  {mensaje} (s/n): ").strip().lower()
        if respuesta in ("s", "si", "sí", "y", "yes", ""):
            return True
        if respuesta in ("n", "no"):
            return False
        print("  Responde s o n.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graba audios en ESTE microfono y re-entrena (calibracion VM)."
    )
    parser.add_argument(
        "--hablante",
        choices=HABLANTES,
        default=HABLANTE_PREDETERMINADO,
        help=f"Hablante que presentara (default: {HABLANTE_PREDETERMINADO})",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=REPETICIONES_POR_FRASE,
        metavar="N",
        help=f"Repeticiones por frase (1-{REPETICIONES_POR_FRASE}, default: 5)",
    )
    parser.add_argument(
        "--sin-entrenar",
        action="store_true",
        help="Solo grabar WAV, no ejecutar src.entrenar",
    )
    parser.add_argument(
        "--sin-evaluar",
        action="store_true",
        help="No ejecutar src.evaluar al final",
    )
    args = parser.parse_args()
    if not 1 <= args.repeticiones <= REPETICIONES_POR_FRASE:
        parser.error(f"--repeticiones debe estar entre 1 y {REPETICIONES_POR_FRASE}")

    calibrar(
        args.hablante,
        repeticiones=args.repeticiones,
        entrenar=not args.sin_entrenar,
        evaluar=not args.sin_evaluar,
    )


if __name__ == "__main__":
    main()
