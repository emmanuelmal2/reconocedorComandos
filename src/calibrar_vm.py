"""
Calibracion del microfono en la VM (o cualquier maquina de la demo).

El dataset del repo se grabo en otro equipo/microfono. En vivo los comandos
(listar, memoria, procesos) se confunden si no re-grabas con EL MISMO micro
que usaras en la presentacion.

Este script:
    1. Graba las 4 frases activas x N repeticiones por hablante
    2. Re-entrena los modelos (8 HMM: 4 intenciones x 2 hablantes)
    3. Muestra accuracy offline

Uso en la VM antes de la expo (recomendado: los dos hablantes):
    python3 -m src.calibrar_vm --todos

Solo una persona presenta:
    python3 -m src.calibrar_vm --hablante emmanuel
    python3 -m src.asistente --demo --hablante emmanuel
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


def calibrar_hablante(
    hablante: str,
    repeticiones: int = REPETICIONES_POR_FRASE,
) -> None:
    """Graba las frases activas de un hablante en este microfono."""
    total = len(INTENCIONES) * repeticiones
    print("\n" + "=" * 60)
    print(f"  CALIBRACION — hablante: {hablante}")
    print("=" * 60)
    print(f"  Intenciones : {', '.join(INTENCIONES)}")
    print(f"  Repeticiones: {repeticiones} por frase ({total} WAV)")
    print()
    print("  Usa el mismo microfono y distancia que en la expo.")
    print("  Di cada frase EXACTA al oir el beep.")
    print("=" * 60)

    if not _confirmar(f"¿{hablante} listo para grabar?"):
        print(f"  Calibracion de {hablante} omitida.")
        return

    for intencion in INTENCIONES:
        numero_frase = FRASE_INDICE_POR_INTENCION[intencion]
        frase = FRASES[intencion][0]
        print(f'\n>>> [{hablante}] {intencion} — di: "{frase}"')
        for repeticion in range(1, repeticiones + 1):
            grabar_repeticion(hablante, intencion, numero_frase, frase, repeticion)

    print(f"\n  [ok] {hablante}: {total} audios guardados en dataset/")
    print(f"  (Los audios de otros hablantes en dataset/ no se modifican.)")


def entrenar_y_evaluar(*, evaluar: bool = True) -> None:
    print("\n  Re-entrenando modelos (emmanuel + elioth en este microfono)...")
    subprocess.run([sys.executable, "-m", "src.entrenar"], check=True)
    if evaluar:
        print("\n  Evaluacion offline (deberia quedar cerca de 97%):")
        subprocess.run([sys.executable, "-m", "src.evaluar"], check=False)


def calibrar(
    hablantes: list[str],
    repeticiones: int = REPETICIONES_POR_FRASE,
    *,
    entrenar: bool = True,
    evaluar: bool = True,
) -> None:
    """Graba uno o varios hablantes y opcionalmente re-entrena."""
    if not hablantes:
        print("  Nada que calibrar.")
        return

    print("\n" + "#" * 60)
    print("  CALIBRACION DE MICROFONO PARA LA DEMO")
    print("#" * 60)
    print(f"  Hablantes   : {', '.join(hablantes)}")
    print(f"  Total WAV   : {len(hablantes) * len(INTENCIONES) * repeticiones}")

    for hablante in hablantes:
        calibrar_hablante(hablante, repeticiones=repeticiones)

    if entrenar:
        entrenar_y_evaluar(evaluar=evaluar)

    print("\n" + "=" * 60)
    print("  CALIBRACION LISTA")
    print("=" * 60)
    if len(hablantes) == 1:
        h = hablantes[0]
        print(f"  Audios de Elioth en el repo se conservan (entrenamiento / evaluacion offline).")
        print(f"  En la expo en vivo habla {h} y usa solo sus modelos:")
        print(f"    python3 -m src.asistente --demo --hablante {h}")
        print(f"  o: ./scripts/iniciar_demo_linux.sh")
    else:
        print("  Demo con AMBAS voces (elige el mejor HMM por intencion):")
        print("    python3 -m src.asistente --demo")
        print("  o: ./scripts/iniciar_demo_linux.sh")
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
        help="Calibrar solo un hablante (emmanuel o elioth)",
    )
    parser.add_argument(
        "--todos",
        action="store_true",
        help="Calibrar emmanuel y elioth (recomendado si presentan los dos)",
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

    if args.todos and args.hablante:
        parser.error("Usa --todos o --hablante, no ambos.")

    if args.todos:
        hablantes = list(HABLANTES)
    elif args.hablante:
        hablantes = [args.hablante]
    else:
        hablantes = [HABLANTE_PREDETERMINADO]

    calibrar(
        hablantes,
        repeticiones=args.repeticiones,
        entrenar=not args.sin_entrenar,
        evaluar=not args.sin_evaluar,
    )


if __name__ == "__main__":
    main()
