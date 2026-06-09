"""
Prueba rapida del microfono: graba, predice y guarda el WAV.

Sirve para depurar en la VM sin pasar por todo el bucle del asistente.

    python3 -m src.probar_microfono --hablante emmanuel
    python3 -m src.probar_microfono --hablante emmanuel --comandos
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.configuracion import (
    CARPETA_MODELOS,
    FRASES,
    HABLANTES,
    HABLANTE_PREDETERMINADO,
    INTENCIONES,
    INTENCIONES_EJECUTABLES,
)
from src.grabar import grabar_audio_asistente, guardar_audio
from src.predecir import calcular_margen_confianza, mostrar_puntajes, predecir_intencion_desde_senal


def main() -> None:
    parser = argparse.ArgumentParser(description="Graba una frase y muestra la prediccion.")
    parser.add_argument(
        "--hablante",
        choices=HABLANTES,
        default=HABLANTE_PREDETERMINADO,
        help="Modelos a usar en prediccion",
    )
    parser.add_argument(
        "--comandos",
        action="store_true",
        help="Solo intenciones ejecutables (sin activacion)",
    )
    parser.add_argument(
        "--guardar",
        type=Path,
        default=CARPETA_MODELOS / "ultimo_microfono.wav",
        help="Ruta donde guardar el WAV grabado",
    )
    args = parser.parse_args()

    intenciones = INTENCIONES_EJECUTABLES if args.comandos else INTENCIONES
    modo = "comando" if args.comandos else "activacion o comando"

    print("\n" + "=" * 60)
    print("  PRUEBA DE MICROFONO")
    print("=" * 60)
    print(f"  Hablante modelos: {args.hablante}")
    print(f"  Modo: {modo}")
    if args.comandos:
        print("\n  Frases de comando:")
        for i in INTENCIONES_EJECUTABLES:
            print(f'    {i}: "{FRASES[i][0]}"')
    else:
        print(f'\n  Activacion: "{FRASES["activacion"][0]}"')
    print("=" * 60)

    input("\n  Enter para grabar (beep = habla ya)...")
    senal = grabar_audio_asistente(aviso=True)
    guardar_audio(args.guardar, senal)
    print(f"  WAV guardado: {args.guardar}")

    intencion, puntajes = predecir_intencion_desde_senal(
        senal,
        intenciones_permitidas=intenciones,
        hablante=args.hablante,
        ruta_debug_wav=args.guardar,
    )
    margen, _, segundo = calcular_margen_confianza(puntajes)
    mostrar_puntajes(puntajes, intencion)
    print(f"\n  Prediccion: {intencion}")
    print(f"  Margen 1.-2.: {margen:.1f}  |  2.º: {segundo}")
    print(f'  Frase asociada: "{FRASES.get(intencion, ["?"])[0]}"')
    print("\n  Mismo archivo con predecir offline:")
    excluir = "activacion" if args.comandos else None
    cmd = f"python3 -m src.predecir {args.guardar} --hablante {args.hablante}"
    if excluir:
        cmd += f" --excluir {excluir}"
    print(f"    {cmd}")


if __name__ == "__main__":
    main()
