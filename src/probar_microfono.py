"""
Prueba rapida del microfono: graba, predice y guarda el WAV.

    python3 -m src.probar_microfono --activacion          # probar "oye computadora"
    python3 -m src.probar_microfono --comandos            # probar listar/memoria/procesos
    python3 -m src.probar_microfono --hablante emmanuel
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
    MARGEN_MINIMO_ACTIVACION,
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
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument(
        "--activacion",
        action="store_true",
        help='Solo activacion ("oye computadora")',
    )
    modo.add_argument(
        "--comandos",
        action="store_true",
        help="Solo comandos (listar, memoria, procesos)",
    )
    parser.add_argument(
        "--guardar",
        type=Path,
        default=CARPETA_MODELOS / "ultimo_microfono.wav",
        help="Ruta donde guardar el WAV grabado",
    )
    args = parser.parse_args()

    if args.activacion:
        intenciones = ["activacion"]
        titulo_modo = "activacion"
        frase_hint = FRASES["activacion"][0]
    elif args.comandos:
        intenciones = INTENCIONES_EJECUTABLES
        titulo_modo = "comando (sin activacion)"
        frase_hint = "listar archivos | muestra la memoria | ver procesos"
    else:
        intenciones = INTENCIONES
        titulo_modo = "cualquier intencion"
        frase_hint = "cualquier frase del asistente"

    print("\n" + "=" * 60)
    print("  PRUEBA DE MICROFONO")
    print("=" * 60)
    print(f"  Hablante : {args.hablante}")
    print(f"  Modo     : {titulo_modo}")
    print(f'  Di       : "{frase_hint}"')
    if args.comandos:
        print("\n  (Si dices oye computadora aqui, el modo --comandos NO aplica — usa --activacion)")
    print("=" * 60)

    input("\n  [Enter] → graba 3 s → habla")
    senal = grabar_audio_asistente(aviso=True)
    guardar_audio(args.guardar, senal)
    print(f"  WAV guardado: {args.guardar}")

    # En modo comando: detectar si dijo activacion y avisar (no forzar un falso comando)
    if args.comandos:
        act, pts_act = predecir_intencion_desde_senal(
            senal,
            intenciones_permitidas=INTENCIONES,
            hablante=args.hablante,
            ruta_debug_wav=args.guardar,
        )
        margen_act, _, _ = calcular_margen_confianza(pts_act)
        if act == "activacion" and margen_act >= MARGEN_MINIMO_ACTIVACION:
            print("\n  [aviso] Parece frase de ACTIVACION, no un comando.")
            print('  Dijiste algo como "oye computadora".')
            print("  Prueba activacion con:")
            print("    python3 -m src.probar_microfono --activacion --hablante emmanuel")
            print("  O di un comando: listar archivos / muestra la memoria / ver procesos")
            return

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


if __name__ == "__main__":
    main()
