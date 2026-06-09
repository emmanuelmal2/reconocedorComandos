"""
Re-graba audios que el modelo clasifica mal (util para memoria, listar, etc.).

Flujo:
    1. Evaluar el dataset con los modelos actuales
    2. Listar WAV mal clasificados (filtrados por hablante e intencion)
    3. Re-grabar solo esas tomas (sobrescribe el archivo)
    4. Recordar ejecutar: python -m src.entrenar
"""

from __future__ import annotations

import argparse
import re

from src.configuracion import FRASES, HABLANTES, INTENCIONES_REGRABAR_SUGERIDAS
from src.evaluar import evaluar_dataset, listar_audios_fallidos
from src.grabar import _confirmar_si_no, grabar_repeticion

_PATRON_AUDIO = re.compile(
    r"^(?P<intencion>\w+)_(?P<hablante>emmanuel|elioth)_frase(?P<frase>\d{2})_rep(?P<rep>\d{2})\.wav$"
)


def _parsear_nombre_audio(nombre: str) -> dict | None:
    """Extrae intencion, hablante, frase y repeticion del nombre de archivo."""
    coincidencia = _PATRON_AUDIO.match(nombre)
    if not coincidencia:
        return None
    return {
        "intencion": coincidencia.group("intencion"),
        "hablante": coincidencia.group("hablante"),
        "frase": int(coincidencia.group("frase")),
        "repeticion": int(coincidencia.group("rep")),
    }


def _filtrar_fallos(
    fallos: list[dict],
    hablante: str | None,
    intenciones: list[str],
) -> list[dict]:
    """Filtra fallos por hablante (en el nombre) e intencion real."""
    permitidas = set(intenciones)
    filtrados = []

    for item in fallos:
        if item["real"] not in permitidas:
            continue
        metadatos = _parsear_nombre_audio(item["nombre"])
        if metadatos is None:
            continue
        if hablante and metadatos["hablante"] != hablante:
            continue
        item = {**item, **metadatos}
        filtrados.append(item)

    return filtrados


def regrabar_fallos(
    hablante: str | None = None,
    intenciones: list[str] | None = None,
    forzar: bool = False,
) -> int:
    """
    Re-graba audios mal clasificados.

    Returns:
        Cantidad de audios re-grabados.
    """
    if intenciones is None:
        intenciones = INTENCIONES_REGRABAR_SUGERIDAS

    print("\n" + "=" * 60)
    print("  RE-GRABACION DE AUDIOS MAL CLASIFICADOS")
    print("=" * 60)

    resultado = evaluar_dataset()
    fallos = _filtrar_fallos(listar_audios_fallidos(resultado), hablante, intenciones)

    if not fallos:
        print("\n  No hay fallos en el alcance indicado. El modelo ya acierta esos audios.")
        return 0

    print(f"\n  Hablante filtro   : {hablante or 'todos'}")
    print(f"  Intenciones       : {', '.join(intenciones)}")
    print(f"  Audios a re-grabar: {len(fallos)}\n")

    for item in fallos:
        print(
            f"  - {item['nombre']}: {item['real']} -> {item['predicho']}  "
            f"(margen={item['margen']:.1f})"
        )

    if not forzar and not _confirmar_si_no("\nQuieres re-grabar estos audios?"):
        print("\n  Cancelado. No se modifico el dataset.")
        return 0

    regrabados = 0
    for item in fallos:
        frase_texto = FRASES[item["intencion"]][item["frase"] - 1]
        print("\n" + "#" * 60)
        print(f"  Re-grabando: {item['nombre']}")
        print(f"  Frase       : \"{frase_texto}\"")
        print("#" * 60)

        grabar_repeticion(
            hablante=item["hablante"],
            intencion=item["intencion"],
            numero_frase=item["frase"],
            frase=frase_texto,
            repeticion=item["repeticion"],
        )
        regrabados += 1

    print("\n" + "=" * 60)
    print(f"  Re-grabacion completada: {regrabados} audios")
    print("  Siguiente paso: python -m src.entrenar")
    print("=" * 60)
    return regrabados


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-graba audios que el clasificador predice mal."
    )
    parser.add_argument(
        "--hablante",
        choices=HABLANTES,
        help="Solo audios de este hablante (por defecto: todos)",
    )
    parser.add_argument(
        "--intenciones",
        nargs="+",
        choices=list(FRASES.keys()),
        default=INTENCIONES_REGRABAR_SUGERIDAS,
        help="Intenciones reales a considerar (default: memoria listar)",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Re-grabar sin confirmacion",
    )
    args = parser.parse_args()
    regrabar_fallos(
        hablante=args.hablante,
        intenciones=args.intenciones,
        forzar=args.forzar,
    )


if __name__ == "__main__":
    main()
