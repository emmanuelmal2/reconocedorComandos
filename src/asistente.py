"""
Asistente de voz con activacion por frase.

Flujo:
    1. Grabar audio corto y predecir intencion (buscar "activacion")
    2. Si no es activacion -> no hacer nada, volver a escuchar
    3. Si es activacion -> grabar segundo audio y predecir comando
    4. Solo intenciones ejecutables en fase de comando
    5. Si el margen de confianza es bajo -> pedir repetir
    6. Ejecutar Bash solo si la intencion esta en INTENCIONES_EJECUTABLES
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.configuracion import (
    FRASES,
    INTENCIONES,
    INTENCIONES_EJECUTABLES,
    MARGEN_MINIMO_ACTIVACION,
    MARGEN_MINIMO_COMANDO,
    MAX_REINTENTOS_COMANDO,
)
from src.ejecutar import ejecutar_intencion
from src.grabar import grabar_audio
from src.predecir import (
    calcular_margen_confianza,
    mostrar_puntajes,
    predecir_intencion,
    predecir_intencion_desde_senal,
)


def mostrar_frases_comando() -> None:
    """Muestra frases de entrenamiento para orientar al usuario."""
    print("\n  Frases que el modelo conoce (di una parecida):")
    for intencion in INTENCIONES_EJECUTABLES:
        ejemplos = FRASES[intencion][:2]
        texto = '" | "'.join(ejemplos)
        print(f"    {intencion:10s}: \"{texto}\"")


def mostrar_frases_red() -> None:
    """Frases especificas de red (intencion que suele fallar en vivo)."""
    print("\n  Para RED, prueba exactamente una de estas:")
    for i, frase in enumerate(FRASES["red"], start=1):
        print(f"    {i}. \"{frase}\"")


def escuchar_activacion(verbose: bool = False) -> bool:
    """
    Graba audio y verifica activacion con margen minimo de confianza.

    Returns:
        True si se detecto activacion confiable.
    """
    print("\n=== Escuchando frase de activacion ===")
    print('  Di por ejemplo: "oye computadora" o "hola computadora"')
    input("  Presiona Enter para grabar...")

    senal = grabar_audio()
    intencion, puntajes = predecir_intencion_desde_senal(
        senal,
        intenciones_permitidas=INTENCIONES,
        verbose=verbose,
    )
    margen, primero, segundo = calcular_margen_confianza(puntajes)
    mostrar_puntajes(puntajes, intencion)

    if intencion != "activacion":
        print(f"  Activacion no detectada (prediccion: {intencion}).")
        return False

    if margen < MARGEN_MINIMO_ACTIVACION:
        print(
            f"  Activacion ambigua (margen {margen:.1f} < {MARGEN_MINIMO_ACTIVACION}). "
            f"Segundo lugar: {segundo}. Repite mas claro."
        )
        return False

    print(f"  Activacion confiable (margen {margen:.1f}).")
    return True


def escuchar_comando(verbose: bool = False) -> tuple[str | None, dict[str, float]]:
    """
    Graba y predice comando con reintentos si la confianza es baja.

    Returns:
        (intencion, puntajes) o (None, {}) si no hubo prediccion confiable.
    """
    mostrar_frases_comando()
    mostrar_frases_red()

    for intento in range(1, MAX_REINTENTOS_COMANDO + 1):
        print(f"\n=== Escuchando comando (intento {intento}/{MAX_REINTENTOS_COMANDO}) ===")
        input("  Presiona Enter y habla al instante...")
        senal = grabar_audio()

        intencion, puntajes = predecir_intencion_desde_senal(
            senal,
            intenciones_permitidas=INTENCIONES_EJECUTABLES,
            verbose=verbose,
        )
        margen, primero, segundo = calcular_margen_confianza(puntajes)
        mostrar_puntajes(puntajes, intencion)

        if margen >= MARGEN_MINIMO_COMANDO:
            print(f"  Confianza OK (margen {margen:.1f}, 2.º lugar: {segundo}).")
            return intencion, puntajes

        print(
            f"  [aviso] Poca confianza (margen {margen:.1f} < {MARGEN_MINIMO_COMANDO}). "
            f"Gano '{primero}' pero '{segundo}' estaba muy cerca."
        )
        if intento < MAX_REINTENTOS_COMANDO:
            print("  Repite la frase, igual que al grabar el dataset.")
            if intencion == "red" or segundo == "red":
                mostrar_frases_red()

    print("  No se obtuvo un comando confiable. Cancelando este ciclo.")
    return None, {}


def probar_comando_desde_archivo(ruta_audio: Path, verbose: bool = False) -> None:
    """Prueba prediccion de comando sin microfono (util para depurar red, etc.)."""
    print(f"\n=== Prueba offline: {ruta_audio} ===")
    intencion, puntajes = predecir_intencion(
        ruta_audio,
        intenciones_permitidas=INTENCIONES_EJECUTABLES,
        verbose=verbose,
    )
    margen, _, segundo = calcular_margen_confianza(puntajes)
    mostrar_puntajes(puntajes, intencion)
    print(f"  Prediccion: {intencion}  |  margen: {margen:.1f}  |  2.º: {segundo}")

    if intencion in INTENCIONES_EJECUTABLES:
        ejecutar_intencion(intencion)


def ejecutar_flujo_asistente(verbose: bool = False, una_vez: bool = False) -> None:
    """Bucle principal del asistente de voz."""
    print("Asistente de voz iniciado.")
    print("Intenciones ejecutables:", ", ".join(INTENCIONES_EJECUTABLES))
    print(
        f"Margenes minimos: activacion={MARGEN_MINIMO_ACTIVACION}, "
        f"comando={MARGEN_MINIMO_COMANDO}"
    )

    while True:
        if not escuchar_activacion(verbose=verbose):
            if una_vez:
                break
            continue

        print("\nActivacion detectada. Di un comando.")
        intencion, puntajes = escuchar_comando(verbose=verbose)

        if intencion is None:
            if una_vez:
                break
            print("\n--- Volviendo a modo escucha ---")
            continue

        if intencion in INTENCIONES_EJECUTABLES:
            print(f"\nComando reconocido: {intencion}")
            codigo = ejecutar_intencion(intencion)
            if codigo != 0:
                print(f"[aviso] El comando termino con codigo {codigo}.")
        else:
            print(f"\nIntencion '{intencion}' no esta en la lista de comandos ejecutables.")

        if una_vez:
            break

        print("\n--- Volviendo a modo escucha ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Asistente de voz con activacion.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra detalle de extraccion LPC y puntajes",
    )
    parser.add_argument(
        "--una-vez",
        action="store_true",
        help="Solo un ciclo activacion + comando y termina",
    )
    parser.add_argument(
        "--probar-comando",
        type=Path,
        metavar="AUDIO.wav",
        help="Prueba un comando desde archivo WAV (sin microfono)",
    )
    args = parser.parse_args()

    if args.probar_comando:
        probar_comando_desde_archivo(args.probar_comando, verbose=args.verbose)
        return

    ejecutar_flujo_asistente(verbose=args.verbose, una_vez=args.una_vez)


if __name__ == "__main__":
    main()
