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
import sys
from pathlib import Path

from src.configuracion import (
    FRASES_DEMO,
    INTENCIONES,
    INTENCIONES_EJECUTABLES,
    MARGEN_MINIMO_ACTIVACION,
    MARGEN_MINIMO_COMANDO,
    MAX_REINTENTOS_COMANDO,
)
from src import consola
from src.demo_linux import (
    abrir_terminal_activacion,
    ejecutar_comando_en_terminal_nueva,
    es_linux,
)
from src.evaluar import evaluar_dataset, mostrar_reporte_demo
from src.ejecutar import ejecutar_intencion
from src.grabar import grabar_audio
from src.predecir import (
    calcular_margen_confianza,
    mostrar_puntajes,
    predecir_intencion,
    predecir_intencion_desde_senal,
)


def mostrar_frases_comando() -> None:
    """Muestra la frase exacta que debe decir el usuario por comando."""
    print("\n  Di exactamente esta frase para cada comando:")
    for intencion in INTENCIONES_EJECUTABLES:
        frase = FRASES_DEMO[intencion][0]
        print(f'    {intencion:10s}: "{frase}"')


def escuchar_activacion(verbose: bool = False, demo: bool = False) -> bool:
    """
    Graba audio y verifica activacion con margen minimo de confianza.

    Returns:
        True si se detecto activacion confiable.
    """
    if demo:
        consola.estado("escucha")
    else:
        print("\n=== Escuchando frase de activacion ===")
    print('  Di exactamente: "oye computadora"')
    if demo:
        consola.info("  Presiona Enter cuando vayas a hablar...")
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

    if demo:
        consola.exito(f"  Activacion confiable (margen {margen:.1f}).")
        consola.banner_activacion()
        if es_linux():
            if abrir_terminal_activacion():
                consola.info("  Se abrio una ventana de terminal (activacion).")
            else:
                consola.aviso("  No se pudo abrir terminal extra (sigue en esta ventana).")
    else:
        print(f"  Activacion confiable (margen {margen:.1f}).")
    return True


def escuchar_comando(verbose: bool = False, demo: bool = False) -> tuple[str | None, dict[str, float]]:
    """
    Graba y predice comando con reintentos si la confianza es baja.

    Returns:
        (intencion, puntajes) o (None, {}) si no hubo prediccion confiable.
    """
    mostrar_frases_comando()

    for intento in range(1, MAX_REINTENTOS_COMANDO + 1):
        if demo:
            consola.estado("comando")
        else:
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
            print("  Repite la frase, igual que en la demo.")
            mostrar_frases_comando()

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


def mostrar_evaluacion_demo() -> None:
    """Evalua el dataset offline y muestra matriz de confusion al iniciar la demo."""
    consola.titulo("Evaluando modelo sobre el dataset...")
    consola.info("  (tarda unos segundos; clasifica todos los WAV de dataset/)")
    try:
        resultado = evaluar_dataset()
        mostrar_reporte_demo(resultado, mostrar_grafica=True)
    except (ValueError, FileNotFoundError) as error:
        consola.aviso(f"  No se pudo evaluar el dataset: {error}")
    consola.separador("Asistente en vivo")


def ejecutar_flujo_asistente(
    verbose: bool = False,
    una_vez: bool = False,
    demo: bool = False,
    mostrar_evaluacion: bool = True,
) -> None:
    """Bucle principal del asistente de voz."""
    sistema = "macOS (desarrollo)" if sys.platform == "darwin" else "Linux"

    if demo:
        consola.banner_inicio(sistema, INTENCIONES_EJECUTABLES)
        consola.info(
            f"Margenes: activacion={MARGEN_MINIMO_ACTIVACION}, "
            f"comando={MARGEN_MINIMO_COMANDO}"
        )
        if mostrar_evaluacion:
            mostrar_evaluacion_demo()
        if es_linux():
            consola.exito("Modo demo Linux: proceso siempre activo hasta Ctrl+C.")
            consola.info("Al activar con voz se abre una terminal extra (si hay emulador).")
        else:
            consola.aviso("Modo demo en Mac: sin ventanas extra (usa la VM Linux para la demo).")
    else:
        print("Asistente de voz iniciado.")
        print("Intenciones ejecutables:", ", ".join(INTENCIONES_EJECUTABLES))
        print(
            f"Margenes minimos: activacion={MARGEN_MINIMO_ACTIVACION}, "
            f"comando={MARGEN_MINIMO_COMANDO}"
        )

    while True:
        if not escuchar_activacion(verbose=verbose, demo=demo):
            if una_vez:
                break
            continue

        if demo:
            consola.estado("activo")
        else:
            print("\nActivacion detectada. Di un comando.")
        intencion, puntajes = escuchar_comando(verbose=verbose, demo=demo)

        if intencion is None:
            if una_vez:
                break
            if demo:
                consola.separador("Volviendo a modo escucha")
            else:
                print("\n--- Volviendo a modo escucha ---")
            continue

        if intencion in INTENCIONES_EJECUTABLES:
            if demo:
                consola.estado("ejecutando")
                consola.exito(f"Comando reconocido: {intencion}")
            else:
                print(f"\nComando reconocido: {intencion}")

            if demo and es_linux() and ejecutar_comando_en_terminal_nueva(intencion):
                consola.exito("Comando lanzado en una terminal nueva.")
                codigo = 0
            else:
                codigo = ejecutar_intencion(intencion)

            if codigo != 0:
                msg = f"El comando termino con codigo {codigo}."
                if demo:
                    consola.aviso(msg)
                else:
                    print(f"[aviso] {msg}")
        else:
            msg = f"Intencion '{intencion}' no esta en la lista de comandos ejecutables."
            if demo:
                consola.aviso(msg)
            else:
                print(f"\n{msg}")

        if una_vez:
            break

        if demo:
            consola.separador("Volviendo a modo escucha")
        else:
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
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Modo presentacion: colores, banner, matriz de confusion al inicio "
            "y en Linux abre terminales al activar y al ejecutar comandos"
        ),
    )
    parser.add_argument(
        "--sin-evaluacion",
        action="store_true",
        help="En modo --demo, no evaluar el dataset ni mostrar matriz al inicio",
    )
    args = parser.parse_args()

    if args.probar_comando:
        probar_comando_desde_archivo(args.probar_comando, verbose=args.verbose)
        return

    ejecutar_flujo_asistente(
        verbose=args.verbose,
        una_vez=args.una_vez,
        demo=args.demo,
        mostrar_evaluacion=not args.sin_evaluacion,
    )


if __name__ == "__main__":
    main()
