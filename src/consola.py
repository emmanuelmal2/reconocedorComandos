"""
Salida bonita en terminal para la demo del asistente (Linux / presentacion).

Usa colores ANSI. En terminales sin color, los textos se ven normales.
"""

from __future__ import annotations

import shutil
import sys

# Colores ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
VERDE = "\033[92m"
AMARILLO = "\033[93m"
CIAN = "\033[96m"
MAGENTA = "\033[95m"
ROJO = "\033[91m"
GRIS = "\033[90m"


def _soporta_color() -> bool:
    """True si la salida es una terminal con colores."""
    return sys.stdout.isatty() and not shutil.which("NO_COLOR")


def _pintar(texto: str, *estilos: str) -> str:
    if not _soporta_color():
        return texto
    prefijo = "".join(estilos)
    return f"{prefijo}{texto}{RESET}"


def titulo(texto: str) -> None:
    print(_pintar(texto, BOLD, CIAN))


def exito(texto: str) -> None:
    print(_pintar(texto, BOLD, VERDE))


def aviso(texto: str) -> None:
    print(_pintar(texto, AMARILLO))


def error(texto: str) -> None:
    print(_pintar(texto, BOLD, ROJO))


def info(texto: str) -> None:
    print(_pintar(texto, GRIS))


def estado(modo: str) -> None:
    """Muestra el estado actual del asistente."""
    etiquetas = {
        "escucha": ("MODO ESCUCHA", "Di: oye computadora", CIAN),
        "activo": ("ASISTENTE ACTIVADO", "Di un comando de voz", VERDE),
        "comando": ("RECONOCIENDO COMANDO", "Habla al microfono", MAGENTA),
        "ejecutando": ("EJECUTANDO COMANDO", "Salida del sistema", AMARILLO),
    }
    etiqueta, hint, color = etiquetas.get(modo, (modo, "", GRIS))
    linea = "=" * 62
    print()
    print(_pintar(linea, color))
    print(_pintar(f"  {etiqueta}", BOLD, color))
    print(_pintar(f"  {hint}", color))
    print(_pintar(linea, color))
    print()


def banner_inicio(sistema: str, intenciones: list[str]) -> None:
    """Banner de bienvenida al iniciar el asistente en modo demo."""
    print()
    print(_pintar("+" + "-" * 60 + "+", CIAN))
    print(_pintar("|  ASISTENTE DE VOZ — LPC + HMM                          |", BOLD, CIAN))
    print(_pintar("|  Reconocimiento de Patrones                            |", CIAN))
    print(_pintar("+" + "-" * 60 + "+", CIAN))
    print(_pintar(f"  Sistema objetivo : {sistema}", GRIS))
    print(_pintar(f"  Comandos         : {', '.join(intenciones)}", GRIS))
    print(_pintar("  Proceso activo   : si (bucle continuo, Ctrl+C para salir)", GRIS))
    print(_pintar('  Activacion       : "oye computadora"', GRIS))
    print()


def banner_activacion() -> None:
    """Pantalla llamativa cuando se detecta activacion."""
    print()
    print(_pintar("*" * 62, BOLD, VERDE))
    print(_pintar("*                                                            *", VERDE))
    print(_pintar("*        >>>  ACTIVACION DETECTADA  <<<                      *", BOLD, VERDE))
    print(_pintar("*        El asistente esta listo para tu comando.           *", VERDE))
    print(_pintar("*                                                            *", VERDE))
    print(_pintar("*" * 62, BOLD, VERDE))
    print()


def separador(texto: str = "") -> None:
    if texto:
        print(_pintar(f"\n--- {texto} ---", GRIS))
    else:
        print(_pintar("-" * 62, GRIS))
