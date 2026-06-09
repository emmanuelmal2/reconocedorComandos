"""
Ejecucion de comandos Bash para intenciones ejecutables.

Solo se permiten intenciones en INTENCIONES_EJECUTABLES.
La intencion "activacion" no ejecuta nada.

Seguridad:
    - Lista blanca fija de intenciones y comandos
    - subprocess.run con shell=False (sin texto libre)

Nota: el proyecto apunta a Linux. En macOS se usan comandos equivalentes
para poder probar en desarrollo (free/ip no existen en Mac por defecto).
"""

from __future__ import annotations

import subprocess
import sys

from src.configuracion import INTENCIONES_EJECUTABLES

# Linux (objetivo del proyecto / presentacion)
COMANDOS_LINUX: dict[str, list[str]] = {
    "listar": ["ls", "-la"],
    "memoria": ["free", "-h"],
    "disco": ["df", "-h"],
    "red": ["ip", "addr"],
    "procesos": ["ps", "aux", "--sort=-pcpu"],
}

# macOS (pruebas locales en desarrollo)
COMANDOS_MACOS: dict[str, list[str]] = {
    "listar": ["ls", "-la"],
    "memoria": ["vm_stat"],
    "disco": ["df", "-h"],
    "red": ["ifconfig"],
    "procesos": ["ps", "aux", "-r"],
}


def _comandos_del_sistema() -> dict[str, list[str]]:
    """Devuelve la tabla de comandos segun el sistema operativo."""
    if sys.platform == "darwin":
        return COMANDOS_MACOS
    return COMANDOS_LINUX


def obtener_argumentos_comando(intencion: str) -> list[str]:
    """Devuelve la lista de argumentos del comando Bash para una intencion."""
    if intencion not in INTENCIONES_EJECUTABLES:
        raise ValueError(f"Intencion no ejecutable: {intencion}")
    return list(_comandos_del_sistema()[intencion])


def ejecutar_intencion(intencion: str) -> int:
    """
    Ejecuta el comando Bash asociado a una intencion.

    Returns:
        Codigo de salida del proceso (0 = exito, 1 = error)
    """
    if intencion not in INTENCIONES_EJECUTABLES:
        print(f"[aviso] La intencion '{intencion}' no tiene comando Bash asociado.")
        return 1

    argumentos = obtener_argumentos_comando(intencion)
    sistema = "macOS" if sys.platform == "darwin" else "Linux"
    print(f"\nEjecutando ({sistema}): {' '.join(argumentos)}")

    try:
        resultado = subprocess.run(
            argumentos,
            shell=False,
            check=False,
            text=True,
            capture_output=False,
        )
        return resultado.returncode
    except FileNotFoundError:
        print(
            f"[error] No se encontro el programa '{argumentos[0]}' en este sistema.\n"
            f"  Este proyecto esta pensado para Linux. En Mac usa los equivalentes\n"
            f"  o prueba en una maquina Linux."
        )
        return 1
