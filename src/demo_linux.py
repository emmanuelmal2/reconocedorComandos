"""
Utilidades para la presentacion en Linux (VM): abrir terminales y notificaciones.

Solo se usan con `python -m src.asistente --demo` en sistemas tipo Linux.
"""

from __future__ import annotations

import shlex
import subprocess
import sys

from src.configuracion import RAIZ_PROYECTO
from src.ejecutar import obtener_argumentos_comando


def es_linux() -> bool:
    return sys.platform.startswith("linux")


def notificar_escritorio(titulo: str, mensaje: str) -> None:
    """Notificacion de escritorio si notify-send esta disponible."""
    if not es_linux():
        return
    try:
        subprocess.run(
            ["notify-send", titulo, mensaje],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def _lanzar_terminal(titulo: str, script_bash: str) -> bool:
    """
    Abre una ventana de terminal con un script bash.
    Prueba emuladores comunes en Ubuntu/Debian VM.
    """
    if not es_linux():
        return False

    script_bash = (
        f"cd {shlex.quote(str(RAIZ_PROYECTO))} && {script_bash}"
    )

    candidatos: list[list[str]] = [
        ["gnome-terminal", "--title", titulo, "--", "bash", "-lc", script_bash],
        [
            "xfce4-terminal",
            "--title",
            titulo,
            "--hold",
            "-e",
            f"bash -lc {shlex.quote(script_bash)}",
        ],
        ["konsole", "--title", titulo, "-e", "bash", "-lc", script_bash],
        ["xterm", "-T", titulo, "-e", "bash", "-lc", script_bash],
    ]

    for comando in candidatos:
        ejecutable = comando[0]
        try:
            subprocess.Popen(
                comando,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except FileNotFoundError:
            continue

    return False


def abrir_terminal_activacion() -> bool:
    """
    Al detectar 'oye computadora', abre una terminal visible (efecto demo).
    La ventana principal sigue escuchando el comando de voz.
    """
    script = (
        'clear; '
        'echo "========================================"; '
        'echo "  ASISTENTE ACTIVADO"; '
        'echo "  Di tu comando en la ventana principal"; '
        'echo "========================================"; '
        'sleep 8'
    )
    ok = _lanzar_terminal("Asistente — activado", script)
    if ok:
        notificar_escritorio("Asistente LPC+HMM", "Activacion detectada")
    return ok


def ejecutar_comando_en_terminal_nueva(intencion: str) -> bool:
    """
    Ejecuta el comando de la intencion en una terminal nueva (visible en la demo).
    """
    argumentos = obtener_argumentos_comando(intencion)
    comando = " ".join(shlex.quote(a) for a in argumentos)
    script = (
        f'clear; '
        f'echo "=== Comando reconocido: {intencion} ==="; '
        f'echo "$ {comando}"; '
        f'echo; '
        f'{comando}; '
        f'echo; '
        f'read -p "Pulsa Enter para cerrar esta ventana..."'
    )
    ok = _lanzar_terminal(f"Asistente — {intencion}", script)
    if ok:
        notificar_escritorio("Asistente LPC+HMM", f"Ejecutando: {intencion}")
    return ok
