#!/usr/bin/env bash
# Lanza el asistente en modo demo dentro de una terminal visible (VM Linux).
# Uso: ./scripts/iniciar_demo_linux.sh

set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"

if [[ ! -d ".venv" ]]; then
  echo "[error] No existe .venv. Crea el entorno e instala dependencias primero."
  exit 1
fi

if [[ ! -f "models/escalador.pkl" ]]; then
  echo "[aviso] No hay modelos entrenados. Ejecuta: python -m src.entrenar"
  exit 1
fi

ACTIVAR="source .venv/bin/activate && python -m src.asistente --demo"
TITULO="Asistente de voz — LPC+HMM"

lanzar() {
  local cmd=("$@")
  if command -v "${cmd[0]}" >/dev/null 2>&1; then
    exec "${cmd[@]}"
  fi
  return 1
}

# Intenta abrir emulador grafico; si no hay GUI, corre en la terminal actual.
if launch_terminal() {
  lanzar gnome-terminal --title "$TITULO" --geometry=100x32 -- bash -lc "$ACTIVAR; echo; read -p 'Pulsa Enter para cerrar...'" \
    || lanzar xfce4-terminal --title "$TITULO" --hold -e bash -lc "$ACTIVAR" \
    || lanzar konsole --title "$TITULO" -e bash -lc "$ACTIVAR" \
    || lanzar xterm -T "$TITULO" -e bash -lc "$ACTIVAR"
  return 0
}

if [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
  if launch_terminal; then
    exit 0
  fi
fi

echo "Sin emulador grafico detectado; iniciando en esta terminal..."
bash -lc "$ACTIVAR"
