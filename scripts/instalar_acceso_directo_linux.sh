#!/usr/bin/env bash
# Instala acceso directo en el menu de aplicaciones de Ubuntu.
# Uso: ./scripts/instalar_acceso_directo_linux.sh

set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DESTINO="${HOME}/.local/share/applications/reconocedor-comandos.desktop"

if [[ ! -d "${RAIZ}/.venv" ]]; then
  echo "[error] No existe .venv en ${RAIZ}"
  exit 1
fi

mkdir -p "${HOME}/.local/share/applications"
sed "s|@RAIZ@|${RAIZ}|g" "${RAIZ}/scripts/reconocedor-comandos.desktop" > "${DESTINO}"
chmod +x "${DESTINO}"

echo "[ok] Acceso directo instalado:"
echo "     ${DESTINO}"
echo "Buscalo en el menu como: Asistente de voz LPC+HMM"
