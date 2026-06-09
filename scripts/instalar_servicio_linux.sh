#!/usr/bin/env bash
# Instala servicio systemd de usuario para el asistente siempre activo.
# Uso: ./scripts/instalar_servicio_linux.sh

set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DESTINO="${HOME}/.config/systemd/user/reconocedor-comandos.service"

if [[ ! -d "${RAIZ}/.venv" ]]; then
  echo "[error] No existe .venv en ${RAIZ}"
  exit 1
fi

if [[ ! -f "${RAIZ}/models/escalador.pkl" ]]; then
  echo "[aviso] No hay modelos. Ejecuta primero: python -m src.entrenar"
  exit 1
fi

mkdir -p "${HOME}/.config/systemd/user"
sed "s|@RAIZ@|${RAIZ}|g" "${RAIZ}/scripts/reconocedor-comandos.service" > "${DESTINO}"

systemctl --user daemon-reload
systemctl --user enable reconocedor-comandos.service

echo "[ok] Servicio instalado: ${DESTINO}"
echo ""
echo "Comandos utiles:"
echo "  systemctl --user start reconocedor-comandos    # iniciar ahora"
echo "  systemctl --user stop reconocedor-comandos     # detener"
echo "  systemctl --user status reconocedor-comandos   # ver estado"
echo "  journalctl --user -u reconocedor-comandos -f   # ver logs"
echo ""
echo "Nota: el servicio usa --sin-evaluacion (sin ventana matplotlib al inicio)."
echo "      Necesita microfono y sesion grafica (DISPLAY) en la VM."
