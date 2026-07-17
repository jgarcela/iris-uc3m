#!/usr/bin/env bash
# Paso 2 — EJECUTAR DENTRO DEL CLUSTER (máquina de acceso).
# Crea el venv en el volumen compartido, instala deps y comprueba imports.
set -euo pipefail

CL_BASE="${CL_BASE:-/export/usuarios01/$USER/iris}"
VENV="${VENV:-/export/clusterdata/$USER/venvs/iris}"

echo "==> Usuario: $(whoami)   HOME: $HOME"
echo "==> Base:    $CL_BASE"
echo "==> Venv:    $VENV"

# Directorios de trabajo (modelos y logs los usa el llm.json)
mkdir -p "/data/tmp/$USER/models" "$HOME/logs_llm" "$CL_BASE/results_cluster"

# Crear/activar venv
if [ ! -d "$VENV" ]; then
  echo "==> Creando venv..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$CL_BASE/experimento_12_cluster/requirements.txt"

# Comprobar que los scripts de apoyo importan bien
echo "==> Comprobando imports..."
EXPERIMENTOS_DIR="$CL_BASE/Experimentos" python3 - <<'PY'
import os, sys
sys.path.append(os.environ["EXPERIMENTOS_DIR"])
import variables, utils  # noqa
print("imports OK ->", os.environ["EXPERIMENTOS_DIR"])
PY

echo
echo "Entorno listo. Recuerda activar el venv en cada sesión:"
echo "  source $VENV/bin/activate"
echo "Siguiente: levantar Ollama (launch_process.py -c llm.json) y correr run_clients.sh"
