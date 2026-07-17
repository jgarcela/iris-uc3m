#!/usr/bin/env bash
# Paso 1 — EJECUTAR EN queron (tu PC).
# Sube Experimentos/, el cliente y el CSV al volumen compartido del cluster.
set -euo pipefail

# --- RELLENA ESTO ---
CL_USER="${CL_USER:-jggomez}"                 # confirma con 'whoami' dentro del cluster
CL_HOST="${CL_HOST:-ceres.tsc.uc3m.es}"       # máquina de acceso de tu grupo (pregunta a Harold)
CL_BASE="${CL_BASE:-/export/usuarios01/$CL_USER/iris}"
# --------------------

LOCAL_REPO="/home/jggomez/Desktop/IRIS/iris-uc3m"
CSV="2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"

# Excluimos lo pesado/innecesario para no saturar la red
EXCLUDES=(
  --exclude '__pycache__/'
  --exclude '.ipynb_checkpoints/'
  --exclude 'results/'
  --exclude 'metrics/'
  --exclude '*.csv'          # los CSV de Experimentos/ no hacen falta; el de datos va aparte
)

echo "==> Subiendo Experimentos/ a $CL_HOST:$CL_BASE/Experimentos/"
rsync -avz --progress "${EXCLUDES[@]}" \
  "$LOCAL_REPO/Experimentos/" \
  "$CL_USER@$CL_HOST:$CL_BASE/Experimentos/"

echo "==> Subiendo el cliente experimento_12_cluster/"
rsync -avz --progress --exclude '__pycache__/' --exclude 'logs/' --exclude 'results/' \
  "$LOCAL_REPO/CLUSTER/experimento_12_cluster/" \
  "$CL_USER@$CL_HOST:$CL_BASE/experimento_12_cluster/"

echo "==> Subiendo el CSV de datos (solo el de exp12)"
rsync -avz --progress \
  "$LOCAL_REPO/data/$CSV" \
  "$CL_USER@$CL_HOST:$CL_BASE/data/"

echo
echo "Listo. Ahora entra al cluster y ejecuta 2_setup.sh:"
echo "  ssh $CL_USER@$CL_HOST"
echo "  bash $CL_BASE/experimento_12_cluster/2_setup.sh"
