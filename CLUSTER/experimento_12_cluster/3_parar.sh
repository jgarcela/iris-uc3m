#!/usr/bin/env bash
# Paso 3 — EJECUTAR DENTRO DEL CLUSTER.
# Para todo lo tuyo: clientes exp12 + jobs Ollama. Deja squeue limpio.
# Nota: lo ideal es parar el launcher con Ctrl+C (cancela sus jobs solo).
#       Este script es la red de seguridad para no dejar procesos dormidos.
set -uo pipefail

echo "==> Matando clientes exp12 (main_cluster.py)..."
pkill -f main_cluster.py 2>/dev/null && echo "   clientes terminados" || echo "   no había clientes"

echo "==> Jobs tuyos ANTES de cancelar:"
squeue -u "$USER" || true

echo "==> Cancelando jobs del servidor LLM..."
# OJO: SLURM trunca el nombre a "LLM Eval", así que 'scancel --name "LLM Evaluation"'
# NO funciona. Cancelamos todos tus jobs (si tuvieras otros que salvar, hazlo por ID).
scancel -u "$USER"

sleep 3
echo "==> Jobs tuyos DESPUÉS (debería estar vacío):"
squeue -u "$USER" || true

echo
echo "Si arriba sigue apareciendo algo, cancélalo a mano: scancel JOBID"
