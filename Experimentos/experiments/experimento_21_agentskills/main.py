#!/usr/bin/env python3
"""
Experimento 21 — Agent Skills (nivel B1).

Por cada artículo lanza 5 agentes especializados (uno por variable), cada uno con
progressive disclosure de su SKILL.md (+ auxiliares). Escribe, por variable:
  <var>, <var>_explicacion, <var>_evidencias
y columnas de traza del agente:
  <var>_n_tools, <var>_skills, <var>_colapso_b0

Escritura incremental + reanudable por COLUMNA_ID (como exp 15).

Uso:
    python3 main.py --input ../../<corpus>.csv --modelo claude-haiku-4-5-20251001 \
        --output-dir results/claude-haiku [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import agente  # noqa: E402
from tools import SKILLS_VARIABLE  # noqa: E402

COLUMNA_ID = "IdNoticia"
COLUMNA_TEXTO = "contenido_articulo"
# Artículos consecutivos con las 5 variables fallidas antes de abortar (saldo/API caída).
MAX_FALLOS_SEGUIDOS = 3


MODO_BASELINE = False  # True → B0 (metodología inyectada, sin tools)


def procesar_fila(texto: str, modelo: str) -> dict:
    fila: dict = {}
    coste_total = 0.0
    coste_estimable = True
    tokens_total = 0
    n_error = 0
    clasificar = (agente.clasificar_variable_baseline if MODO_BASELINE
                  else agente.clasificar_variable)
    for variable in SKILLS_VARIABLE:
        res, traza = clasificar(variable, texto, modelo=modelo)
        fila[f"{variable}_error"] = traza.get("error") or ""
        if traza.get("error"):
            n_error += 1
        # Predicción con prefijo modelo_ para no colisionar con la columna GT <variable>.
        fila[f"modelo_{variable}"] = res["codigo"]
        fila[f"{variable}_explicacion"] = res["explicacion"]
        fila[f"{variable}_evidencias"] = json.dumps(res["evidencias"], ensure_ascii=False)
        fila[f"{variable}_n_tools"] = traza["n_tools"]
        fila[f"{variable}_skills"] = "|".join(traza["skills_cargadas"])
        fila[f"{variable}_guias"] = "|".join(traza["guias_consultadas"])
        fila[f"{variable}_colapso_b0"] = int(traza["colapso_b0"])
        tks = traza["prompt_tokens"] + traza["completion_tokens"]
        fila[f"{variable}_tokens"] = tks
        fila[f"{variable}_prompt_tokens"] = traza["prompt_tokens"]
        fila[f"{variable}_cache_read"] = traza["cache_read_tokens"]
        fila[f"{variable}_coste_usd"] = traza["coste_usd"]
        tokens_total += tks
        if traza["coste_usd"] is None:
            coste_estimable = False
        else:
            coste_total += traza["coste_usd"]
    fila["tokens_articulo"] = tokens_total
    fila["prompt_tokens_articulo"] = sum(
        fila[f"{v}_prompt_tokens"] for v in SKILLS_VARIABLE)
    fila["cache_read_articulo"] = sum(
        fila[f"{v}_cache_read"] for v in SKILLS_VARIABLE)
    fila["coste_articulo_usd"] = round(coste_total, 6) if coste_estimable else None
    fila["n_variables_error"] = n_error
    return fila


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV del corpus")
    ap.add_argument("--modelo", default="claude-haiku-4-5-20251001")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="Procesar sólo N filas (pruebas)")
    ap.add_argument("--only-labeled", action="store_true",
                    help="Sólo filas con GT en las 5 variables (para métricas)")
    ap.add_argument("--sin-resumenes-guias", action="store_true",
                    help="Ablación: no listar las skills-resumen de guías en el catálogo")
    ap.add_argument("--sin-consultar-guia", action="store_true",
                    help="Ablación: desactivar la tool RAG en vivo CONSULTAR_GUIA")
    ap.add_argument("--sin-cache", action="store_true",
                    help="Ablación: desactivar prompt caching (prompt en un único mensaje)")
    ap.add_argument("--baseline", action="store_true",
                    help="Nivel B0: metodología inyectada en el prompt, sin tools ni "
                         "progressive disclosure (comparación contra B1 skills)")
    args = ap.parse_args()

    if args.sin_resumenes_guias:
        agente.INCLUIR_RESUMENES_GUIAS = False
        print("Ablación: catálogo SIN skills-resumen de guías.")
    if args.sin_consultar_guia:
        agente.HABILITAR_CONSULTAR_GUIA = False
        print("Ablación: SIN tool RAG en vivo CONSULTAR_GUIA.")
    if args.sin_cache:
        agente.USAR_PROMPT_CACHE = False
        print("Ablación: SIN prompt caching (prompt en un único mensaje).")
    if args.baseline:
        global MODO_BASELINE
        MODO_BASELINE = True
        print("Nivel B0: metodología INYECTADA, sin tools (baseline sin skills).")

    df = pd.read_csv(args.input)
    if args.only_labeled:
        cols_gt = [v for v in SKILLS_VARIABLE if v in df.columns]
        if cols_gt:
            df = df.dropna(subset=cols_gt)
    if args.limit:
        df = df.head(args.limit)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    salida = out_dir / f"exp21_{args.modelo.replace('/', '_')}.csv"

    procesados: set = set()
    if salida.is_file():
        prev = pd.read_csv(salida)
        if COLUMNA_ID in prev.columns:
            procesados = set(prev[COLUMNA_ID].astype(str))
        print(f"Reanudando: {len(procesados)} filas ya procesadas.")

    primera = not salida.is_file()
    fallos_seguidos = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        rid = str(row.get(COLUMNA_ID, ""))
        if rid in procesados:
            continue
        texto = str(row[COLUMNA_TEXTO]) if pd.notna(row.get(COLUMNA_TEXTO)) else ""
        fila = {COLUMNA_ID: rid, **procesar_fila(texto, args.modelo)}

        # Corta-circuitos: si varios artículos fallan íntegros (saldo agotado, API caída,
        # clave revocada), abortar SIN escribir — si no, quedarían como negativos falsos
        # y la reanudación por IdNoticia los daría por buenos.
        if fila["n_variables_error"] == len(SKILLS_VARIABLE):
            fallos_seguidos += 1
            print(f"\n⚠️  Artículo {rid}: fallaron las {len(SKILLS_VARIABLE)} variables "
                  f"({fallos_seguidos}/{MAX_FALLOS_SEGUIDOS}). No se escribe la fila.")
            if fallos_seguidos >= MAX_FALLOS_SEGUIDOS:
                print("\n❌ ABORTADO: demasiados fallos consecutivos. Revisa saldo/API key.\n"
                      f"   Progreso guardado en {salida}. Relanza el mismo comando para reanudar.")
                return 2
            continue
        fallos_seguidos = 0

        df_row = pd.DataFrame([fila])
        df_row.to_csv(salida, mode="w" if primera else "a",
                      header=primera, index=False, encoding="utf-8")
        primera = False
    print(f"Hecho → {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
