#!/usr/bin/env python3
"""
Experimento 22 — runner de confianza. Clasifica las 5 variables pidiendo prob_si.
Salida compatible con las métricas del exp21 (columnas modelo_<var>) más
modelo_<var>_prob_si. Escritura incremental + reanudable por IdNoticia.

Uso:
    python3 main.py --input ../../<corpus>.csv --modelo gpt-4o-mini \
        --output-dir results/gpt-4o-mini [--only-labeled] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

EXP22_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP22_DIR))
import clasificador  # noqa: E402

VARIABLES = ["lenguaje_sexista", "masc_generico", "sexismo_discurso",
             "asimetria_mujer_hombre", "denominacion_sexualizada"]
COLUMNA_ID = "IdNoticia"
COLUMNA_TEXTO = "contenido_articulo"
MAX_FALLOS_SEGUIDOS = 3


def procesar_fila(texto: str, modelo: str) -> dict:
    fila: dict = {}
    coste = 0.0
    n_err = 0
    for v in VARIABLES:
        res, tz = clasificador.clasificar(v, texto, modelo)
        fila[f"modelo_{v}"] = res["codigo"]
        fila[f"modelo_{v}_prob_si"] = res["prob_si"]
        fila[f"{v}_explicacion"] = res["explicacion"]
        fila[f"{v}_error"] = tz.get("error") or ""
        if tz.get("error"):
            n_err += 1
        if tz.get("coste_usd") is not None:
            coste += tz["coste_usd"]
    fila["coste_articulo_usd"] = round(coste, 6)
    fila["n_variables_error"] = n_err
    return fila


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--modelo", default="gpt-4o-mini")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-labeled", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.only_labeled:
        cols = [v for v in VARIABLES if v in df.columns]
        if cols:
            df = df.dropna(subset=cols)
    if args.limit:
        df = df.head(args.limit)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    salida = out_dir / f"exp22_{args.modelo.replace('/', '_')}.csv"

    procesados: set = set()
    if salida.is_file():
        prev = pd.read_csv(salida)
        if COLUMNA_ID in prev.columns:
            procesados = set(prev[COLUMNA_ID].astype(str))
        print(f"Reanudando: {len(procesados)} filas ya hechas.")

    primera = not salida.is_file()
    fallos = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        rid = str(row.get(COLUMNA_ID, ""))
        if rid in procesados:
            continue
        texto = str(row[COLUMNA_TEXTO]) if pd.notna(row.get(COLUMNA_TEXTO)) else ""
        fila = {COLUMNA_ID: rid, **procesar_fila(texto, args.modelo)}

        if fila["n_variables_error"] == len(VARIABLES):
            fallos += 1
            print(f"\n⚠️  {rid}: fallaron las {len(VARIABLES)} variables ({fallos}/{MAX_FALLOS_SEGUIDOS}).")
            if fallos >= MAX_FALLOS_SEGUIDOS:
                print(f"\n❌ ABORTADO: revisa saldo/API. Progreso en {salida} (reanudable).")
                return 2
            continue
        fallos = 0

        pd.DataFrame([fila]).to_csv(salida, mode="w" if primera else "a",
                                    header=primera, index=False, encoding="utf-8")
        primera = False
    print(f"Hecho → {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
