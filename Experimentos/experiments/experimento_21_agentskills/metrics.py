#!/usr/bin/env python3
"""
Métricas del Experimento 21 vs ground truth de las expertas.

Mergea las predicciones (`modelo_<var>` en el CSV de main.py) con el GT del corpus
(columnas `<var>`) por IdNoticia, y calcula accuracy, Cohen's Kappa y F1 macro por
variable sobre las filas anotadas. Reporta además métricas de comportamiento del
agente (uso de tools, colapso a B0).

Uso:
    python3 metrics.py --pred results/smoke/exp21_gpt-4o-mini.csv \
        --corpus "../../../data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

VARIABLES = [
    "lenguaje_sexista", "masc_generico", "sexismo_discurso",
    "asimetria_mujer_hombre", "denominacion_sexualizada",
]
ID = "IdNoticia"
CORPUS_DEFAULT = ("../../../data/2026_02_10_imio_def_todo_envio_heidy.xlsx - "
                  "2026_02_09_imio_def_todo_clara_scrape.csv")


def clean_val(x):
    """Normaliza 1 / 1.0 / '1,0' / '1' a string entero; NaN/'' → None."""
    if pd.isna(x) or str(x).strip() == "":
        return None
    s = str(x).strip().replace(",", ".")
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="CSV de main.py (columnas modelo_<var>)")
    ap.add_argument("--corpus", default=CORPUS_DEFAULT, help="Corpus con GT y IdNoticia")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    corpus = pd.read_csv(args.corpus)
    pred[ID] = pred[ID].astype(str)
    corpus[ID] = corpus[ID].astype(str)

    gt = corpus[[ID] + [v for v in VARIABLES if v in corpus.columns]]
    df = pred.merge(gt, on=ID, how="inner", suffixes=("", "_gt"))
    print(f"Filas predichas: {len(pred)} | con GT (merge): {len(df)}\n")

    filas = []
    for v in VARIABLES:
        pcol, tcol = f"modelo_{v}", v
        if pcol not in df.columns or tcol not in df.columns:
            print(f"  (salto {v}: falta {pcol} o {tcol})")
            continue
        sub = df[[tcol, pcol]].copy()
        sub[tcol] = sub[tcol].apply(clean_val)
        sub[pcol] = sub[pcol].apply(clean_val)
        sub = sub.dropna()
        if sub.empty:
            continue
        yt, yp = sub[tcol], sub[pcol]
        try:
            kappa = cohen_kappa_score(yt, yp)
        except Exception:
            kappa = float("nan")
        filas.append({
            "variable": v,
            "N": len(sub),
            "accuracy": round(accuracy_score(yt, yp), 3),
            "kappa": round(kappa, 3),
            "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 3),
        })

    res = pd.DataFrame(filas)
    print("=== ACUERDO CON GT (por variable) ===")
    print(res.to_string(index=False) if not res.empty else "(sin datos)")

    # Comportamiento del agente
    print("\n=== COMPORTAMIENTO DEL AGENTE (medias) ===")
    for v in VARIABLES:
        nt, cb = f"{v}_n_tools", f"{v}_colapso_b0"
        if nt in df.columns:
            print(f"  {v:<26} n_tools≈{df[nt].mean():.1f} | colapso_b0={df[cb].mean()*100:.0f}%"
                  if cb in df.columns else f"  {v}: n_tools≈{df[nt].mean():.1f}")

    # Coste
    if "coste_articulo_usd" in df.columns and df["coste_articulo_usd"].notna().any():
        c = df["coste_articulo_usd"].dropna()
        tk = df["tokens_articulo"].dropna() if "tokens_articulo" in df.columns else None
        print("\n=== COSTE ===")
        print(f"  coste/artículo (5 vars): media ${c.mean():.5f} | min ${c.min():.5f} | max ${c.max():.5f}")
        print(f"  total {len(c)} artículos: ${c.sum():.5f}")
        if tk is not None:
            print(f"  tokens/artículo: media {tk.mean():.0f}")
        if {"cache_read_articulo", "prompt_tokens_articulo"} <= set(df.columns):
            cr, pt = df["cache_read_articulo"].sum(), df["prompt_tokens_articulo"].sum()
            pct = (cr / pt * 100) if pt else 0.0
            print(f"  caché: {cr:.0f}/{pt:.0f} tokens de prompt servidos ({pct:.1f}% aciertos)")
            print(f"  coste por 1M tokens: ${c.sum() / tk.sum() * 1e6:.4f}")
        print(f"  → extrapolado a 1315 anotados (evaluación): ${c.mean()*1315:.2f}")
    else:
        print("\n(coste no estimable para este modelo — ver costes.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
