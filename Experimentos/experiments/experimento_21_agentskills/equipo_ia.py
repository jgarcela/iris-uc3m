#!/usr/bin/env python3
"""
El modelo como tercer equipo anotador (§7.3 de REUNION_DIRECTORA.md).

Trata cada modelo como una anotadora más y lo sitúa junto a Indexa y UCM3 en un
mismo espacio de criterios. La pregunta deja de ser "¿acierta el modelo?" y pasa
a ser "¿dónde cae el modelo entre los criterios humanos?".

Restricción de diseño del corpus: cada noticia la anotó UNA sola persona (1.313
noticias, 1.313 filas) → no hay solape humano-humano y la κ entre personas no es
calculable. El equipo IA sí tiene solape total (cada modelo anota las 1.313), así
que es el único equipo con cohesión interna medible directamente. Lo que sí es
comparable para todo el mundo sin solape es la PREVALENCIA de "Sí", y el modelo
sirve además de instrumento común para descontar qué parte de esa prevalencia se
debe a las noticias que a cada cual le tocaron.

Salidas (revision/equipo_ia/<nivel>/):
    prevalencia_coders.csv    % "Sí" por coder × variable (humanos + modelos)
    prevalencia_ajustada.csv  humano − modelo sobre las MISMAS noticias
    kappa_intra_ia.csv        κ pareada dentro del equipo IA (solape total)
    kappa_coder_vs_ia.csv     κ de cada humano contra cada modelo
    mapa_criterios.png        figura: coders situados por severidad

Uso:
    python3 equipo_ia.py                 # nivel B1 (results/bench_*)
    python3 equipo_ia.py --nivel b0
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

VARIABLES = [
    "lenguaje_sexista", "masc_generico", "sexismo_discurso",
    "asimetria_mujer_hombre", "denominacion_sexualizada",
]
ID = "IdNoticia"
CODER = "no_NombreUsuario"
CORPUS = "real1315_corpus.csv"
OUT = Path("revision/equipo_ia")

# Codificación del libro de códigos: 2 = "Sí" (el rasgo está presente), 1 = "No".
SI = 2
MIN_PIEZAS = 5  # coders con menos piezas no dan una prevalencia interpretable


def normaliza(s: pd.Series) -> pd.Series:
    """1 / 1.0 / '1,0' → 1 (entero). Valores fuera de {1,2} → NaN."""
    v = pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False),
                      errors="coerce")
    return v.where(v.isin([1, 2]))


def carga_modelos(nivel: str) -> dict[str, pd.DataFrame]:
    """Un DataFrame por modelo, indexado por IdNoticia, columnas = VARIABLES."""
    modelos = {}
    for d in sorted(Path("results").glob(f"{nivel}_*")):
        csvs = list(d.glob("*.csv"))
        if not csvs:
            continue
        raw = pd.read_csv(csvs[0])
        nombre = d.name.split("_", 1)[1]
        df = pd.DataFrame({ID: raw[ID].astype(str)})
        for v in VARIABLES:
            df[v] = normaliza(raw[f"modelo_{v}"])
        modelos[nombre] = df.set_index(ID)
    return modelos


def prevalencia(df: pd.DataFrame) -> dict[str, float]:
    return {v: (df[v] == SI).sum() / df[v].notna().sum() if df[v].notna().any()
            else float("nan") for v in VARIABLES}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nivel", default="bench", choices=["bench", "b0"],
                    help="bench = B1 (Agent Skills) · b0 = control")
    ap.add_argument("--corpus", default=CORPUS)
    args = ap.parse_args()

    global OUT
    OUT = OUT / args.nivel
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(args.corpus)
    corpus[ID] = corpus[ID].astype(str)
    hum = corpus[[ID, CODER]].copy()
    for v in VARIABLES:
        hum[v] = normaliza(corpus[v])
    hum = hum.set_index(ID)

    modelos = carga_modelos(args.nivel)
    if not modelos:
        print(f"No hay resultados en results/{args.nivel}_*")
        return 1
    print(f"Nivel {args.nivel} · modelos en el equipo IA: {', '.join(modelos)}")
    print(f"Corpus: {len(hum)} noticias · {hum[CODER].nunique()} anotadoras humanas\n")

    coders = [c for c, n in hum[CODER].value_counts().items() if n >= MIN_PIEZAS]
    descartados = set(hum[CODER].dropna().unique()) - set(coders)
    if descartados:
        print(f"Coders descartados por n < {MIN_PIEZAS}: {', '.join(sorted(descartados))}\n")

    # ── 1. Prevalencia de "Sí" por coder ──────────────────────────────────────
    # Único eje comparable entre humanos y modelos sin necesidad de solape.
    filas = []
    for c in sorted(coders):
        sub = hum[hum[CODER] == c]
        filas.append({"coder": c, "equipo": c.split()[0], "n": len(sub), **prevalencia(sub)})
    for m, df in modelos.items():
        filas.append({"coder": m, "equipo": "IA", "n": len(df), **prevalencia(df)})
    prev = pd.DataFrame(filas)
    prev["media"] = prev[VARIABLES].mean(axis=1)
    prev.to_csv(OUT / "prevalencia_coders.csv", index=False)
    print("── Prevalencia de «Sí» por coder (equipo IA incluido) ──")
    print(prev.round(3).to_string(index=False), "\n")

    # ── 2. Prevalencia ajustada por composición de la muestra ─────────────────
    # Cada humano vio noticias distintas, así que su prevalencia mezcla criterio
    # con qué le tocó. El modelo anota TODO: comparándolo consigo mismo sobre el
    # subconjunto de cada coder, la diferencia aísla el criterio.
    filas = []
    for c in sorted(coders):
        sub = hum[hum[CODER] == c]
        fila = {"coder": c, "equipo": c.split()[0], "n": len(sub)}
        for v in VARIABLES:
            deltas = []
            for m, df in modelos.items():
                comun = df.reindex(sub.index)[v]
                if comun.notna().any():
                    deltas.append((sub[v] == SI).mean() - (comun == SI).mean())
            fila[v] = sum(deltas) / len(deltas) if deltas else float("nan")
        filas.append(fila)
    ajust = pd.DataFrame(filas)
    ajust["media"] = ajust[VARIABLES].mean(axis=1)
    ajust.to_csv(OUT / "prevalencia_ajustada.csv", index=False)
    print("── Δ prevalencia humano − IA sobre las MISMAS noticias (>0 = más severo) ──")
    print(ajust.round(3).to_string(index=False), "\n")

    # ── 3. Cohesión interna del equipo IA (solape total: 1.313 noticias) ──────
    filas = []
    for a, b in itertools.combinations(sorted(modelos), 2):
        fila = {"modelo_a": a, "modelo_b": b}
        for v in VARIABLES:
            x, y = modelos[a][v], modelos[b][v].reindex(modelos[a].index)
            ok = x.notna() & y.notna()
            fila[v] = cohen_kappa_score(x[ok], y[ok]) if ok.sum() else float("nan")
        filas.append(fila)
    intra = pd.DataFrame(filas)
    if not intra.empty:
        intra["media"] = intra[VARIABLES].mean(axis=1)
        intra.to_csv(OUT / "kappa_intra_ia.csv", index=False)
        print("── κ dentro del equipo IA (cohesión interna, solape total) ──")
        print(intra.round(3).to_string(index=False), "\n")

    # ── 4. κ de cada humano contra cada modelo ────────────────────────────────
    filas = []
    for c in sorted(coders):
        sub = hum[hum[CODER] == c]
        for m, df in modelos.items():
            fila = {"coder": c, "equipo": c.split()[0], "modelo": m, "n": len(sub)}
            for v in VARIABLES:
                x = sub[v]
                y = df.reindex(sub.index)[v]
                ok = x.notna() & y.notna()
                # κ indefinida si una de las dos series es constante en el subset
                fila[v] = (cohen_kappa_score(x[ok], y[ok])
                           if ok.sum() >= MIN_PIEZAS and x[ok].nunique() > 1
                           and y[ok].nunique() > 1 else float("nan"))
            fila["media"] = pd.Series({v: fila[v] for v in VARIABLES}).mean()
            filas.append(fila)
    cross = pd.DataFrame(filas)
    cross.to_csv(OUT / "kappa_coder_vs_ia.csv", index=False)
    print("── κ humano × modelo (κ media sobre las 5 variables) ──")
    print(cross.pivot(index="coder", columns="modelo", values="media").round(3).to_string(), "\n")

    figura(prev)
    print(f"Salidas en {OUT}/")
    return 0


def figura(prev: pd.DataFrame) -> None:
    """Mapa de criterios: cada coder situado por su prevalencia de «Sí»."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"Indexa": "#4C72B0", "UCM3": "#DD8452", "IA": "#55A868"}
    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(16, 4.2), sharey=True)
    for ax, v in zip(axes, VARIABLES):
        d = prev.sort_values(v)
        ax.barh(d["coder"], d[v], color=[color[e] for e in d["equipo"]])
        ax.set_title(v, fontsize=9)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("anotador")
    fig.suptitle("Espacio de criterios: prevalencia de «Sí» por anotador "
                 "(Indexa · UCM3 · equipo IA)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color.values()]
    fig.legend(handles, color.keys(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "mapa_criterios.png", dpi=200)


if __name__ == "__main__":
    raise SystemExit(main())
