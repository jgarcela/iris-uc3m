# -*- coding: utf-8 -*-
"""Heterogeneidad de la anotacion humana, variable a variable.

Sustituye a la tabla de kappa por equipo. Mide la dispersion del criterio humano sin
intervencion de los modelos y sin correccion por azar, comparando la brecha entre los
dos equipos con la que separa a las anotadoras individuales. Se restringe a las que
codificaron 50 piezas o mas, para que las proporciones descansen sobre una base
suficiente. Escribe tabla_heterogeneidad.tex. No editar la tabla a mano.
"""
import io, os
import numpy as np
import pandas as pd

BASE = "/home/jggomez/Desktop/IRIS/iris-uc3m/Experimentos/experiments/experimento_21_agentskills"
DEST = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TFM",
        "TFM___JORGE_GARCELA_N_GO_MEZ", "Plantilla_TFG_ingles_2019",
        "chapters", "5 results and discussion"))

VAR = [("lenguaje_sexista", "V25"), ("masc_generico", "V26"),
       ("sexismo_discurso", "V30"), ("asimetria_mujer_hombre", "V33"),
       ("denominacion_sexualizada", "V35")]
V = [v for v, _ in VAR]
MIN_N = 50

def binar(s):
    x = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
    return x.map(lambda v: 0 if v == 1 else (1 if v in (2, 3) else np.nan))

gt = pd.read_csv(f"{BASE}/real1315_corpus.csv", low_memory=False)
G = pd.DataFrame({"u": gt["no_NombreUsuario"].astype(str)})
for v in V:
    G[v] = binar(gt[v])
G = G.dropna(subset=V)
assert len(G) == 1313, len(G)
G["equipo"] = G.u.str.contains("ndexa", case=False).map({True: "Indexa", False: "UC3M"})

tam = G.groupby("u").size()
keep = tam[tam >= MIN_N].index
g = G[G.u.isin(keep)]

def pct(x):
    return f"{x*100:.1f}".replace(".", ",")

o = io.StringIO()
o.write("% Generado por gen_tabla_heterogeneidad.py. No editar a mano.\n\n")
o.write(r"""\begin{table}[H]
    \centering
    \begingroup
    \footnotesize
    \setlength{\tabcolsep}{4pt}
    \renewcommand{\texttt}[1]{{\ttfamily #1}}
    \ttabbox[\FBwidth]{
        \caption{Heterogeneidad de la anotación humana, en porcentaje de piezas marcadas como \textit{Sí}. Las tres primeras columnas comparan a los dos equipos y las tres últimas recorren a las anotadoras una a una. Todas las cifras se restringen a las """ + str(len(keep)) + r""" anotadoras que codificaron al menos """ + str(MIN_N) + r""" piezas (""" + f"{len(g):,}".replace(",", ".") + r""" de las 1.313). Ningún modelo interviene en estas cifras.}
        \label{tab:iris-heterogeneidad}
    }{
        \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l cc c cc >{\columncolor{gray!15}}c@{}}
            \toprule
            & \multicolumn{3}{c}{\textbf{Entre equipos}} & \multicolumn{3}{c}{\textbf{Entre anotadoras}} \\
            \cmidrule(lr){2-4} \cmidrule(lr){5-7}
            \textbf{Variable} & Indexa & UC3M & \textbf{Brecha} & Mín. & Máx. & \textbf{Rango} \\
            \midrule
""")
for v, cod in VAR:
    i = g[g.equipo == "Indexa"][v].mean()
    u = g[g.equipo == "UC3M"][v].mean()
    p = g.groupby("u")[v].mean()
    o.write(f"            {cod} \\texttt{{{v.replace('_', chr(92)+'_')}}} & {pct(i)} & {pct(u)} & "
            f"\\textbf{{{pct(abs(u-i))}}} & {pct(p.min())} & {pct(p.max())} & "
            f"\\textbf{{{pct(p.max()-p.min())}}} \\\\\n")
o.write(r"""            \bottomrule
        \end{tabular*}
    }
    \endgroup
\end{table}
""")
dest = os.path.join(DEST, "tabla_heterogeneidad.tex")
io.open(dest, "w", encoding="utf-8").write(o.getvalue())
print("escrito:", dest)
