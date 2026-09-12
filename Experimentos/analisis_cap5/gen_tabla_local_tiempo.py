# -*- coding: utf-8 -*-
"""Coste en tiempo de computo del modelo local, por configuracion.

Es el equivalente para el modelo local de la tabla de coste en dolares: donde los
modelos de API gastan tarifa, el local gasta ocupacion de la granja de GPU. Lee la
columna modelo_tiempo_procesamiento_seg de los ficheros del cluster, restringida a
las 1.313 piezas evaluadas, con el mismo criterio de calc.py. Escribe
tabla_local_tiempo.tex. No editar la tabla a mano.
"""
import io, os
import numpy as np
import pandas as pd

BASE = "/home/jggomez/Desktop/IRIS/iris-uc3m/Experimentos/experiments/experimento_21_agentskills"
CL = "/home/jggomez/Desktop/IRIS/iris-uc3m/CLUSTER/experimento_21_cluster"
DEST = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TFM",
        "TFM___JORGE_GARCELA_N_GO_MEZ", "Plantilla_TFG_ingles_2019",
        "chapters", "5 results and discussion"))

V = ["lenguaje_sexista", "masc_generico", "sexismo_discurso",
     "asimetria_mujer_hombre", "denominacion_sexualizada"]
SERV = 4  # servidores en paralelo, uno por shard

CONF = [("B0, metodología en el \\textit{prompt}", "results_b0/FULL_dedup.csv"),
        ("B1, solo \\textit{skills}",              "results_b1_minimo/FULL.csv"),
        ("B1, \\textit{skills} y resúmenes",       "results_b1_singuia/FULL.csv"),
        ("B1, \\textit{skills} y RAG",             "results_b1_sinres/FULL.csv"),
        ("B1, configuración completa",             "results_b1_completo/FULL.csv")]

def binar(s):
    x = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
    return x.map(lambda v: 0 if v == 1 else (1 if v in (2, 3) else np.nan))

gt = pd.read_csv(f"{BASE}/real1315_corpus.csv", low_memory=False)
G = pd.DataFrame({"IdNoticia": gt.IdNoticia})
for v in V:
    G[v] = binar(gt[v])
IDS = set(G.dropna().IdNoticia)
assert len(IDS) == 1313, len(IDS)

def coma(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")

o = io.StringIO()
o.write("% Generado por gen_tabla_local_tiempo.py. No editar a mano.\n\n")
o.write(r"""\begin{table}[H]
    \centering
    \begingroup
    \small
    \setlength{\tabcolsep}{6pt}
    \ttabbox[\FBwidth]{
        \caption{Coste en tiempo de cómputo del modelo local sobre las 1.313 piezas, por configuración. Las dos últimas columnas recogen lo que tarda una pasada completa sobre el corpus en un solo servidor y repartida en los cuatro que se emplearon, uno por \textit{shard}. La instrumentación registra el tiempo por pieza y no por variable, de modo que la primera columna es ese tiempo dividido entre las cinco y no un desglose medido.}
        \label{tab:iris-local-tiempo}
    }{
        \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c >{\columncolor{gray!15}}c >{\columncolor{gray!15}}c@{}}
            \toprule
            \textbf{Configuración} & \textbf{s/variable} & \textbf{s/pieza} & \textbf{1 servidor} & \textbf{4 servidores} \\
            \midrule
""")
for nom, f in CONF:
    d = pd.read_csv(os.path.join(CL, f), low_memory=False)
    d = d[d.IdNoticia.isin(IDS)].drop_duplicates("IdNoticia")
    t = pd.to_numeric(d["modelo_tiempo_procesamiento_seg"], errors="coerce").dropna()
    h = t.sum() / 3600
    o.write(f"            {nom} & {coma(t.mean()/len(V))} & {coma(t.mean())} & "
            f"{coma(h)}\\,h & {coma(h/SERV)}\\,h \\\\\n")
    if nom.startswith("B0"):
        o.write("            \\midrule\n")
o.write(r"""            \bottomrule
        \end{tabular*}
    }
    \endgroup
\end{table}
""")
dest = os.path.join(DEST, "tabla_local_tiempo.tex")
io.open(dest, "w", encoding="utf-8").write(o.getvalue())
print("escrito:", dest)
