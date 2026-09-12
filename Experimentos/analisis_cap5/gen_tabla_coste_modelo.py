# -*- coding: utf-8 -*-
"""Coste de inferencia por variable, modelo y nivel (B0 y B1).

Desglosa lo que la tabla de coste agregado suma. Solo los tres modelos accesibles
por API: el local registra el tiempo por pieza y no por variable. Se incluyen los
dos niveles porque el texto contrasta el reparto uniforme del control con la
concentracion del gasto en B1. Escribe tabla_coste_modelo.tex. No editar la tabla
a mano.
"""
import csv, io, os, collections

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.normpath(os.path.join(BASE, "..", "TFM", "TFM___JORGE_GARCELA_N_GO_MEZ",
        "Plantilla_TFG_ingles_2019", "chapters", "5 results and discussion"))

VAR = [("V25", "lenguaje sexista"), ("V26", "masculino genérico"),
       ("V30", "sexismo en el discurso"), ("V33", "asimetría mujer/hombre"),
       ("V35", "denominación sexualizada")]
MOD = [("gemini-3.1-flash-lite", "gemini"), ("gpt-4o-mini", "gpt-4o-mini"),
       ("gpt-5.4-nano", "gpt-5.4-nano")]

rows = list(csv.DictReader(io.open(os.path.join(BASE, "coste_por_variable.csv"), encoding="utf-8")))
c = collections.defaultdict(float)
for r in rows:
    c[(r["nivel"], r["codigo"], r["modelo"])] += float(r["coste"])

def usd(x):
    return f"{x:.2f}".replace(".", ",")

o = io.StringIO()
o.write("% Generado por gen_tabla_coste_modelo.py. No editar a mano.\n\n")
o.write(r"""\begin{table}[H]
    \centering
    \begingroup
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \ttabbox[\FBwidth]{
        \caption{Coste de inferencia en USD sobre las 1.313 piezas, desglosado por variable, modelo y nivel, B0 (\textit{baseline}) y B1 (Agent Skills).}
        \label{tab:iris-coste-modelo}
    }{
        \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l cc cc cc >{\columncolor{gray!15}}c >{\columncolor{gray!15}}c@{}}
            \toprule
            & \multicolumn{2}{c}{\textbf{gemini}} & \multicolumn{2}{c}{\textbf{gpt-4o-mini}} & \multicolumn{2}{c}{\textbf{gpt-5.4-nano}} & \multicolumn{2}{>{\columncolor{gray!15}}c}{\textbf{Total}} \\
            \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
            \textbf{Variable} & B0 & B1 & B0 & B1 & B0 & B1 & B0 & B1 \\
            \midrule
""")

def celdas(sel):
    """Devuelve las ocho cifras de una fila: B0 y B1 de cada modelo y los totales."""
    v = []
    for m, _ in MOD:
        v += [sel("B0", m), sel("B1", m)]
    v += [sum(sel("B0", m) for m, _ in MOD), sum(sel("B1", m) for m, _ in MOD)]
    return v

for cod, nom in VAR:
    v = celdas(lambda n, m, cod=cod: c[(n, cod, m)])
    o.write(f"            {nom} ({cod}) & " + " & ".join(usd(x) for x in v) + " \\\\\n")

tot = celdas(lambda n, m: sum(c[(n, cod, m)] for cod, _ in VAR))
o.write("            \\midrule\n")
o.write("            \\textbf{Total} & " + " & ".join(f"\\textbf{{{usd(x)}}}" for x in tot) + " \\\\\n")
o.write(r"""            \bottomrule
        \end{tabular*}
    }
    \endgroup
\end{table}
""")
dest = os.path.join(DEST, "tabla_coste_modelo.tex")
io.open(dest, "w", encoding="utf-8").write(o.getvalue())
print("escrito:", dest)
