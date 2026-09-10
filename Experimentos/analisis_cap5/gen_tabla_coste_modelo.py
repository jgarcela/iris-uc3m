# -*- coding: utf-8 -*-
"""Coste de inferencia por variable y modelo en el nivel B1.

Desglosa lo que la tabla de coste agregado suma. Solo los tres modelos accesibles
por API: el local registra el tiempo por pieza y no por variable. Escribe
tabla_coste_modelo.tex. No editar la tabla a mano.
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
    if r["nivel"] == "B1":
        c[(r["codigo"], r["modelo"])] += float(r["coste"])

def usd(x):
    return f"{x:.2f}".replace(".", ",")

o = io.StringIO()
o.write("% Generado por gen_tabla_coste_modelo.py. No editar a mano.\n\n")
o.write(r"""\begin{table}[H]
    \centering
    \begingroup
    \footnotesize
    \setlength{\tabcolsep}{4pt}
    \ttabbox[\FBwidth]{
        \caption{Coste de inferencia en el nivel B1, en USD sobre las 1.313 piezas, desglosado por variable y modelo. Solo los tres modelos accesibles por API, ya que el local registra el tiempo de proceso por pieza y no por variable.}
        \label{tab:iris-coste-modelo}
    }{
        \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c >{\columncolor{gray!15}}c@{}}
            \toprule
            \textbf{Variable} & \textbf{gemini} & \textbf{gpt-4o-mini} & \textbf{gpt-5.4-nano} & \textbf{Total} \\
            \midrule
""")
for cod, nom in VAR:
    v = [c[(cod, m)] for m, _ in MOD]
    o.write(f"            {nom} ({cod}) & " + " & ".join(usd(x) for x in v) + f" & {usd(sum(v))} \\\\\n")
tot = [sum(c[(cod, m)] for cod, _ in VAR) for m, _ in MOD]
o.write("            \\midrule\n")
o.write("            \\textbf{Total} & " + " & ".join(f"\\textbf{{{usd(x)}}}" for x in tot)
        + f" & \\textbf{{{usd(sum(tot))}}} \\\\\n")
o.write(r"""            \bottomrule
        \end{tabular*}
    }
    \endgroup
\end{table}
""")
dest = os.path.join(DEST, "tabla_coste_modelo.tex")
io.open(dest, "w", encoding="utf-8").write(o.getvalue())
print("escrito:", dest)
