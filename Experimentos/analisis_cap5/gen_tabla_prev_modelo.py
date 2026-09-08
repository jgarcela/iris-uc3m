# -*- coding: utf-8 -*-
"""Genera la tabla de prevalencia de Si predicha por cada modelo en el nivel B1.

Lee metricas_por_variable.csv (la misma fuente que el resto de tablas del Cap 5)
y escribe tabla_prev_modelo.tex. No editar la tabla a mano.
"""
import csv, io, os

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.normpath(os.path.join(BASE, "..", "TFM", "TFM___JORGE_GARCELA_N_GO_MEZ",
        "Plantilla_TFG_ingles_2019", "chapters", "5 results and discussion"))

VAR = [("V25", "lenguaje_sexista"), ("V26", "masc_generico"),
       ("V30", "sexismo_discurso"), ("V33", "asimetria_mujer_hombre"),
       ("V35", "denominacion_sexualizada")]
MOD = ["gemini-3.1-flash-lite", "gpt-4o-mini", "gpt-5.4-nano", "gemma4:e4b"]

rows = list(csv.DictReader(io.open(os.path.join(BASE, "metricas_por_variable.csv"), encoding="utf-8")))
d = {(r["codigo"], r["modelo"]): r for r in rows if r["nivel"] == "B1"}

def pct(x):
    return f"{float(x) * 100:.1f}".replace(".", ",")

o = io.StringIO()
o.write("% Generado por gen_tabla_prev_modelo.py. No editar a mano.\n\n")
o.write(r"""\begin{table}[H]
    \centering
    \begingroup
    \scriptsize
    \renewcommand{\texttt}[1]{{\ttfamily #1}}
    \ttabbox[\FBwidth]{
        \caption{Prevalencia de \textit{Sí} (\%) de cada modelo por variable, en el nivel B1 ($N=1.313$). La columna \textbf{Prev. Sí} (sombreada) es la prevalencia anotada por las expertas, que sirve de referencia.}
        \label{tab:iris-prev-modelo-var}
    }{
        \begin{tabularx}{\textwidth}{
            >{\hsize=2\hsize\raggedright\arraybackslash\ttfamily}X
            >{\hsize=0.7875\hsize\columncolor{gray!12}\centering\arraybackslash}X
            >{\hsize=0.7875\hsize\centering\arraybackslash}X
            >{\hsize=0.7875\hsize\centering\arraybackslash}X
            >{\hsize=0.7875\hsize\centering\arraybackslash}X
            >{\hsize=0.7875\hsize\centering\arraybackslash}X
        }
            \toprule
            \normalfont\textbf{Variable} & \textbf{Prev. Sí} & \textbf{Gemini} & \textbf{GPT-4o-mini} & \textbf{GPT-5.4-nano} & \textbf{Gemma} \\
            \midrule
""")
for cod, nom in VAR:
    prev = pct(d[(cod, MOD[0])]["prev_real"])
    # Sin resaltar ninguna celda: acercarse a la prevalencia anotada no significa
    # marcar las mismas piezas, de modo que destacarlo induciria a error.
    celdas = [pct(d[(cod, m)]["prev_pred"]) for m in MOD]
    o.write(f"            {nom.replace('_', chr(92) + '_')} & {prev} & " + " & ".join(celdas) + r" \\" + "\n")
o.write(r"""            \bottomrule
        \end{tabularx}
    }
    \endgroup
\end{table}
""")
dest = os.path.join(DEST, "tabla_prev_modelo.tex")
io.open(dest, "w", encoding="utf-8").write(o.getvalue())
print("escrito:", dest)
