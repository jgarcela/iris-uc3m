# -*- coding: utf-8 -*-
"""Tabla de sintesis por variable: la lectura de conjunto que si admite el capitulo.

Una fila por variable, con el mejor modelo del nivel desplegado (B1), su mejor
configuracion entre las cuatro de la ablacion, el coste y un veredicto. Escribe
tabla_sintesis.tex. No editar la tabla a mano.
"""
import csv, io, os

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.normpath(os.path.join(BASE, "..", "TFM", "TFM___JORGE_GARCELA_N_GO_MEZ",
        "Plantilla_TFG_ingles_2019", "chapters", "5 results and discussion"))

def leer(f):
    return list(csv.DictReader(io.open(os.path.join(BASE, f), encoding="utf-8")))

met, abl, cos = leer("metricas_por_variable.csv"), leer("ablacion_por_variable.csv"), leer("coste_por_variable.csv")

VAR = [("V25", "lenguaje sexista"), ("V26", "masculino genérico"),
       ("V30", "sexismo en el discurso"), ("V33", "asimetría mujer/hombre"),
       ("V35", "denominación sexualizada")]
CFG = {"abl_minimo": "solo \\textit{skills}", "abl_singuia": "\\textit{skills} + resúmenes",
       "abl_sinres": "\\textit{skills} + RAG", "B1": "completa"}
CORTO = {"gemini-3.1-flash-lite": "gemini", "gpt-4o-mini": "gpt-4o-mini",
         "gpt-5.4-nano": "gpt-5.4-nano", "gemma4:e4b": "gemma"}

def num(x):
    v = float(x)
    if abs(v) < 5e-4:
        v = 0.0
    return f"{v:.3f}".replace(".", ",").replace("-", "$-$")

o = io.StringIO()
o.write("% Generado por gen_tabla_sintesis.py. No editar a mano.\n\n")
o.write(r"""\begin{table}[H]
    \centering
    \begingroup
    \footnotesize
    \setlength{\tabcolsep}{4pt}
    \ttabbox[\FBwidth]{
        \caption{Síntesis por variable ($N=1.313$). \textbf{Mejor modelo} es el de mayor F1 de la clase positiva en el nivel desplegado (B1), y \textbf{recall} y \textbf{F1} son los suyos. \textbf{Mejor configuración} es la que ese mismo modelo alcanza entre las cuatro de la ablación. El coste suma los tres modelos accesibles por API. Se considera que la variable admite apoyo automático cuando ese modelo detecta más de la mitad de lo que marcan las expertas con un acuerdo por encima del azar.}
        \label{tab:iris-sintesis}
    }{
        \begin{tabularx}{\textwidth}{
            >{\hsize=1.45\hsize\raggedright\arraybackslash}X
            >{\hsize=0.62\hsize\centering\arraybackslash}X
            >{\hsize=1.0\hsize\centering\arraybackslash}X
            >{\hsize=0.62\hsize\centering\arraybackslash}X
            >{\hsize=0.62\hsize\centering\arraybackslash}X
            >{\hsize=1.32\hsize\centering\arraybackslash}X
            >{\hsize=0.62\hsize\centering\arraybackslash}X
            >{\hsize=0.75\hsize\columncolor{gray!15}\centering\arraybackslash}X
        }
            \toprule
            \textbf{Variable} & \textbf{Prev.} & \textbf{Mejor modelo} & \textbf{Recall} & \textbf{F1 (\textit{Sí})} & \textbf{Mejor config.} & \textbf{USD} & \textbf{¿Apoyo?} \\
            \midrule
""")
for cod, nom in VAR:
    b1 = [r for r in met if r["codigo"] == cod and r["nivel"] == "B1"]
    mej = max(b1, key=lambda r: float(r["f1_pos"]))
    m = mej["modelo"]
    mc = max([r for r in abl if r["codigo"] == cod and r["modelo"] == m],
             key=lambda r: float(r["f1_pos"]))
    usd = sum(float(r["coste"]) for r in cos if r["codigo"] == cod and r["nivel"] == "B1")
    apoyo = "Sí" if float(mej["recall"]) > 0.5 and float(mej["kappa"]) > 0 else "No"
    coste = f"{usd:.2f}".replace(".", ",")
    o.write(f"            {nom} ({cod}) & {num(mej['prev_real'])} & {CORTO[m]} & "
            f"{num(mej['recall'])} & {num(mej['f1_pos'])} & {CFG[mc['configuracion']]} & "
            f"{coste} & \\textbf{{{apoyo}}} \\\\\n")
o.write(r"""            \bottomrule
        \end{tabularx}
    }
    \endgroup
\end{table}
""")
dest = os.path.join(DEST, "tabla_sintesis.tex")
io.open(dest, "w", encoding="utf-8").write(o.getvalue())
print("escrito:", dest)
