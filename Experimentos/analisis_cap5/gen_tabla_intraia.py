# -*- coding: utf-8 -*-
"""Tabla de acuerdo entre modelos POR VARIABLE (celdas B0 / B1)."""
import csv, os, io
BASE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(BASE,"kappa_entre_modelos_por_variable.csv"))))
SH={"gemini-3.1-flash-lite":"gemini","gpt-4o-mini":"gpt-4o-mini",
    "gpt-5.4-nano":"gpt-5.4-nano","gemma4:e4b":"gemma"}
PAIRS=[("gemini-3.1-flash-lite","gpt-4o-mini"),("gemini-3.1-flash-lite","gpt-5.4-nano"),
       ("gpt-4o-mini","gpt-5.4-nano"),("gemma4:e4b","gemini-3.1-flash-lite"),
       ("gemma4:e4b","gpt-4o-mini"),("gemma4:e4b","gpt-5.4-nano")]
VAR=["V25","V26","V30","V33","V35"]
def num(v):
    v=float(v)
    if abs(v)<5e-4: v=0.0
    return f"{v:.3f}".replace(".",",").replace("-","$-$")
idx={(r["nivel"],r["codigo"],frozenset((r["modelo_a"],r["modelo_b"]))):r["kappa"] for r in rows}
o=io.StringIO()
o.write("""% Generado por gen_tabla_intraia.py. No editar a mano.
\\begin{table}[H]
    \\centering
    \\begingroup
    \\scriptsize
    \\setlength{\\tabcolsep}{4pt}
    \\renewcommand{\\texttt}[1]{{\\ttfamily #1}}
    \\ttabbox[\\FBwidth]{
        \\caption{Acuerdo entre modelos ($\\kappa$ de Cohen) por variable y nivel ($N=1.313$). Cada celda muestra B0 / B1. No se promedia entre variables, porque sus prevalencias no lo permiten.}
        \\label{tab:iris-intraia}
    }{
        \\begin{tabularx}{\\textwidth}{
            >{\\hsize=1.6\\hsize\\raggedright\\arraybackslash}X
            >{\\hsize=0.88\\hsize\\centering\\arraybackslash}X
            >{\\hsize=0.88\\hsize\\centering\\arraybackslash}X
            >{\\hsize=0.88\\hsize\\centering\\arraybackslash}X
            >{\\hsize=0.88\\hsize\\centering\\arraybackslash}X
            >{\\hsize=0.88\\hsize\\centering\\arraybackslash}X
        }
            \\toprule
            \\textbf{Par de modelos} & \\textbf{V25} & \\textbf{V26} & \\textbf{V30} & \\textbf{V33} & \\textbf{V35} \\\\
            \\midrule
""")
for i,(a,b) in enumerate(PAIRS):
    if i==3: o.write("            \\midrule\n")
    cells=[]
    for v in VAR:
        k0=idx[("B0",v,frozenset((a,b)))]; k1=idx[("B1",v,frozenset((a,b)))]
        cells.append(f"{num(k0)} / {num(k1)}")
    o.write(f"            \\texttt{{{SH[a]}}} $\\leftrightarrow$ \\texttt{{{SH[b]}}} & "+" & ".join(cells)+" \\\\\n")
o.write("""            \\bottomrule
        \\end{tabularx}
    }
    \\endgroup
\\end{table}
""")
dest=os.path.normpath(os.path.join(BASE,"..","TFM","TFM___JORGE_GARCELA_N_GO_MEZ","Plantilla_TFG_ingles_2019","chapters","5 results and discussion","tabla_intraia.tex"))
open(dest,"w",encoding="utf-8").write(o.getvalue()); print("escrito:",dest)
