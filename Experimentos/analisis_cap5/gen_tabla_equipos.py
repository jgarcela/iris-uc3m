# -*- coding: utf-8 -*-
"""Tabla de recall y kappa por equipo de anotacion y variable."""
import csv, os, io
BASE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(BASE,"equipos_por_variable.csv"))))
MODS=[("gemini-3.1-flash-lite","gemini"),("gpt-4o-mini","gpt-4o-mini"),
      ("gpt-5.4-nano","gpt-5.4-nano"),("gemma4:e4b","gemma")]
VAR=[("V25",r"lenguaje\_sexista"),("V26",r"masc\_generico"),("V30",r"sexismo\_discurso"),
     ("V33",r"asimetria\_mujer\_hombre"),("V35",r"denominacion\_sexualizada")]
def num(x,pct=False):
    v=float(x)
    if pct: return f"{v*100:.1f}".replace(".",",")
    if abs(v)<5e-4: v=0.0
    return f"{v:.3f}".replace(".",",").replace("-","$-$")
o=io.StringIO()
o.write("% Generado por gen_tabla_equipos.py. No editar a mano.\n")
for metric,nom,lab in [("recall","recall","recall"),("kappa","kappa",r"$\kappa$")]:
    o.write("\n\\begin{table}[H]\n    \\centering\n    \\begingroup\n    \\scriptsize\n")
    o.write("    \\setlength{\\tabcolsep}{4pt}\n    \\renewcommand{\\texttt}[1]{{\\ttfamily #1}}\n")
    o.write("    \\ttabbox[\\FBwidth]{\n")
    if metric=="recall":
        o.write("        \\caption{Prevalencia anotada por cada equipo y "+lab+" de los cuatro modelos "
                "frente a cada uno, por variable (nivel B1). Indexa codificó 548 piezas y UCM3 las 765 "
                "restantes.}\n        \\label{tab:iris-equipos-recall}\n")
    else:
        o.write("        \\caption{Coeficiente "+lab+" de los cuatro modelos frente a cada equipo de "
                "anotación, por variable (nivel B1).}\n        \\label{tab:iris-equipos-kappa}\n")
    o.write("    }{\n        \\begin{tabular}{@{}l l c"+" c"*len(MODS)+"@{}}\n            \\toprule\n")
    head = "\\textbf{Prev.}" if metric=="recall" else "\\textbf{Prev.}"
    o.write("            \\textbf{Variable} & \\textbf{Equipo} & "+head+" & "
            +" & ".join("\\texttt{%s}"%s for _,s in MODS)+" \\\\\n")
    for cod,nomv in VAR:
        o.write("            \\midrule\n")
        for i,eq in enumerate(["Indexa","UCM3"]):
            r=[x for x in rows if x["codigo"]==cod and x["equipo"]==eq][0]
            cells=[num(r[f"{metric}_{m}"]) for m,_ in MODS]
            first = "\\texttt{%s}"%nomv if i==0 else ""
            o.write(f"            {first} & {eq} & {num(r['prev'],pct=True)}\\,\\% & "+" & ".join(cells)+" \\\\\n")
    o.write("            \\bottomrule\n        \\end{tabular}\n    }\n    \\endgroup\n\\end{table}\n")
dest=os.path.normpath(os.path.join(BASE,"..","TFM","TFM___JORGE_GARCELA_N_GO_MEZ","Plantilla_TFG_ingles_2019",
     "chapters","5 results and discussion","tablas_equipos.tex"))
open(dest,"w",encoding="utf-8").write(o.getvalue()); print("escrito:",dest)
