# -*- coding: utf-8 -*-
"""Genera las tablas de ablacion por variable en LaTeX desde ablacion_por_variable.csv.
Sin resaltados: el maximo de una metrica dentro de un modelo suele venir de la
configuracion mas conservadora, de modo que la negrita señalaba artefactos. Que
configuracion conviene en cada variable se razona en el texto."""
import csv, collections, io, os
BASE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(BASE,"ablacion_por_variable.csv"))))
NOM={"B1":"Completo","abl_minimo":"Solo \\textit{skills}","abl_singuia":"\\textit{Skills} + resúmenes","abl_sinres":"\\textit{Skills} + RAG"}
ORD=["abl_minimo","abl_singuia","abl_sinres","B1"]
MOD=[("gemini-3.1-flash-lite","gemini"),
     ("gpt-4o-mini","gpt-4o-mini"),
     ("gpt-5.4-nano","gpt-5.4-nano"),
     ("gemma4:e4b","gemma")]
MET=[("exactitud","Exac."),("precision","Prec."),("recall","Recall"),
     ("f1_pos","F1 (\\textit{Sí})"),("f1_macro","F1 macro"),("kappa","$\\kappa$")]
VAR=[("V25","lenguaje sexista"),("V26","masculino genérico"),("V30","sexismo en el discurso"),
     ("V33","asimetría entre mujeres y hombres"),("V35","denominación sexualizada")]
def num(x):
    v=float(x)
    if abs(v)<5e-4: v=0.0
    s=f"{v:.3f}".replace(".",",")
    return s.replace("-","$-$")
out=io.StringIO()
out.write("% Generado por gen_tablas_ablacion.py. No editar a mano.\n")
for cod,nom in VAR:
    sub=[r for r in rows if r["codigo"]==cod]
    prev=num(sub[0]["prev_real"])
    out.write("\n\\begin{table}[H]\n    \\centering\n    \\begingroup\n    \\small\n")
    out.write("    \\setlength{\\tabcolsep}{4pt}\n")
    out.write("    \\ttabbox[\\FBwidth]{\n")
    out.write(f"        \\caption{{Ablación por componentes en {nom} ({cod}), nivel B1 "
              f"($N=1.313$, prevalencia {prev}). Los modelos se nombran de forma abreviada, gemini por gemini-3.1-flash-lite y gemma por gemma4:e4b.}}\n")
    out.write(f"        \\label{{tab:abl-{cod.lower()}}}\n    }}{{\n")
    out.write("        \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l l"+" c"*len(MET)+"@{}}\n            \\toprule\n")
    out.write("            \\textbf{Modelo} & \\textbf{Configuración} & "+" & ".join("\\textbf{%s}"%t for _,t in MET)+" \\\\\n")
    for mk,mlab in MOD:
        block=[r for r in sub if r["modelo"]==mk]
        block={r["configuracion"]:r for r in block}
        out.write("            \\midrule\n")
        for i,c in enumerate(ORD):
            if c not in block: continue
            r=block[c]
            cells=[]
            for m,_ in MET:
                v=float(r[m]); s=num(v)
                cells.append(s)
            first=mlab if i==0 else ""
            out.write(f"            {first} & {NOM[c]} & "+" & ".join(cells)+" \\\\\n")
    out.write("            \\bottomrule\n        \\end{tabular*}\n    }\n    \\endgroup\n\\end{table}\n")
dest=os.path.join(BASE,"..","TFM","TFM___JORGE_GARCELA_N_GO_MEZ","Plantilla_TFG_ingles_2019","chapters","5 results and discussion","tablas_ablacion.tex")
dest=os.path.normpath(dest)
open(dest,"w",encoding="utf-8").write(out.getvalue())
print("escrito:",dest)
