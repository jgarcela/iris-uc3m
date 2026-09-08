# -*- coding: utf-8 -*-
"""Tablas por variable: B0 frente a B1 con todas las metricas."""
import csv, os, io
BASE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(BASE, "metricas_por_variable.csv"))))

MOD = [("gemini-3.1-flash-lite", r"\texttt{gemini-3.1-flash-lite}"),
       ("gpt-4o-mini",           r"\texttt{gpt-4o-mini}"),
       ("gpt-5.4-nano",          r"\texttt{gpt-5.4-nano}"),
       ("gemma4:e4b",            r"\texttt{gemma4:e4b}")]
COLS = [("prev_pred", "Prev. pred."), ("exactitud", "Exac."), ("precision", "Prec."),
        ("recall", "Recall"), ("f1_pos", r"F1 (\textit{Sí})"),
        ("f1_macro", "F1 macro"), ("kappa", r"$\kappa$")]
VAR = [("V25", r"lenguaje\_sexista"), ("V26", r"masc\_generico"),
       ("V30", r"sexismo\_discurso"), ("V33", r"asimetria\_mujer\_hombre"),
       ("V35", r"denominacion\_sexualizada")]

def num(x):
    v = float(x)
    if abs(v) < 5e-4:
        v = 0.0
    return f"{v:.3f}".replace(".", ",").replace("-", "$-$")

DEST = os.path.normpath(os.path.join(BASE, "..", "TFM", "TFM___JORGE_GARCELA_N_GO_MEZ",
        "Plantilla_TFG_ingles_2019", "chapters", "5 results and discussion"))
for cod, nom in VAR:
    o = io.StringIO()
    o.write("% Generado por gen_tablas_porvariable.py. No editar a mano.\n")
    sub = [r for r in rows if r["codigo"] == cod]
    prev = num(sub[0]["prev_real"])
    o.write("\n\\begin{table}[H]\n    \\centering\n    \\begingroup\n    \\scriptsize\n")
    o.write("    \\setlength{\\tabcolsep}{4pt}\n    \\ttabbox[\\FBwidth]{\n")
    o.write(f"        \\caption{{Resultados en \\texttt{{{nom}}} ({cod}) por modelo y nivel "
            f"($N=1.313$, prevalencia anotada {prev}).}}\n")
    o.write(f"        \\label{{tab:var-{cod.lower()}}}\n    }}{{\n")
    o.write("        \\begin{tabular}{@{}l l" + " c" * len(COLS) + "@{}}\n            \\toprule\n")
    o.write("            \\textbf{Modelo} & \\textbf{Nivel} & "
            + " & ".join("\\textbf{%s}" % t for _, t in COLS) + " \\\\\n")
    for mk, mlab in MOD:
        o.write("            \\midrule\n")
        for i, niv in enumerate(["B0", "B1"]):
            r = [x for x in sub if x["modelo"] == mk and x["nivel"] == niv][0]
            cells = []
            for c, _ in COLS:
                # Sin resaltar el maximo: en las variables de prevalencia extrema el
                # mejor valor de exactitud o de precision suele venir de un modelo que
                # apenas marca, de modo que destacarlo induciria a error.
                cells.append(num(float(r[c])))
            o.write(f"            {mlab if i == 0 else ''} & {niv} & " + " & ".join(cells) + " \\\\\n")
    o.write("            \\bottomrule\n        \\end{tabular}\n    }\n    \\endgroup\n\\end{table}\n")

    dest = os.path.join(DEST, f"tabla_var_{cod.lower()}.tex")
    open(dest, "w", encoding="utf-8").write(o.getvalue())
    print("escrito:", dest)
