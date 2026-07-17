import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================

ARCHIVO_INPUT = "../../data/COMPARACIÓN-03_2026_resultados_modelo_2024_scrape - PROTAGONISTAS.csv"
CARPETA_OUTPUT = "./resultados"

ETIQUETAS_GENERO = {1: "Hombre", 2: "Mujer", 3: "Mixto", 4: "Neutro/Institucional"}

EXPERIMENTOS = {
    "EXP-5": {
        "pred_col": "EXP-5_recod_cla_genero_prota",
        "coincidencia_col": "EXP-5_COINCIDENCIA",
    },
    "EXP-8": {
        "pred_col": "EXP-8_recod_cla_genero_prota",
        "coincidencia_col": "EXP-8_COINCIDENCIA",
    },
    "EXP-9": {
        "pred_col": "EXP-9_recod_cla_genero_prota",
        "coincidencia_col": "EXP-9_COINCIDENCIA",
    },
}

COL_REAL = "cla_genero_prota"
COL_NOTICIA = "IdNoticia"
COL_USUARIO = "no_NombreUsuario"

# ==========================================
# CARGA DE DATOS
# ==========================================

print(f"Cargando datos de: {ARCHIVO_INPUT}")
df = pd.read_csv(ARCHIVO_INPUT)
print(f"Total filas: {len(df)} | Noticias únicas: {df[COL_NOTICIA].nunique()} | Usuarios: {df[COL_USUARIO].nunique()}")
print(f"Usuarios encontrados: {sorted(df[COL_USUARIO].unique())}\n")

os.makedirs(CARPETA_OUTPUT, exist_ok=True)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def calcular_metricas(y_true, y_pred, nombre):
    """Calcula métricas de clasificación entre valores reales y predichos."""
    mask = y_true.notna() & y_pred.notna()
    y_t = y_true[mask].astype(int)
    y_p = y_pred[mask].astype(int)

    if len(y_t) == 0:
        return {"Grupo": nombre, "N": 0, "Accuracy": np.nan, "Kappa": np.nan,
                "F1_Macro": np.nan, "F1_Weighted": np.nan, "Coincidencias": 0, "Errores": 0}

    acc = accuracy_score(y_t, y_p)
    try:
        kappa = cohen_kappa_score(y_t, y_p)
    except:
        kappa = np.nan
    f1_mac = f1_score(y_t, y_p, average="macro", zero_division=0)
    f1_w = f1_score(y_t, y_p, average="weighted", zero_division=0)
    coincidencias = (y_t == y_p).sum()
    errores = (y_t != y_p).sum()

    return {
        "Grupo": nombre,
        "N": len(y_t),
        "Accuracy": round(acc, 4),
        "Kappa": round(kappa, 4),
        "F1_Macro": round(f1_mac, 4),
        "F1_Weighted": round(f1_w, 4),
        "Coincidencias": coincidencias,
        "Errores": errores,
    }


def matriz_confusion_str(y_true, y_pred, etiquetas):
    """Genera una matriz de confusión como string formateado."""
    mask = y_true.notna() & y_pred.notna()
    y_t = y_true[mask].astype(int)
    y_p = y_pred[mask].astype(int)
    labels = sorted(set(y_t.unique()) | set(y_p.unique()))
    cm = confusion_matrix(y_t, y_p, labels=labels)
    nombres = [etiquetas.get(l, str(l)) for l in labels]
    df_cm = pd.DataFrame(cm, index=nombres, columns=nombres)
    df_cm.index.name = "Real \\ Predicho"
    return df_cm


# ==========================================
# 1. COMPARATIVA GLOBAL POR EXPERIMENTO
# ==========================================

print("=" * 70)
print("1. COMPARATIVA GLOBAL: EXP-5 vs EXP-8 vs EXP-9")
print("=" * 70)

metricas_globales = []
for exp_name, exp_info in EXPERIMENTOS.items():
    m = calcular_metricas(df[COL_REAL], df[exp_info["pred_col"]], exp_name)
    metricas_globales.append(m)

df_global = pd.DataFrame(metricas_globales)
print(df_global.to_markdown(index=False))
print()

# ==========================================
# 2. COMPARATIVA POR USUARIO
# ==========================================

print("=" * 70)
print("2. COMPARATIVA POR USUARIO (no_NombreUsuario)")
print("=" * 70)

metricas_por_usuario = []
for usuario in sorted(df[COL_USUARIO].unique()):
    df_u = df[df[COL_USUARIO] == usuario]
    for exp_name, exp_info in EXPERIMENTOS.items():
        m = calcular_metricas(df_u[COL_REAL], df_u[exp_info["pred_col"]], f"{usuario} | {exp_name}")
        m["Usuario"] = usuario
        m["Experimento"] = exp_name
        metricas_por_usuario.append(m)

df_usuario = pd.DataFrame(metricas_por_usuario)
cols_orden = ["Usuario", "Experimento", "N", "Accuracy", "Kappa", "F1_Macro", "F1_Weighted", "Coincidencias", "Errores"]
df_usuario = df_usuario[cols_orden]
print(df_usuario.to_markdown(index=False))
print()

# ==========================================
# 3. TASA DE COINCIDENCIA POR USUARIO Y EXPERIMENTO
# ==========================================

print("=" * 70)
print("3. TASA DE COINCIDENCIA (columnas COINCIDENCIA) POR USUARIO")
print("=" * 70)

coincidencias_list = []
for usuario in sorted(df[COL_USUARIO].unique()):
    df_u = df[df[COL_USUARIO] == usuario]
    fila = {"Usuario": usuario, "N_noticias": len(df_u)}
    for exp_name, exp_info in EXPERIMENTOS.items():
        col_coinc = exp_info["coincidencia_col"]
        total_valid = df_u[col_coinc].notna().sum()
        if total_valid > 0:
            tasa = df_u[col_coinc].sum() / total_valid
            fila[f"{exp_name}_Coincide"] = round(tasa, 4)
            fila[f"{exp_name}_N_Coinc"] = int(df_u[col_coinc].sum())
        else:
            fila[f"{exp_name}_Coincide"] = np.nan
            fila[f"{exp_name}_N_Coinc"] = 0
    coincidencias_list.append(fila)

df_coinc = pd.DataFrame(coincidencias_list)
print(df_coinc.to_markdown(index=False))
print()

# ==========================================
# 4. DETALLE POR NOTICIA (IdNoticia)
# ==========================================

print("=" * 70)
print("4. DETALLE POR NOTICIA: Valor real vs predicciones EXP-5/8/9")
print("=" * 70)

cols_detalle = [COL_NOTICIA, COL_USUARIO, COL_REAL]
for exp_name, exp_info in EXPERIMENTOS.items():
    cols_detalle.extend([exp_info["pred_col"], exp_info["coincidencia_col"]])

df_detalle = df[cols_detalle].sort_values([COL_NOTICIA, COL_USUARIO])
print(f"Mostrando primeras 30 filas (de {len(df_detalle)} totales):")
print(df_detalle.head(30).to_markdown(index=False))
print()

# ==========================================
# 5. NOTICIAS DONDE TODOS LOS EXPERIMENTOS FALLAN
# ==========================================

print("=" * 70)
print("5. NOTICIAS DONDE NINGÚN EXPERIMENTO ACIERTA")
print("=" * 70)

df["todos_fallan"] = (
    (~df["EXP-5_COINCIDENCIA"]) &
    (~df["EXP-8_COINCIDENCIA"]) &
    (~df["EXP-9_COINCIDENCIA"])
)
df_fallan = df[df["todos_fallan"]]
print(f"Total noticias donde los 3 experimentos fallan: {len(df_fallan)} de {len(df)}")
if len(df_fallan) > 0:
    cols_mostrar = [COL_NOTICIA, COL_USUARIO, COL_REAL,
                    "EXP-5_recod_cla_genero_prota", "EXP-8_recod_cla_genero_prota",
                    "EXP-9_recod_cla_genero_prota"]
    print(df_fallan[cols_mostrar].head(20).to_markdown(index=False))
print()

# ==========================================
# 6. NOTICIAS DONDE HAY DISCREPANCIA ENTRE EXPERIMENTOS
# ==========================================

print("=" * 70)
print("6. NOTICIAS DONDE LOS EXPERIMENTOS NO COINCIDEN ENTRE SÍ")
print("=" * 70)

df["exp_discrepancia"] = (
    (df["EXP-5_recod_cla_genero_prota"] != df["EXP-8_recod_cla_genero_prota"]) |
    (df["EXP-5_recod_cla_genero_prota"] != df["EXP-9_recod_cla_genero_prota"]) |
    (df["EXP-8_recod_cla_genero_prota"] != df["EXP-9_recod_cla_genero_prota"])
)
df_disc = df[df["exp_discrepancia"]]
print(f"Total noticias con discrepancia entre experimentos: {len(df_disc)} de {len(df)}")
if len(df_disc) > 0:
    cols_mostrar = [COL_NOTICIA, COL_USUARIO, COL_REAL,
                    "EXP-5_recod_cla_genero_prota", "EXP-8_recod_cla_genero_prota",
                    "EXP-9_recod_cla_genero_prota"]
    print(df_disc[cols_mostrar].head(20).to_markdown(index=False))
print()

# ==========================================
# 7. MATRICES DE CONFUSIÓN GLOBALES
# ==========================================

print("=" * 70)
print("7. MATRICES DE CONFUSIÓN POR EXPERIMENTO (Global)")
print("=" * 70)

for exp_name, exp_info in EXPERIMENTOS.items():
    print(f"\n--- {exp_name} ---")
    cm = matriz_confusion_str(df[COL_REAL], df[exp_info["pred_col"]], ETIQUETAS_GENERO)
    print(cm.to_markdown())
    print()

# ==========================================
# 8. GUARDAR RESULTADOS EN CSV
# ==========================================

print("=" * 70)
print("8. GUARDANDO RESULTADOS EN CSV")
print("=" * 70)

df_global.to_csv(f"{CARPETA_OUTPUT}/comparativa_global_exp5_8_9.csv", index=False)
print(f"  -> {CARPETA_OUTPUT}/comparativa_global_exp5_8_9.csv")

df_usuario.to_csv(f"{CARPETA_OUTPUT}/comparativa_por_usuario_exp5_8_9.csv", index=False)
print(f"  -> {CARPETA_OUTPUT}/comparativa_por_usuario_exp5_8_9.csv")

df_coinc.to_csv(f"{CARPETA_OUTPUT}/tasa_coincidencia_por_usuario.csv", index=False)
print(f"  -> {CARPETA_OUTPUT}/tasa_coincidencia_por_usuario.csv")

df_detalle.to_csv(f"{CARPETA_OUTPUT}/detalle_por_noticia.csv", index=False)
print(f"  -> {CARPETA_OUTPUT}/detalle_por_noticia.csv")

cols_fallan = [COL_NOTICIA, COL_USUARIO, COL_REAL,
               "EXP-5_recod_cla_genero_prota", "EXP-8_recod_cla_genero_prota",
               "EXP-9_recod_cla_genero_prota"]
df_fallan[cols_fallan].to_csv(f"{CARPETA_OUTPUT}/noticias_todos_fallan.csv", index=False)
print(f"  -> {CARPETA_OUTPUT}/noticias_todos_fallan.csv")

df_disc[cols_fallan].to_csv(f"{CARPETA_OUTPUT}/noticias_discrepancia_entre_exp.csv", index=False)
print(f"  -> {CARPETA_OUTPUT}/noticias_discrepancia_entre_exp.csv")

print("\n¡Comparativas completadas!")
