import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Cargar datos
archivo = 'results/Experimento-1_02-2026_resultados_modelo_2024_scrape.csv'
df = pd.read_csv(archivo)

# --- FUNCIONES TRADUCTORAS (Modelo -> Realidad) ---

def traducir_titular(val):
    try:
        x = int(float(val)) # Maneja "1.0" o 1
    except:
        return 1 # Si es error o NaN, asumimos "No hay" (Código 1 Realidad)
    
    if x == 1: return 2       # Hombre
    if x == 2: return 3       # Mujer
    if x in [3, 32, 33]: return 4 # Mixto
    if x in [4, 41, 42]: return 5 # Neutro/Cosas
    return 1 # Por defecto "No hay"

def traducir_protagonista(val):
    try:
        x = int(float(val))
    except:
        return 4 # Ante la duda, Neutro (Código 4 Realidad)
        
    if x == 1: return 1       # Hombre
    if x == 2: return 2       # Mujer
    if x in [3, 32, 33]: return 3 # Mixto
    # 4 (Inst), 41 (Lugar), 42 (IA) -> Neutro
    return 4 

def traducir_periodista(val):
    try:
        x = int(float(val))
    except:
        return 4 # Si falla, Ns/Nc (Código 4 Realidad)
    
    if x in [1, 2, 3, 6, 7]: return x # Coinciden
    if x == 0: return 4  # Modelo 0 (Desc) -> Realidad 4 (Ns/Nc)
    if x in [4, 5]: return 5 # Modelo 4 y 5 -> Realidad 5 (Agencia/Otros)
    return 4

# --- APLICAR TRADUCCIÓN ---

print("Traduciendo códigos del modelo para que coincidan con la realidad...")

df['modelo_recod_nombre_propio_titular'] = df['modelo_nombre_propio_titular'].apply(traducir_titular)
df['modelo_recod_cla_genero_prota'] = df['modelo_cla_genero_prota'].apply(traducir_protagonista)
df['modelo_recod_genero_periodista'] = df['modelo_genero_periodista'].apply(traducir_periodista)

# Asegurar que la realidad sea int (quitar .0)
cols_realidad = ['nombre_propio_titular', 'cla_genero_prota', 'genero_periodista']
for col in cols_realidad:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# --- CÁLCULO DE MÉTRICAS ---

metricas = [
    ('nombre_propio_titular', 'modelo_recod_nombre_propio_titular', "Titular"),
    ('cla_genero_prota', 'modelo_recod_cla_genero_prota', "Protagonista"),
    ('genero_periodista', 'modelo_recod_genero_periodista', "Periodista")
]

print("\n" + "="*50)
print(" RESULTADOS FINALES (YA CORREGIDOS)")
print("="*50)

for real, pred, nombre in metricas:
    # Filtramos filas donde la realidad sea válida (suponiendo que 0 en realidad es 'dato faltante' si tu esquema empieza en 1)
    # Si tus datos reales usan 0, quita el filtro `mask`.
    mask = df[real] != 0 
    
    y_true = df.loc[mask, real]
    y_pred = df.loc[mask, pred]
    
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n📊 VARIABLE: {nombre}")
        print(f"   Accuracy: {acc:.2%}")
        print(f"   F1-Score: {f1:.2%}")
        
        # Detalle por clase para ver dónde falla
        print("\n   Detalle por clase:")
        print(classification_report(y_true, y_pred, zero_division=0))
    else:
        print(f"\n⚠️ {nombre}: No hay datos válidos en la columna Ground Truth.")

# Guardar csv para revisión
df.to_csv(archivo, index=False)