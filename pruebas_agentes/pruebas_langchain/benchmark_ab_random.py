import pandas as pd
import time
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
from tqdm import tqdm

# --- IMPORTS DINÁMICOS ---
from graph import app_graph, MODEL_NAME 
from rag_engine import inicializar_vector_db, EMBEDDING_MODEL 

# --- CONFIGURACIÓN ---
LLM_MODEL_NAME = MODEL_NAME
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL

FILE_PATH = "/home/jggomez/Desktop/IRIS/iris-uc3m/data/Base de datos Noticias 27102025.xlsx - Noticias_scrape.csv"
TEMP_CSV_FILE = "resultados_parciales_temp.csv"

COL_MAPPING = {
    'Temática noticias': 'tema_id',
    'Género periodista': 'genero_periodista_id',
    'Cita titular': 'cita_titular_id',
    'Género personas noticias': 'genero_personas_mencionadas_id',
    'Nombre propio titular': 'nombre_propio_titular'
}

# --- FUNCIÓN DE LIMPIEZA / NORMALIZACIÓN ---
def normalizar_valor(valor):
    """
    Intenta estandarizar la salida para que '1', '1.0' y '1 ' sean iguales.
    Si el LLM devuelve '1 - Política', intenta quedarse con lo relevante si es posible,
    pero por defecto limpiamos espacios y tipos numéricos.
    """
    if valor is None:
        return "Ns/Nc"
    
    s_val = str(valor).strip()
    
    # Quitar el .0 típico de pandas/excel en números
    if s_val.endswith('.0'):
        s_val = s_val[:-2]
        
    # Convertir 'nan' o vacíos a Ns/Nc explícito
    if s_val.lower() == 'nan' or s_val == '':
        return "Ns/Nc"
        
    return s_val.lower() # Devolvemos en minúsculas para comparar mejor

def check_match(real, pred):
    """Compara dos valores normalizados"""
    val_real = normalizar_valor(real)
    val_pred = normalizar_valor(pred)
    
    # 1. Coincidencia exacta
    if val_real == val_pred:
        return True
        
    # 2. Heurística simple: Si el real es 'política' y el pred es '1 - política', asumimos match
    # (Ojo: esto puede dar falsos positivos si no se tiene cuidado, pero ayuda con LLMs verborrágicos)
    if (val_real in val_pred) and (len(val_real) > 2): 
        return True
        
    return False

def cargar_y_normalizar():
    print(f"📂 Cargando archivo original: {FILE_PATH}")
    try:
        if FILE_PATH.endswith('.csv'):
            try: df = pd.read_csv(FILE_PATH, sep=',')
            except: df = pd.read_csv(FILE_PATH, sep=';')
        else:
            df = pd.read_excel(FILE_PATH)
    except Exception as e:
        print(f"❌ Error cargando archivo: {e}")
        return None

    renames = {'contenido_articulo': 'texto', 'Titular': 'titular', 'IdNoticia': 'id'}
    df.rename(columns=renames, inplace=True)
    
    # Normalizamos el dataframe original para evitar problemas de tipos float/int
    print("🧹 Normalizando etiquetas del Excel...")
    for col_excel in COL_MAPPING.keys():
        if col_excel in df.columns:
            # Forzamos todo a string limpio desde el inicio
            df[col_excel] = df[col_excel].apply(normalizar_valor)
        else:
            df[col_excel] = "ns/nc"

    df['full_text'] = df['titular'].fillna('').astype(str) + " \n " + df['texto'].fillna('').astype(str)
    
    print("\n🎲 Seleccionando 100 noticias ALEATORIAS...")
    if len(df) > 100:
        df = df.sample(n=100, random_state=42)
    
    return df

def guardar_fila_parcial(fila_dict):
    df_row = pd.DataFrame([fila_dict])
    if not os.path.exists(TEMP_CSV_FILE):
        df_row.to_csv(TEMP_CSV_FILE, index=False, encoding='utf-8-sig')
    else:
        df_row.to_csv(TEMP_CSV_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

def obtener_procesados_previos():
    if not os.path.exists(TEMP_CSV_FILE):
        return set()
    try:
        df_temp = pd.read_csv(TEMP_CSV_FILE)
        # Convertimos ID a string para asegurar consistencia
        procesados = set(df_temp['ID'].astype(str) + "_" + df_temp['MODO'])
        return procesados
    except:
        return set()

def ejecutar_ronda(df, usar_rag):
    modo_label = "CON_RAG" if usar_rag else "SIN_RAG"
    modo_visual = "CON RAG" if usar_rag else "SIN RAG"
    
    ya_procesados = obtener_procesados_previos()
    resultados_ronda = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"   🤖 {modo_visual}", unit="noticia", ncols=100):
        id_str = str(row['id'])
        clave_unica = f"{id_str}_{modo_label}"
        
        if clave_unica in ya_procesados:
            continue

        inputs = {
            "id_noticia": id_str,
            "texto_noticia": row['full_text'],
            "use_rag": usar_rag,
            "intentos": 0
        }
        
        start_t = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            out = app_graph.invoke(inputs)
            res_dict = out['resultado'].model_dump()
        except:
            res_dict = {}

        end_t = time.time()
        
        # Guardamos RAW en el CSV parcial, la normalización final la hacemos al consolidar el Excel
        fila_save = {
            "ID": id_str,
            "MODO": modo_label,
            "TITULAR": row['titular'][:100],
            "TIMESTAMP": timestamp,
            "TIEMPO_SEG": round(end_t - start_t, 2),
            "MODELO_LLM": LLM_MODEL_NAME,
            "MODELO_EMBEDDING": EMBEDDING_MODEL_NAME if usar_rag else "N/A",
            "RAZONAMIENTO": res_dict.get('razonamiento', '')
        }
        
        for col_excel, col_model in COL_MAPPING.items():
            val_real = row[col_excel] # Ya está normalizado del paso 'cargar'
            val_pred = str(res_dict.get(col_model, 'Error')) # Forzamos string en predicción
            
            fila_save[f"{col_model.upper()}_REAL"] = val_real
            fila_save[f"{col_model.upper()}_PRED"] = val_pred
            # No calculamos match aquí para el CSV temporal, lo hacemos al final para poder ajustar la lógica

        guardar_fila_parcial(fila_save)
        resultados_ronda.append(fila_save)

    return resultados_ronda

def calcular_metricas_robustas(y_true, y_pred, label_prefix):
    """Calcula métricas normalizando antes"""
    y_true_clean = [normalizar_valor(x) for x in y_true]
    y_pred_clean = [normalizar_valor(x) for x in y_pred]
    
    acc = accuracy_score(y_true_clean, y_pred_clean)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, average='weighted', zero_division=0
    )
    return acc, prec, rec, f1

def consolidar_resultados_finales():
    if not os.path.exists(TEMP_CSV_FILE):
        print("⚠️ No hay datos procesados.")
        return

    print("\n💾 Consolidando datos y generando Excel final...")
    df_res = pd.read_csv(TEMP_CSV_FILE)
    
    # Asegurar que ID sea string
    df_res['ID'] = df_res['ID'].astype(str)
    
    # --- REPORTE EN CONSOLA ---
    print("\n" + "="*100)
    print(f"🔬 REPORTE CIENTÍFICO (LLM: {LLM_MODEL_NAME})")
    print("="*100)
    metricas = []
    
    for modo in ["SIN_RAG", "CON_RAG"]:
        subset = df_res[df_res['MODO'] == modo]
        if subset.empty: continue
        
        for _, col_model in COL_MAPPING.items():
            col_real = f"{col_model.upper()}_REAL"
            col_pred = f"{col_model.upper()}_PRED"
            
            acc, prec, rec, f1 = calcular_metricas_robustas(
                subset[col_real].tolist(), 
                subset[col_pred].tolist(),
                col_model
            )
            
            metricas.append([modo, col_model, f"{acc:.1%}", f"{prec:.1%}", f"{rec:.1%}", f"{f1:.1%}" ])
            
    print(tabulate(metricas, headers=["Modo", "Variable", "Accuracy", "Precision", "Recall", "F1-Score"], tablefmt="github"))
    
    # --- GENERACIÓN DE EXCEL FINAL (PIVOT) ---
    df_sin = df_res[df_res['MODO'] == 'SIN_RAG'].set_index('ID')
    df_con = df_res[df_res['MODO'] == 'CON_RAG'].set_index('ID')
    
    data_final = []
    all_ids = set(df_sin.index).union(set(df_con.index))
    
    for uid in all_ids:
        row_sin = df_sin.loc[uid] if uid in df_sin.index else {}
        row_con = df_con.loc[uid] if uid in df_con.index else {}
        
        item = {
            "ID": uid,
            "TITULAR": str(row_sin.get('TITULAR', row_con.get('TITULAR', ''))),
            "LLM": LLM_MODEL_NAME
        }
        
        # Metadatos
        item["TIEMPO_SIN"] = row_sin.get('TIEMPO_SEG', 0)
        item["TIEMPO_CON"] = row_con.get('TIEMPO_SEG', 0)
        item["RAZONAMIENTO_SIN"] = row_sin.get('RAZONAMIENTO', '')
        item["RAZONAMIENTO_CON"] = row_con.get('RAZONAMIENTO', '')
        
        # Variables y Checks independientes
        for _, col_model in COL_MAPPING.items():
            base = col_model.upper()
            
            # Recuperar valores
            val_real = str(row_sin.get(f"{base}_REAL", row_con.get(f"{base}_REAL", '')))
            val_sin = str(row_sin.get(f"{base}_PRED", '-'))
            val_con = str(row_con.get(f"{base}_PRED", '-'))
            
            # Check SIN RAG
            is_match_sin = check_match(val_real, val_sin)
            # Check CON RAG
            is_match_con = check_match(val_real, val_con)
            
            item[f"{base}_REAL"] = val_real
            item[f"{base}_SIN"] = val_sin
            item[f"{base}_CHECK_SIN"] = "✅" if is_match_sin else "❌" # Check INDEPENDIENTE
            
            item[f"{base}_CON"] = val_con
            item[f"{base}_CHECK_CON"] = "✅" if is_match_con else "❌" # Check INDEPENDIENTE

        data_final.append(item)
        
    df_final = pd.DataFrame(data_final)
    output_xlsx = "resultados_finales_comparativos_v2.xlsx"
    df_final.to_excel(output_xlsx, index=False)
    print(f"✅ Excel guardado: {os.path.abspath(output_xlsx)}")

def main():
    df = cargar_y_normalizar() 
    if df is None: return

    inicializar_vector_db(df, COL_MAPPING)
    
    print("\n" + "="*60)
    print("🚀 BENCHMARK (V2: Checks Independientes + Normalización)")
    print("="*60)
    
    ejecutar_ronda(df, usar_rag=False)
    print("\n")
    ejecutar_ronda(df, usar_rag=True)
    
    consolidar_resultados_finales()

if __name__ == "__main__":
    main()