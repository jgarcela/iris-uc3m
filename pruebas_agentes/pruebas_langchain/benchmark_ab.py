import pandas as pd
import time
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
from tqdm import tqdm
from graph import app_graph
from rag_engine import inicializar_vector_db

# --- CONFIGURACIÓN DE METADATOS ---
# Define aquí los nombres de tus modelos para que queden registrados
LLM_MODEL_NAME = "Llama-3.1-8B (Ollama)"     # Ajusta según tu configuración real
EMBEDDING_MODEL_NAME = "nomic-embed-text"    # El que usas en rag_engine.py

FILE_PATH = "/home/jggomez/Desktop/IRIS/iris-uc3m/data/Base de datos Noticias 27102025.xlsx - Noticias_scrape.csv"
TEMP_CSV_FILE = "resultados_parciales_temp.csv" # Archivo de seguridad

COL_MAPPING = {
    'Temática noticias': 'tema_id',
    'Género periodista': 'genero_periodista_id',
    'Cita titular': 'cita_titular_id',
    'Género personas noticias': 'genero_personas_mencionadas_id',
    'Nombre propio titular': 'nombre_propio_titular'
}

def cargar_y_normalizar(limit=None):
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
    
    required = ['texto', 'titular', 'id']
    if not all(col in df.columns for col in required):
        print(f"❌ Faltan columnas clave. Tienes: {list(df.columns)}")
        return None
    
    df['full_text'] = df['titular'].fillna('').astype(str) + " \n " + df['texto'].fillna('').astype(str)
    
    print("🧹 Normalizando etiquetas...")
    for col_excel in COL_MAPPING.keys():
        if col_excel in df.columns:
            df[col_excel] = df[col_excel].astype(str).str.replace('.0', '', regex=False).str.strip().replace('nan', 'Ns/Nc')
        else:
            df[col_excel] = "Ns/Nc"
            
    if limit:
        df = df.head(limit)
    return df

def guardar_fila_parcial(fila_dict):
    """Guarda una única fila en el CSV temporal (Append Mode)"""
    df_row = pd.DataFrame([fila_dict])
    # Si no existe, crea con cabecera. Si existe, añade sin cabecera.
    if not os.path.exists(TEMP_CSV_FILE):
        df_row.to_csv(TEMP_CSV_FILE, index=False, encoding='utf-8-sig')
    else:
        df_row.to_csv(TEMP_CSV_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

def obtener_procesados_previos():
    """Lee el CSV temporal para saber qué IDs ya hemos terminado"""
    if not os.path.exists(TEMP_CSV_FILE):
        return set()
    try:
        df_temp = pd.read_csv(TEMP_CSV_FILE)
        # Creamos una clave única: ID + MODO (para distinguir SIN RAG de CON RAG)
        procesados = set(df_temp['ID'].astype(str) + "_" + df_temp['MODO'])
        return procesados
    except:
        return set()

def ejecutar_ronda(df, usar_rag):
    modo_label = "CON_RAG" if usar_rag else "SIN_RAG"
    modo_visual = "CON RAG" if usar_rag else "SIN RAG"
    
    # Recuperamos qué ya se hizo para no repetir
    ya_procesados = obtener_procesados_previos()
    
    resultados_ronda = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"   🤖 {modo_visual}", unit="noticia", ncols=100):
        id_str = str(row['id'])
        clave_unica = f"{id_str}_{modo_label}"
        
        # 1. CHECK DE REANUDACIÓN
        if clave_unica in ya_procesados:
            # Si ya existe en el CSV, cargamos esos datos (opcional) o simplemente saltamos
            # Para simplificar, saltamos el procesamiento y no lo añadimos a la lista RAM todavía
            # (Lo leeremos todo junto del CSV al final)
            continue

        # 2. PROCESAMIENTO
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
        tiempo_seg = end_t - start_t
        
        # 3. CONSTRUIR FILA PLANA PARA CSV
        fila_save = {
            "ID": id_str,
            "MODO": modo_label, # Clave para separar rondas
            "TITULAR": row['titular'][:100],
            "TIMESTAMP": timestamp,
            "TIEMPO_SEG": round(tiempo_seg, 2),
            "MODELO_LLM": LLM_MODEL_NAME,
            "MODELO_EMBEDDING": EMBEDDING_MODEL_NAME if usar_rag else "N/A",
            "RAZONAMIENTO": res_dict.get('razonamiento', '')
        }
        
        # Añadir las variables predichas y reales
        for col_excel, col_model in COL_MAPPING.items():
            val_real = row[col_excel]
            val_pred = res_dict.get(col_model, 'Error')
            
            fila_save[f"{col_model.upper()}_REAL"] = val_real
            fila_save[f"{col_model.upper()}_PRED"] = val_pred
            fila_save[f"{col_model.upper()}_MATCH"] = 1 if val_pred == val_real else 0

        # 4. GUARDAR INMEDIATAMENTE
        guardar_fila_parcial(fila_save)
        resultados_ronda.append(fila_save)

    return resultados_ronda

def calcular_reporte_cientifico(df_completo):
    print("\n" + "="*100)
    print("🔬 REPORTE CIENTÍFICO DE MÉTRICAS (Weighted Avg)")
    print("="*100)
    
    metricas = []
    
    for modo in ["SIN_RAG", "CON_RAG"]:
        subset = df_completo[df_completo['MODO'] == modo]
        if subset.empty: continue
        
        for _, col_model in COL_MAPPING.items():
            col_real = f"{col_model.upper()}_REAL"
            col_pred = f"{col_model.upper()}_PRED"
            
            y_true = subset[col_real].tolist()
            y_pred = subset[col_pred].tolist()
            
            # Filtramos 'Error' o 'Ns/Nc' si quieres métricas puras, 
            # o los dejamos para que penalicen. Aquí los dejamos.
            
            acc = accuracy_score(y_true, y_pred)
            # 'weighted': calcula metricas para cada etiqueta y hace la media ponderada por soporte
            # 'zero_division=0': evita crash si una clase no se predice nunca
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            metricas.append([
                modo,
                col_model,
                f"{acc:.1%}",
                f"{prec:.1%}",
                f"{rec:.1%}",
                f"{f1:.1%}"
            ])
            
    print(tabulate(metricas, headers=["Modo", "Variable", "Accuracy", "Precision", "Recall", "F1-Score"], tablefmt="github"))

def consolidar_resultados_finales():
    """Lee el CSV temporal completo y genera el Excel final bonito"""
    if not os.path.exists(TEMP_CSV_FILE):
        print("⚠️ No hay datos procesados para generar reporte.")
        return

    print("\n💾 Consolidando datos y generando Excel final...")
    df_res = pd.read_csv(TEMP_CSV_FILE)
    
    # Calcular métricas antes de pivotar
    calcular_reporte_cientifico(df_res)
    
    # Generar Excel Comparativo (Pivotar la tabla para poner SIN/CON lado a lado)
    # Separamos en dos DF
    df_sin = df_res[df_res['MODO'] == 'SIN_RAG'].set_index('ID')
    df_con = df_res[df_res['MODO'] == 'CON_RAG'].set_index('ID')
    
    # Lista combinada
    data_final = []
    # Usamos los IDs que existan en cualquiera de los dos
    all_ids = set(df_sin.index).union(set(df_con.index))
    
    for uid in all_ids:
        row_sin = df_sin.loc[uid] if uid in df_sin.index else {}
        row_con = df_con.loc[uid] if uid in df_con.index else {}
        
        # Datos base (cogemos de uno de los dos)
        item = {
            "ID": uid,
            "TITULAR": row_sin.get('TITULAR', row_con.get('TITULAR', '')),
            "LLM": LLM_MODEL_NAME
        }
        
        # Métricas de tiempo
        item["TIEMPO_SIN"] = row_sin.get('TIEMPO_SEG', 0)
        item["TIEMPO_CON"] = row_con.get('TIEMPO_SEG', 0)
        
        # Razonamientos
        item["RAZONAMIENTO_SIN"] = row_sin.get('RAZONAMIENTO', '')
        item["RAZONAMIENTO_CON"] = row_con.get('RAZONAMIENTO', '')
        
        # Variables
        for _, col_model in COL_MAPPING.items():
            base = col_model.upper()
            # Real (debería ser igual en ambos)
            real = row_sin.get(f"{base}_REAL", row_con.get(f"{base}_REAL", ''))
            
            pred_sin = row_sin.get(f"{base}_PRED", '-')
            pred_con = row_con.get(f"{base}_PRED", '-')
            
            item[f"{base}_REAL"] = real
            item[f"{base}_SIN"] = pred_sin
            item[f"{base}_CON"] = pred_con
            item[f"{base}_CHECK"] = "✅" if str(pred_con) == str(real) else "❌"
            
        data_final.append(item)
        
    df_final = pd.DataFrame(data_final)
    output_xlsx = "resultados_finales_metricas.xlsx"
    df_final.to_excel(output_xlsx, index=False)
    print(f"✅ Reporte guardado: {os.path.abspath(output_xlsx)}")
    print(f"ℹ️  El archivo temporal '{TEMP_CSV_FILE}' se mantiene por seguridad.")

def main():
    # 1. CARGAR
    df = cargar_y_normalizar(limit=None) 
    if df is None: return

    # 2. VECTOR DB
    inicializar_vector_db(df, COL_MAPPING)
    
    print("\n" + "="*60)
    print("🚀 INICIANDO BENCHMARK (MODO GUARDADO SEGURO)")
    print("ℹ️  Los resultados se guardan fila a fila en 'resultados_parciales_temp.csv'")
    print("ℹ️  Si paras el script, vuelve a ejecutarlo y continuará donde se quedó.")
    print("="*60)
    
    # 3. RONDAS (La función ahora comprueba internamente si ya existe el ID)
    ejecutar_ronda(df, usar_rag=False)
    print("\n")
    ejecutar_ronda(df, usar_rag=True)
    
    # 4. REPORTING FINAL
    consolidar_resultados_finales()

if __name__ == "__main__":
    main()