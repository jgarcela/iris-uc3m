import pandas as pd
import time
import os
from datetime import datetime
from tqdm import tqdm

# --- IMPORTS DINÁMICOS ---
from graph import app_graph, MODEL_NAME 
from utils import extraer_datos_newspaper, cargar_config_ini

# --- CONFIGURACIÓN ---
LLM_MODEL_NAME = MODEL_NAME
CONFIG = cargar_config_ini()

FILE_PATH = "/home/jggomez/Desktop/IRIS/iris-uc3m/data/Base de datos Noticias 27102025.xlsx - Noticias_scrape.csv"
OUTPUT_EXCEL = "resultados_extraccion_noticias.xlsx"

# Limitar número de noticias a procesar (None = todas, o un número para pruebas)
NUM_NOTICIAS_PRUEBA = 10  # Cambiar a None para procesar todas

def validar_codigo(variable, codigo, config_data):
    """Valida que un código sea válido según el config.ini"""
    contenido_general = config_data.get("CONTENIDO_GENERAL", {})
    
    # Mapeo de variables a sus diccionarios en el config
    mapeo_variables = {
        'tema_id': 'TEMA',
        'genero_periodista_id': 'GENERO_PERIODISTA',
        'cita_titular_id': 'CITA_TITULAR',
        'genero_personas_mencionadas_id': 'GENERO_PERSONAS_MENCIONADAS',
        'genero_nombre_propio_titular_id': 'GENERO_NOMBRE_PROPIO_TITULAR',
        'personas_mencionadas_id': 'PERSONAS_MENCIONADAS'
    }
    
    nombre_config = mapeo_variables.get(variable)
    if not nombre_config:
        return codigo  # Si no está en el mapeo, devolver tal cual
    
    codigos_validos = contenido_general.get(nombre_config, {})
    
    # Convertir código a string para comparar
    codigo_str = str(codigo).strip()
    
    # Verificar si el código es válido
    if codigo_str in codigos_validos:
        return codigo_str
    else:
        # Si no es válido, devolver el código por defecto más común (Ns/Nc o '1')
        print(f"⚠️  Código inválido '{codigo_str}' para {variable}, usando valor por defecto")
        return '4' if 'GENERO' in nombre_config else '1'

def cargar_datos():
    """Carga el archivo CSV con las noticias"""
    print(f"📂 Cargando archivo original: {FILE_PATH}")
    try:
        if FILE_PATH.endswith('.csv'):
            try: 
                df = pd.read_csv(FILE_PATH, sep=',')
            except: 
                df = pd.read_csv(FILE_PATH, sep=';')
        else:
            df = pd.read_excel(FILE_PATH)
    except Exception as e:
        print(f"❌ Error cargando archivo: {e}")
        return None
    
    print(f"✅ Archivo cargado: {len(df)} noticias encontradas")
    return df

def procesar_noticia(row, idx):
    """Procesa una noticia: extrae datos de newspaper y analiza con gemma"""
    try:
        # Obtener URL de la columna "Pagina"
        url = row.get('Pagina', None)
        
        if pd.isna(url) or url == '':
            print(f"⚠️  Fila {idx}: URL vacía, saltando...")
            return None
        
        # Extraer datos con newspaper
        datos_newspaper = extraer_datos_newspaper(url)
        
        if not datos_newspaper:
            print(f"⚠️  Fila {idx}: No se pudo extraer datos de {url}")
            return None
        
        # Preparar datos para gemma
        titular = datos_newspaper.get('Titular', '')
        texto = datos_newspaper.get('textonoticia', '') or datos_newspaper.get('Contenido', '')
        autor = datos_newspaper.get('Autor', '')
        
        # Analizar con gemma
        inputs = {
            "id_noticia": str(idx),
            "titular": titular,
            "texto_noticia": texto,
            "autor": autor,
            "intentos": 0
        }
        
        start_t = time.time()
        try:
            out = app_graph.invoke(inputs)
            res_dict = out['resultado'].model_dump() if out.get('resultado') else {}
            
            # Validar y corregir códigos según config.ini
            if res_dict:
                res_dict['tema_id'] = validar_codigo('tema_id', res_dict.get('tema_id', ''), CONFIG)
                res_dict['genero_periodista_id'] = validar_codigo('genero_periodista_id', res_dict.get('genero_periodista_id', ''), CONFIG)
                res_dict['cita_titular_id'] = validar_codigo('cita_titular_id', res_dict.get('cita_titular_id', ''), CONFIG)
                res_dict['genero_personas_mencionadas_id'] = validar_codigo('genero_personas_mencionadas_id', res_dict.get('genero_personas_mencionadas_id', ''), CONFIG)
                res_dict['genero_nombre_propio_titular_id'] = validar_codigo('genero_nombre_propio_titular_id', res_dict.get('genero_nombre_propio_titular_id', ''), CONFIG)
                res_dict['personas_mencionadas_id'] = validar_codigo('personas_mencionadas_id', res_dict.get('personas_mencionadas_id', ''), CONFIG)
        except Exception as e:
            print(f"⚠️  Error en análisis gemma para fila {idx}: {e}")
            res_dict = {}
        
        end_t = time.time()
        
        # Construir resultado final
        resultado = {
            # Datos de newspaper
            'Autor': datos_newspaper.get('Autor', ''),
            'MES': datos_newspaper.get('MES', ''),
            'MMCC': datos_newspaper.get('MMCC', ''),
            'Titular': datos_newspaper.get('Titular', ''),
            'Contenido': datos_newspaper.get('Contenido', ''),
            'Caracteres': datos_newspaper.get('Caracteres', 0),
            'Pagina': datos_newspaper.get('Pagina', ''),
            'textonoticia': datos_newspaper.get('textonoticia', ''),
            'nombre_periodista': datos_newspaper.get('nombre_periodista', ''),
            
            # Datos de gemma (validados)
            'Nombre propio titular': res_dict.get('nombre_propio_titular', ''),
            'Cita titular': res_dict.get('cita_titular_id', ''),
            'Género periodista': res_dict.get('genero_periodista_id', ''),
            'Género personas noticias': res_dict.get('genero_personas_mencionadas_id', ''),
            'Temática noticias': res_dict.get('tema_id', ''),
            
            # Metadatos
            'TIEMPO_PROCESAMIENTO': round(end_t - start_t, 2),
            'TIMESTAMP': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'MODELO_LLM': LLM_MODEL_NAME
        }
        
        return resultado
        
    except Exception as e:
        print(f"❌ Error procesando fila {idx}: {e}")
        return None

def main():
    """Función principal"""
    df = cargar_datos()
    if df is None:
        return
    
    # Limitar número de noticias si está configurado
    if NUM_NOTICIAS_PRUEBA is not None and NUM_NOTICIAS_PRUEBA > 0:
        df = df.head(NUM_NOTICIAS_PRUEBA)
        print(f"⚠️  MODO PRUEBA: Procesando solo {NUM_NOTICIAS_PRUEBA} noticias")
    
    print("\n" + "="*60)
    print("🚀 PROCESAMIENTO DE NOTICIAS")
    print("="*60)
    print(f"📊 Total de noticias a procesar: {len(df)}")
    print(f"🤖 Modelo LLM: {LLM_MODEL_NAME}")
    print("="*60 + "\n")
    
    resultados = []
    errores = []
    
    # Procesar cada noticia
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Procesando noticias", unit="noticia", ncols=100):
        resultado = procesar_noticia(row, idx)
        
        if resultado:
            resultados.append(resultado)
        else:
            errores.append(idx)
        
        # Pequeña pausa para no saturar el sistema
        time.sleep(0.3)
    
    # Crear DataFrame con resultados
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        
        # Guardar Excel
        df_resultados.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n✅ Excel guardado: {os.path.abspath(OUTPUT_EXCEL)}")
        print(f"📊 Total de noticias procesadas exitosamente: {len(resultados)}")
        
        if errores:
            print(f"⚠️  Noticias con errores: {len(errores)}")
    else:
        print("❌ No se procesó ninguna noticia exitosamente")
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()
