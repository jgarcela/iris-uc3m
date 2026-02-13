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
OUTPUT_METRICAS_EXCEL = "metricas_comparacion.xlsx"
OUTPUT_COMPARACIONES_EXCEL = "comparaciones_detalladas.xlsx"

# Limitar número de noticias a procesar (None = todas, o un número para pruebas)
NUM_NOTICIAS_PRUEBA = 10  # Cambiar a None para procesar todas

def mapear_texto_a_codigo(texto, codigos_dict):
    """Mapea texto descriptivo a código numérico"""
    if not texto or not codigos_dict:
        return None
    
    texto_lower = str(texto).strip().lower()
    
    # Si ya es un código numérico válido, devolverlo
    if texto_lower in codigos_dict:
        return texto_lower
    
    # Buscar coincidencias en los valores del diccionario
    for codigo, descripcion in codigos_dict.items():
        desc_lower = str(descripcion).lower()
        # Coincidencia exacta o parcial
        if texto_lower == desc_lower or texto_lower in desc_lower or desc_lower in texto_lower:
            return str(codigo)
    
    # Mapeos específicos para valores comunes
    mapeos_especificos = {
        # Género
        'masculino': '1',
        'femenino': '2',
        'hombre': '1',
        'mujer': '2',
        'mixto': '3',
        # Binarios (1=No, 2=Sí) - para variables LENGUAJE_VARS, MENCIONA_IA, etc.
        'sí': '2', 'si': '2', 'yes': '2', 'true': '2',
        'no': '1', 'false': '1',
        # CITA_TITULAR (0=No, 1=Sí) - IMPORTANTE: usa '0' y '1', no '1' y '2'
        'hay cita': '1', 'tiene cita': '1', 'con cita': '1', 'cita': '1',
        'sin cita': '0', 'no hay cita': '0', 'no cita': '0',
        # PERSONAS_MENCIONADAS (1=No, 2=Sí)
        'hay personas': '2', 'tiene personas': '2', 'personas mencionadas': '2',
        'no hay personas': '1', 'sin personas': '1', 'no personas': '1',
    }
    
    if texto_lower in mapeos_especificos:
        return mapeos_especificos[texto_lower]
    
    # Si contiene palabras clave
    if any(palabra in texto_lower for palabra in ['sí', 'si', 'yes', 'hay', 'tiene', 'con']):
        return '2'
    elif any(palabra in texto_lower for palabra in ['no', 'sin', 'ningun']):
        return '1'
    
    return None

def validar_coherencia_genero(res_dict, datos_newspaper):
    """
    Valida la coherencia entre variables de género y sus correspondientes.
    ORDEN LÓGICO: Primero la variable "matriz" (nombre/personas), luego el género.
    No tiene sentido determinar el género si no hay nombre/personas.
    """
    # 1. ORDEN LÓGICO: nombre_propio_titular (matriz) → genero_nombre_propio_titular_id (género)
    nombre_propio = res_dict.get('nombre_propio_titular', '').strip()
    genero_nombre_propio = res_dict.get('genero_nombre_propio_titular_id', '')
    
    # Detectar si el modelo puso el titular completo en lugar del nombre propio
    # Si el nombre propio es muy largo (> 50 caracteres) o contiene signos de interrogación, probablemente es el titular completo
    if nombre_propio and len(nombre_propio) > 50:
        print(f"⚠️  Posible error: nombre_propio_titular parece ser el titular completo ('{nombre_propio[:50]}...'). Corrigiendo a 'No aplica' si no hay nombre claro.")
        # Intentar extraer un nombre propio del texto (buscar palabras que parezcan nombres)
        # Por ahora, si es muy largo, asumimos que no hay nombre propio claro
        nombre_propio = 'No aplica'
        res_dict['nombre_propio_titular'] = 'No aplica'
    
    # Si NO hay nombre propio (matriz = 'No aplica'), entonces género DEBE ser '1' (No hay)
    if not nombre_propio or nombre_propio.lower() in ['no aplica', 'no hay', '']:
        if genero_nombre_propio != '1':
            print(f"⚠️  Incoherencia: nombre_propio_titular='{nombre_propio}' (No hay) pero genero_nombre_propio_titular_id='{genero_nombre_propio}'. Corrigiendo a '1' (No hay)")
            res_dict['genero_nombre_propio_titular_id'] = '1'
    else:
        # Si HAY nombre propio (matriz != 'No aplica'), entonces género NO puede ser '1' (No hay)
        if genero_nombre_propio == '1':
            print(f"⚠️  Incoherencia: nombre_propio_titular='{nombre_propio}' (hay nombre) pero genero_nombre_propio_titular_id='1' (No hay). Intentando inferir género del nombre...")
            # Intentar inferir el género del nombre (muy básico)
            nombre_lower = nombre_propio.lower()
            if any(nombre in nombre_lower for nombre in ['maría', 'ana', 'carmen', 'laura', 'sofía', 'elena', 'patricia', 'marta', 'lucia', 'isabel']):
                res_dict['genero_nombre_propio_titular_id'] = '3'  # Mujer
                print(f"  → Inferido género '3' (Mujer) basado en el nombre")
            elif any(nombre in nombre_lower for nombre in ['pedro', 'juan', 'carlos', 'jose', 'luis', 'miguel', 'antonio', 'francisco', 'david', 'manuel']):
                res_dict['genero_nombre_propio_titular_id'] = '2'  # Hombre
                print(f"  → Inferido género '2' (Hombre) basado en el nombre")
            else:
                # Si no se puede inferir, mantener '1' pero advertir
                print(f"  → No se pudo inferir género, manteniendo '1' (puede ser incorrecto)")
    
    # 2. ORDEN LÓGICO: personas_mencionadas_id (matriz) → genero_personas_mencionadas_id (género)
    personas_mencionadas = res_dict.get('personas_mencionadas_id', '')
    genero_personas = res_dict.get('genero_personas_mencionadas_id', '')
    
    # Si NO hay personas mencionadas (matriz = '1'), entonces género DEBE ser '1' (No hay)
    if personas_mencionadas == '1':
        if genero_personas != '1':
            print(f"⚠️  Incoherencia: personas_mencionadas_id='1' (No) pero genero_personas_mencionadas_id='{genero_personas}'. Corrigiendo a '1' (No hay)")
            res_dict['genero_personas_mencionadas_id'] = '1'
    elif personas_mencionadas == '2':
        # Si HAY personas mencionadas (matriz = '2'), entonces género NO puede ser '1' (No hay)
        if genero_personas == '1':
            print(f"⚠️  Incoherencia: personas_mencionadas_id='2' (Sí hay personas) pero genero_personas_mencionadas_id='1' (No hay). Corrigiendo a '2' (Hombre) por defecto")
            res_dict['genero_personas_mencionadas_id'] = '2'  # Por defecto '2' (Hombre) si hay personas pero no se pudo determinar el género
    
    # 3. Coherencia: genero_periodista_id ↔ nombre_periodista (de newspaper) - solo informativo
    genero_periodista = res_dict.get('genero_periodista_id', '')
    nombre_periodista = datos_newspaper.get('nombre_periodista', '').strip()
    
    # Si el género es Agencia/Redacción/Corporativo ('5', '6', '7'), no hay periodista individual
    if genero_periodista in ['5', '6', '7']:
        # No se requiere nombre de periodista individual
        pass
    else:  # '1', '2', '3', '4' (Masculino/Femenino/Mixto/NsNc)
        # Si hay género individual, debería haber un nombre de periodista
        # Pero esto es solo informativo, no corregimos porque el nombre viene de newspaper
        if not nombre_periodista:
            print(f"ℹ️  Nota: genero_periodista_id='{genero_periodista}' (periodista individual) pero nombre_periodista está vacío (puede ser correcto si newspaper no lo extrajo)")

def validar_codigo(variable, codigo, config_data):
    """Valida que un código sea válido según el config.ini"""
    contenido_general = config_data.get("CONTENIDO_GENERAL", {})
    lenguaje = config_data.get("LENGUAJE", {})
    
    # Mapeo de variables a sus diccionarios en el config
    mapeo_variables = {
        # CONTENIDO_GENERAL
        'tema_id': ('CONTENIDO_GENERAL', 'TEMA'),
        'genero_periodista_id': ('CONTENIDO_GENERAL', 'GENERO_PERIODISTA'),
        'cita_titular_id': ('CONTENIDO_GENERAL', 'CITA_TITULAR'),
        'genero_personas_mencionadas_id': ('CONTENIDO_GENERAL', 'GENERO_PERSONAS_MENCIONADAS'),
        'genero_nombre_propio_titular_id': ('CONTENIDO_GENERAL', 'GENERO_NOMBRE_PROPIO_TITULAR'),
        'personas_mencionadas_id': ('CONTENIDO_GENERAL', 'PERSONAS_MENCIONADAS'),
        'menciona_ia_id': ('CONTENIDO_GENERAL', 'MENCIONA_IA'),
        'ia_tema_central_id': ('CONTENIDO_GENERAL', 'IA_TEMA_CENTRAL'),
        'significado_ia_id': ('CONTENIDO_GENERAL', 'SIGNIFICADO_IA'),
        # LENGUAJE
        'lenguaje_sexista_id': ('LENGUAJE', 'LENGUAJE_SEXISTA'),
        # Variables que usan LENGUAJE_VARS (1=No, 2=Sí)
        'masculino_generico_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'hombre_humanidad_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'uso_dual_zorra_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'no_uso_cargos_mujeres_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'sexismo_social_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'androcentrismo_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'mujeres_sin_nombre_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'asimetria_mujer_hombre_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'infantilizacion_mujeres_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'denominacion_sexualizada_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'denominacion_redundante_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'denominacion_dependiente_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'criterios_excepcion_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'comparacion_mujeres_hombres_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'referencias_politicas_igualdad_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'denuncia_desigualdad_genero_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'presencia_mujeres_racializadas_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'presencia_mujeres_discapacidad_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'presencia_diversidad_generacional_id': ('LENGUAJE', 'LENGUAJE_VARS'),
        'tiene_fuente_id': ('LENGUAJE', 'LENGUAJE_VARS'),
    }
    
    mapeo = mapeo_variables.get(variable)
    if not mapeo:
        return str(codigo).strip() if codigo else ''  # Si no está en el mapeo, devolver tal cual
    
    seccion, nombre_config = mapeo
    seccion_data = contenido_general if seccion == 'CONTENIDO_GENERAL' else lenguaje
    codigos_validos = seccion_data.get(nombre_config, {})
    
    # Convertir código a string para comparar
    codigo_str = str(codigo).strip()
    
    # Detectar números muy largos o valores claramente incorrectos (más de 3 dígitos para códigos)
    if len(codigo_str) > 3 and codigo_str.isdigit():
        print(f"⚠️  Valor numérico muy largo detectado '{codigo_str[:30]}...' para {variable}, usando valor por defecto")
        if nombre_config == 'LENGUAJE_VARS':
            return '1'
        elif nombre_config == 'CITA_TITULAR':
            return '0'
        elif nombre_config == 'PERSONAS_MENCIONADAS':
            return '1'
        elif 'GENERO' in nombre_config:
            return '4' if 'GENERO_PERIODISTA' not in nombre_config else '1'
        elif nombre_config in ['MENCIONA_IA', 'IA_TEMA_CENTRAL', 'SIGNIFICADO_IA']:
            return '1'
        elif nombre_config == 'TEMA':
            return '17'  # Otros
        else:
            return '1'
    
    # Para variables numéricas (número de fotografías, declaraciones), solo validar que sea un número
    if variable in ['numero_fotografias', 'numero_declaraciones']:
        try:
            num_val = int(codigo_str)
            # Limitar a un rango razonable (0-100)
            if num_val < 0 or num_val > 100:
                return '0'
            return codigo_str
        except:
            return '0'
    
    # Verificar si es un código numérico válido
    if codigos_validos and codigo_str in codigos_validos:
        return codigo_str
    
    # Si el código parece ser un número pero no está en los válidos, verificar si es un número
    try:
        num_codigo = int(codigo_str)
        # Es un número pero no está en los códigos válidos
        # Para variables binarias LENGUAJE_VARS (1=No, 2=Sí), si es > 2, usar 1 por defecto
        if nombre_config == 'LENGUAJE_VARS' and num_codigo > 2:
            print(f"⚠️  Código numérico inválido '{codigo_str}' para {variable}, usando '1' (No)")
            return '1'
        # Para CITA_TITULAR (0=No, 1=Sí), si es > 1, usar 0 por defecto
        elif nombre_config == 'CITA_TITULAR' and num_codigo > 1:
            print(f"⚠️  Código numérico inválido '{codigo_str}' para {variable}, usando '0' (No)")
            return '0'
        # Para PERSONAS_MENCIONADAS (1=No, 2=Sí), si es > 2, usar 1 por defecto
        elif nombre_config == 'PERSONAS_MENCIONADAS' and num_codigo > 2:
            print(f"⚠️  Código numérico inválido '{codigo_str}' para {variable}, usando '1' (No)")
            return '1'
    except ValueError:
        # No es un número, intentar mapear desde texto descriptivo
        pass
    
    # Si no es un código numérico válido, intentar mapear desde texto descriptivo
    codigo_mapeado = mapear_texto_a_codigo(codigo_str, codigos_validos)
    if codigo_mapeado:
        return codigo_mapeado
    
    # Detectar casos donde gemma devolvió texto completamente incorrecto
    # (nombres propios, frases completas, etc.)
    if len(codigo_str) > 10 or '?' in codigo_str or codigo_str[0].isupper():
        print(f"⚠️  Valor incorrecto detectado '{codigo_str[:50]}...' para {variable}, usando valor por defecto")
        if nombre_config == 'LENGUAJE_VARS':
            return '1'
        elif 'GENERO' in nombre_config:
            return '4'
        elif nombre_config in ['MENCIONA_IA', 'IA_TEMA_CENTRAL', 'SIGNIFICADO_IA']:
            return '1'
        elif nombre_config == 'CITA_TITULAR':
            return '0'
        elif nombre_config == 'PERSONAS_MENCIONADAS':
            return '1'
        else:
            return '1'
    
    # Si es LENGUAJE_VARS y no se pudo mapear, intentar mapeo binario simple
    if nombre_config == 'LENGUAJE_VARS':
        if codigo_str.lower() in ['sí', 'si', 'yes', 'true']:
            return '2'
        elif codigo_str.lower() in ['no', 'false']:
            return '1'
        else:
            print(f"⚠️  Código inválido '{codigo_str}' para {variable} (debe ser 1 o 2), usando '1' (No)")
            return '1'
    
    # Si no hay códigos definidos, intentar mapeo binario
    if not codigos_validos:
        if codigo_str.lower() in ['sí', 'si', 'yes', 'true']:
            return '2'
        elif codigo_str.lower() in ['no', 'false']:
            return '1'
        else:
            return '1'
    
    # Si no es válido y no se pudo mapear, devolver el código por defecto según el tipo
    print(f"⚠️  Código inválido '{codigo_str}' para {variable}, usando valor por defecto")
    if 'GENERO' in nombre_config:
        return '4'  # Ns/Nc para variables de género
    elif nombre_config in ['MENCIONA_IA', 'IA_TEMA_CENTRAL', 'SIGNIFICADO_IA']:
        return '1'  # No para variables de IA
    elif nombre_config == 'CITA_TITULAR':
        return '0'  # No para CITA_TITULAR (usa '0' y '1')
    elif nombre_config == 'PERSONAS_MENCIONADAS':
        return '1'  # No para PERSONAS_MENCIONADAS
    else:
        return '1'  # Por defecto '1' (No)

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

def normalizar_valor_comparacion(valor, variable=None, config_data=None):
    """Normaliza un valor para comparación, mapeando texto a códigos numéricos si es necesario"""
    if pd.isna(valor) or valor == '':
        return ''
    
    valor_str = str(valor).strip()
    
    # Si ya es un número, devolverlo como string (sin lower para preservar '0')
    try:
        int(valor_str)
        return valor_str  # Mantener como está, no convertir a lower
    except ValueError:
        pass
    
    # Si es texto, intentar mapearlo a código numérico usando el config
    if config_data and variable:
        contenido_general = config_data.get("CONTENIDO_GENERAL", {})
        lenguaje = config_data.get("LENGUAJE", {})
        
        # Determinar qué diccionario usar según la variable
        mapeo_variables = {
            'Cita titular': ('CONTENIDO_GENERAL', 'CITA_TITULAR'),
            'Género periodista': ('CONTENIDO_GENERAL', 'GENERO_PERIODISTA'),
            'Género personas noticias': ('CONTENIDO_GENERAL', 'GENERO_PERSONAS_MENCIONADAS'),
            'Temática noticias': ('CONTENIDO_GENERAL', 'TEMA'),
            'IA principal': ('CONTENIDO_GENERAL', 'IA_TEMA_CENTRAL'),
            'Significado IA': ('CONTENIDO_GENERAL', 'SIGNIFICADO_IA'),
            'Menciona IA': ('CONTENIDO_GENERAL', 'MENCIONA_IA'),
            'Lenguaje sexista': ('LENGUAJE', 'LENGUAJE_SEXISTA'),
            # Nota: Las variables de género del titular no están en el Excel original, pero por si acaso:
            'Género nombre propio titular': ('CONTENIDO_GENERAL', 'GENERO_NOMBRE_PROPIO_TITULAR'),
        }
        
        mapeo = mapeo_variables.get(variable)
        if mapeo:
            seccion, nombre_config = mapeo
            seccion_data = contenido_general if seccion == 'CONTENIDO_GENERAL' else lenguaje
            codigos_dict = seccion_data.get(nombre_config, {})
            
            # Intentar mapear el texto al código
            codigo_mapeado = mapear_texto_a_codigo(valor_str, codigos_dict)
            if codigo_mapeado:
                return codigo_mapeado
        
        # Para variables LENGUAJE_VARS (binarias: 1=No, 2=Sí)
        variables_lenguaje_vars = [
            'Referencias políticas igualdad', 'Denuncia desigualdad género',
            'Presencia mujeres racializadas', 'Presencia  mujeres discapacidad', 'Presencia diversidad generacional',
            'Masculino genérico', 'Hombre humanidad', 'Uso dual zorra',
            'No uso cargos mujeres', 'Sexismo social', 'Androcentrismo_', 'Mujeres sin nombre',
            'Asimetría mujer hombre', 'Infantilización mujeres', 'Denominación sexualizada',
            'Denominación redundante', 'Denominación dependiente', 'Criterios excepción',
            'Comparación mujeres/hombres', 'Tiene fuente'
        ]
        
        if variable in variables_lenguaje_vars:
            valor_lower = valor_str.lower()
            # Mapear texto a códigos: Sí -> '2', No -> '1'
            if valor_lower in ['sí', 'si', 'yes', 'true', '2']:
                return '2'
            elif valor_lower in ['no', 'false', '1', '0']:
                return '1'
            # Si el valor es numérico pero no está en los códigos válidos, intentar mapear
            try:
                num_val = int(valor_str)
                if num_val == 2:
                    return '2'
                elif num_val == 1 or num_val == 0:
                    return '1'
            except:
                pass
    
    # Si no se pudo mapear, devolver el valor original (no normalizado a lower)
    return valor_str

def procesar_noticia(row, idx, df_original=None):
    """Procesa una noticia: extrae datos de newspaper y analiza con gemma"""
    try:
        # Obtener URL de la columna "Pagina"
        url = row.get('Pagina', None)
        
        # Obtener valores reales del CSV original si está disponible
        valores_reales = {}
        if df_original is not None and idx in df_original.index:
            fila_original = df_original.loc[idx]
            # Mapeo de columnas predichas a columnas reales en el CSV
            mapeo_columnas = {
                'Cita titular': 'Cita titular',
                'Género periodista': 'Género periodista',
                'Género personas noticias': 'Género personas noticias',
                'Temática noticias': 'Temática noticias',
                'Nombre propio titular': 'Nombre propio titular',
                'IA principal': 'IA principal',
                'Significado IA': 'Significado IA',
                'Menciona IA': 'Menciona IA',
                'Referencias políticas igualdad': 'Referencias políticas igualdad',
                'Denuncia desigualdad género': 'Denuncia desigualdad género',
                'Presencia mujeres racializadas': 'Presencia mujeres racializadas',
                'Presencia  mujeres discapacidad': 'Presencia  mujeres discapacidad',
                'Presencia diversidad generacional': 'Presencia diversidad generacional',
                'Lenguaje sexista': 'Lenguaje sexista',
                'Masculino genérico': 'Masculino genérico',
                'Hombre humanidad': 'Hombre humanidad',
                'Uso dual zorra': 'Uso dual zorra',
                'No uso cargos mujeres': 'No uso cargos mujeres',
                'Sexismo social': 'Sexismo social',
                'Androcentrismo_': 'Androcentrismo_',
                'Mujeres sin nombre': 'Mujeres sin nombre',
                'Asimetría mujer hombre': 'Asimetría mujer hombre',
                'Infantilización mujeres': 'Infantilización mujeres',
                'Denominación sexualizada': 'Denominación sexualizada',
                'Denominación redundante': 'Denominación redundante',
                'Denominación dependiente': 'Denominación dependiente',
                'Criterios excepción': 'Criterios excepción',
                'Comparación mujeres/hombres': 'Comparación mujeres/hombres',
                'Tiene fuente': 'Tiene fuente',
                'Número declaraciones': 'Número declaraciones',
            }
            
            for col_predicha, col_real in mapeo_columnas.items():
                if col_real in fila_original:
                    valores_reales[col_predicha] = fila_original[col_real]
        
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
            
            # Obtener resultados de los 3 bloques independientes
            res_contenido = out.get('resultado_contenido')
            res_lenguaje = out.get('resultado_lenguaje')
            res_fuentes = out.get('resultado_fuentes')
            
            # Convertir a diccionarios
            res_dict_contenido = res_contenido.model_dump() if res_contenido else {}
            res_dict_lenguaje = res_lenguaje.model_dump() if res_lenguaje else {}
            res_dict_fuentes = res_fuentes.model_dump() if res_fuentes else {}
            
            # Combinar todos los resultados en un solo diccionario
            res_dict = {**res_dict_contenido, **res_dict_lenguaje, **res_dict_fuentes}
            
            # Validar y corregir códigos según config.ini
            if res_dict:
                # Variables de CONTENIDO_GENERAL
                res_dict['tema_id'] = validar_codigo('tema_id', res_dict.get('tema_id', ''), CONFIG)
                res_dict['genero_periodista_id'] = validar_codigo('genero_periodista_id', res_dict.get('genero_periodista_id', ''), CONFIG)
                res_dict['cita_titular_id'] = validar_codigo('cita_titular_id', res_dict.get('cita_titular_id', ''), CONFIG)
                res_dict['genero_personas_mencionadas_id'] = validar_codigo('genero_personas_mencionadas_id', res_dict.get('genero_personas_mencionadas_id', ''), CONFIG)
                res_dict['genero_nombre_propio_titular_id'] = validar_codigo('genero_nombre_propio_titular_id', res_dict.get('genero_nombre_propio_titular_id', ''), CONFIG)
                res_dict['personas_mencionadas_id'] = validar_codigo('personas_mencionadas_id', res_dict.get('personas_mencionadas_id', ''), CONFIG)
                
                # Validar coherencia entre variables de género y sus correspondientes
                validar_coherencia_genero(res_dict, datos_newspaper)
                # Variables de IA
                res_dict['menciona_ia_id'] = validar_codigo('menciona_ia_id', res_dict.get('menciona_ia_id', ''), CONFIG)
                res_dict['ia_tema_central_id'] = validar_codigo('ia_tema_central_id', res_dict.get('ia_tema_central_id', ''), CONFIG)
                res_dict['significado_ia_id'] = validar_codigo('significado_ia_id', res_dict.get('significado_ia_id', ''), CONFIG)
                # Variables de Igualdad y Diversidad
                res_dict['referencias_politicas_igualdad_id'] = validar_codigo('referencias_politicas_igualdad_id', res_dict.get('referencias_politicas_igualdad_id', ''), CONFIG)
                res_dict['denuncia_desigualdad_genero_id'] = validar_codigo('denuncia_desigualdad_genero_id', res_dict.get('denuncia_desigualdad_genero_id', ''), CONFIG)
                res_dict['presencia_mujeres_racializadas_id'] = validar_codigo('presencia_mujeres_racializadas_id', res_dict.get('presencia_mujeres_racializadas_id', ''), CONFIG)
                res_dict['presencia_mujeres_discapacidad_id'] = validar_codigo('presencia_mujeres_discapacidad_id', res_dict.get('presencia_mujeres_discapacidad_id', ''), CONFIG)
                res_dict['presencia_diversidad_generacional_id'] = validar_codigo('presencia_diversidad_generacional_id', res_dict.get('presencia_diversidad_generacional_id', ''), CONFIG)
                # Variables de LENGUAJE
                res_dict['lenguaje_sexista_id'] = validar_codigo('lenguaje_sexista_id', res_dict.get('lenguaje_sexista_id', ''), CONFIG)
                res_dict['masculino_generico_id'] = validar_codigo('masculino_generico_id', res_dict.get('masculino_generico_id', ''), CONFIG)
                res_dict['hombre_humanidad_id'] = validar_codigo('hombre_humanidad_id', res_dict.get('hombre_humanidad_id', ''), CONFIG)
                res_dict['uso_dual_zorra_id'] = validar_codigo('uso_dual_zorra_id', res_dict.get('uso_dual_zorra_id', ''), CONFIG)
                res_dict['no_uso_cargos_mujeres_id'] = validar_codigo('no_uso_cargos_mujeres_id', res_dict.get('no_uso_cargos_mujeres_id', ''), CONFIG)
                res_dict['sexismo_social_id'] = validar_codigo('sexismo_social_id', res_dict.get('sexismo_social_id', ''), CONFIG)
                res_dict['androcentrismo_id'] = validar_codigo('androcentrismo_id', res_dict.get('androcentrismo_id', ''), CONFIG)
                res_dict['mujeres_sin_nombre_id'] = validar_codigo('mujeres_sin_nombre_id', res_dict.get('mujeres_sin_nombre_id', ''), CONFIG)
                res_dict['asimetria_mujer_hombre_id'] = validar_codigo('asimetria_mujer_hombre_id', res_dict.get('asimetria_mujer_hombre_id', ''), CONFIG)
                res_dict['infantilizacion_mujeres_id'] = validar_codigo('infantilizacion_mujeres_id', res_dict.get('infantilizacion_mujeres_id', ''), CONFIG)
                res_dict['denominacion_sexualizada_id'] = validar_codigo('denominacion_sexualizada_id', res_dict.get('denominacion_sexualizada_id', ''), CONFIG)
                res_dict['denominacion_redundante_id'] = validar_codigo('denominacion_redundante_id', res_dict.get('denominacion_redundante_id', ''), CONFIG)
                res_dict['denominacion_dependiente_id'] = validar_codigo('denominacion_dependiente_id', res_dict.get('denominacion_dependiente_id', ''), CONFIG)
                res_dict['criterios_excepcion_id'] = validar_codigo('criterios_excepcion_id', res_dict.get('criterios_excepcion_id', ''), CONFIG)
                res_dict['comparacion_mujeres_hombres_id'] = validar_codigo('comparacion_mujeres_hombres_id', res_dict.get('comparacion_mujeres_hombres_id', ''), CONFIG)
                # Variables de Fuentes (fotografías se extraen solo de newspaper, no se predicen)
                res_dict['tiene_fuente_id'] = validar_codigo('tiene_fuente_id', res_dict.get('tiene_fuente_id', ''), CONFIG)
                res_dict['numero_declaraciones'] = validar_codigo('numero_declaraciones', res_dict.get('numero_declaraciones', ''), CONFIG)
        except Exception as e:
            print(f"⚠️  Error en análisis gemma para fila {idx}: {e}")
            res_dict = {}
        
        end_t = time.time()
        
        # Construir resultado final con valores predichos
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
            
            # Datos de gemma - CONTENIDO_GENERAL
            'Nombre propio titular': res_dict.get('nombre_propio_titular', ''),
            'Cita titular': res_dict.get('cita_titular_id', ''),
            'Género periodista': res_dict.get('genero_periodista_id', ''),
            'Género personas noticias': res_dict.get('genero_personas_mencionadas_id', ''),
            'Temática noticias': res_dict.get('tema_id', ''),
            
            # Variables de IA
            'IA principal': res_dict.get('ia_tema_central_id', ''),
            'Significado IA': res_dict.get('significado_ia_id', ''),
            'Menciona IA': res_dict.get('menciona_ia_id', ''),
            
            # Variables de Igualdad y Diversidad
            'Referencias políticas igualdad': res_dict.get('referencias_politicas_igualdad_id', ''),
            'Denuncia desigualdad género': res_dict.get('denuncia_desigualdad_genero_id', ''),
            'Presencia mujeres racializadas': res_dict.get('presencia_mujeres_racializadas_id', ''),
            'Presencia  mujeres discapacidad': res_dict.get('presencia_mujeres_discapacidad_id', ''),
            'Presencia diversidad generacional': res_dict.get('presencia_diversidad_generacional_id', ''),
            
            # Variables de LENGUAJE
            'Lenguaje sexista': res_dict.get('lenguaje_sexista_id', ''),
            'Masculino genérico': res_dict.get('masculino_generico_id', ''),
            'Hombre humanidad': res_dict.get('hombre_humanidad_id', ''),
            'Uso dual zorra': res_dict.get('uso_dual_zorra_id', ''),
            'No uso cargos mujeres': res_dict.get('no_uso_cargos_mujeres_id', ''),
            'Sexismo social': res_dict.get('sexismo_social_id', ''),
            'Androcentrismo_': res_dict.get('androcentrismo_id', ''),
            'Mujeres sin nombre': res_dict.get('mujeres_sin_nombre_id', ''),
            'Asimetría mujer hombre': res_dict.get('asimetria_mujer_hombre_id', ''),
            'Infantilización mujeres': res_dict.get('infantilizacion_mujeres_id', ''),
            'Denominación sexualizada': res_dict.get('denominacion_sexualizada_id', ''),
            'Denominación redundante': res_dict.get('denominacion_redundante_id', ''),
            'Denominación dependiente': res_dict.get('denominacion_dependiente_id', ''),
            'Criterios excepción': res_dict.get('criterios_excepcion_id', ''),
            'Comparación mujeres/hombres': res_dict.get('comparacion_mujeres_hombres_id', ''),
            
            # Variables de Fuentes
            'Tiene fuente': res_dict.get('tiene_fuente_id', ''),
            'Número declaraciones': res_dict.get('numero_declaraciones', ''),
            
            # Metadatos
            'TIEMPO_PROCESAMIENTO': round(end_t - start_t, 2),
            'TIMESTAMP': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'MODELO_LLM': LLM_MODEL_NAME,
            'ID_ORIGINAL': idx  # Guardar índice original para matching
        }
        
        # Agregar comparaciones con valores reales (excluyendo fotografías que solo se extraen de newspaper)
        variables_comparar = [
            'Cita titular', 'Género periodista', 'Género personas noticias', 'Temática noticias',
            'IA principal', 'Significado IA', 'Menciona IA',
            'Referencias políticas igualdad', 'Denuncia desigualdad género',
            'Presencia mujeres racializadas', 'Presencia  mujeres discapacidad', 'Presencia diversidad generacional',
            'Lenguaje sexista', 'Masculino genérico', 'Hombre humanidad', 'Uso dual zorra',
            'No uso cargos mujeres', 'Sexismo social', 'Androcentrismo_', 'Mujeres sin nombre',
            'Asimetría mujer hombre', 'Infantilización mujeres', 'Denominación sexualizada',
            'Denominación redundante', 'Denominación dependiente', 'Criterios excepción',
            'Comparación mujeres/hombres', 'Tiene fuente', 'Número declaraciones'
        ]
        
        for var in variables_comparar:
            valor_predicho = resultado.get(var, '')
            valor_real = valores_reales.get(var, '')
            
            # Normalizar valores para comparación (mapeando texto a códigos)
            pred_norm = normalizar_valor_comparacion(valor_predicho, variable=var, config_data=CONFIG)
            real_norm = normalizar_valor_comparacion(valor_real, variable=var, config_data=CONFIG)
            
            # Determinar si coinciden
            coincide = '✅' if pred_norm == real_norm and pred_norm != '' else '❌'
            
            # Agregar columnas de comparación
            resultado[f'{var}_REAL'] = valor_real
            resultado[f'{var}_PREDICHO'] = valor_predicho
            resultado[f'{var}_COINCIDE'] = coincide
        
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
        resultado = procesar_noticia(row, idx, df_original=df)
        
        if resultado:
            resultados.append(resultado)
        else:
            errores.append(idx)
        
        # Pequeña pausa para no saturar el sistema
        time.sleep(0.3)
    
    # Crear DataFrame con resultados
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        
        # Calcular métricas de precisión
        print("\n" + "="*60)
        print("📊 MÉTRICAS DE PRECISIÓN")
        print("="*60)
        
        variables_comparar = [
            'Cita titular', 'Género periodista', 'Género personas noticias', 'Temática noticias',
            'IA principal', 'Significado IA', 'Menciona IA',
            'Referencias políticas igualdad', 'Denuncia desigualdad género',
            'Presencia mujeres racializadas', 'Presencia  mujeres discapacidad', 'Presencia diversidad generacional',
            'Lenguaje sexista', 'Masculino genérico', 'Hombre humanidad', 'Uso dual zorra',
            'No uso cargos mujeres', 'Sexismo social', 'Androcentrismo_', 'Mujeres sin nombre',
            'Asimetría mujer hombre', 'Infantilización mujeres', 'Denominación sexualizada',
            'Denominación redundante', 'Denominación dependiente', 'Criterios excepción',
            'Comparación mujeres/hombres', 'Tiene fuente', 'Número declaraciones'
        ]
        
        metricas = []
        for var in variables_comparar:
            col_coincide = f'{var}_COINCIDE'
            if col_coincide in df_resultados.columns:
                total = len(df_resultados[df_resultados[col_coincide].notna()])
                coincidencias = len(df_resultados[df_resultados[col_coincide] == '✅'])
                precision = (coincidencias / total * 100) if total > 0 else 0
                metricas.append([var, total, coincidencias, f"{precision:.1f}%"])
        
        if metricas:
            print("\nVariable | Total | Coincidencias | Precisión")
            print("-" * 60)
            for m in metricas:
                print(f"{m[0][:30]:30} | {m[1]:5} | {m[2]:13} | {m[3]}")
            
            # Crear DataFrame con métricas y guardar en Excel separado
            df_metricas = pd.DataFrame(metricas, columns=['Variable', 'Total', 'Coincidencias', 'Precisión (%)'])
            df_metricas.to_excel(OUTPUT_METRICAS_EXCEL, index=False)
            print(f"\n✅ Excel de métricas guardado: {os.path.abspath(OUTPUT_METRICAS_EXCEL)}")
        
        # Guardar Excel principal (sin columnas de comparación para mantenerlo limpio)
        columnas_principales = [col for col in df_resultados.columns if not col.endswith('_REAL') and not col.endswith('_PREDICHO') and not col.endswith('_COINCIDE')]
        df_resultados_limpio = df_resultados[columnas_principales].copy()
        df_resultados_limpio.to_excel(OUTPUT_EXCEL, index=False)
        print(f"✅ Excel principal guardado: {os.path.abspath(OUTPUT_EXCEL)}")
        
        # Guardar Excel de comparaciones detalladas (solo con columnas de comparación)
        columnas_comparacion = ['ID_ORIGINAL', 'Pagina', 'Titular'] + [col for col in df_resultados.columns if col.endswith('_REAL') or col.endswith('_PREDICHO') or col.endswith('_COINCIDE')]
        df_comparaciones = df_resultados[columnas_comparacion].copy()
        df_comparaciones.to_excel(OUTPUT_COMPARACIONES_EXCEL, index=False)
        print(f"✅ Excel de comparaciones detalladas guardado: {os.path.abspath(OUTPUT_COMPARACIONES_EXCEL)}")
        
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
