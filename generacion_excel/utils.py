import configparser
import ast
import pandas as pd
from newspaper import Article
from urllib.parse import urlparse
from datetime import datetime
import re

def cargar_config_ini(ruta='/home/jggomez/Desktop/IRIS/iris-uc3m/pruebas_embeddings/rag/config.ini'):
    """Lee el INI y convierte los strings de diccionarios en dicts reales de Python"""
    config = configparser.ConfigParser()
    config.read(ruta, encoding='utf-8')
    data = {}
    
    for section in config.sections():
        data[section] = {}
        for key, value in config[section].items():
            if key == 'variables':
                # Convertir lista string a lista real
                try:
                    data[section]['variables'] = ast.literal_eval(value)
                except:
                    data[section]['variables'] = []
            else:
                # Convertir dicts string a dict real (ej: "{'1': 'Sí'}")
                try:
                    # Normalizamos claves a mayúsculas para búsqueda fácil
                    data[section][key.upper()] = ast.literal_eval(value)
                except:
                    pass
    return data

def generar_instrucciones_codebook(config_data, seccion):
    """Genera el texto que irá en el Prompt explicando los códigos"""
    texto = f"### LIBRO DE CÓDIGOS PARA: {seccion}\n"
    texto += "Para las siguientes variables, selecciona SOLO el código numérico (ID) que corresponda a la definición:\n\n"
    
    seccion_data = config_data.get(seccion, {})
    
    for variable, opciones in seccion_data.items():
        if variable == 'variables': continue
        if isinstance(opciones, dict):
            texto += f"- **{variable}**:\n"
            for codigo, descripcion in opciones.items():
                texto += f"  - ID '{codigo}': {descripcion}\n"
            texto += "\n"
    return texto

def limpiar_autor(autor_str):
    """Limpia el autor quitando texto innecesario como ', Ver Biografía'"""
    if not autor_str:
        return ''
    
    # Quitar patrones comunes de texto innecesario
    patrones_a_quitar = [
        r',\s*Ver\s+Biografía',
        r',\s*ver\s+biografía',
        r',\s*Ver\s+biografía',
        r',\s*VER\s+BIOGRAFÍA',
        r',\s*Ver\s+perfil',
        r',\s*ver\s+perfil',
    ]
    
    autor_limpio = autor_str
    for patron in patrones_a_quitar:
        autor_limpio = re.sub(patron, '', autor_limpio, flags=re.IGNORECASE)
    
    # Limpiar espacios extra y comas al final
    autor_limpio = autor_limpio.strip().rstrip(',')
    
    return autor_limpio

def formatear_mes(fecha):
    """Convierte una fecha al formato 'Enero' en español (solo el mes)"""
    if not fecha:
        return ''
    
    meses_es = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    
    try:
        mes_nombre = meses_es.get(fecha.month, '')
        return mes_nombre
    except:
        return ''

def extraer_datos_newspaper(url):
    """Extrae datos del artículo usando newspaper"""
    try:
        if pd.isna(url) or url == '':
            return {}
        
        article = Article(
            url,
            language="es",
            fetch_images=False,
            keep_article_html=False,
            memoize_articles=False,
        )
        article.download()
        article.parse()
        
        # Limpiar autor
        autor_raw = ', '.join(article.authors) if article.authors else ''
        autor_limpio = limpiar_autor(autor_raw)
        
        # Extraer datos
        datos = {
            'Autor': autor_limpio,
            'Titular': article.title if article.title else '',
            'Contenido': article.text if article.text else '',
            'textonoticia': article.text if article.text else '',
            'Caracteres': len(article.text) if article.text else 0,
            'Pagina': url,
        }
        
        # Extraer MES de la fecha de publicación con formato en español
        if article.publish_date:
            datos['MES'] = formatear_mes(article.publish_date)
        else:
            datos['MES'] = ''
        
        # MMCC - extraer del dominio
        domain = urlparse(url).netloc.lower()
        datos['MMCC'] = domain
        
        # nombre_periodista es el mismo que Autor (ya limpio)
        datos['nombre_periodista'] = autor_limpio
        
        return datos
    except Exception as e:
        print(f"Error al procesar {url}: {e}")
        return {}