import configparser
import ast

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