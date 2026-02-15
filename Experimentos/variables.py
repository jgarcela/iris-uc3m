

from urllib.parse import urlparse
from newspaper import Article
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Any, List
import re
import json
from utils import consultar_ollama


# =====================================================================================
# 1. IdNoticia
# =====================================================================================
def obtener_id_noticia(articulo_raw):
    """
    Extrae y devuelve el IdNoticia de un registro dado.
    """
    articulo_id = articulo_raw['IdNoticia']
    return articulo_id


# =====================================================================================
# 2. Medio
# =====================================================================================
def clasificar_var_medio(articulo):
    """
    Clasifica el medio según la URL
    """
    # Extraer el dominio de la URL
    url = articulo.url
    domain = urlparse(url).netloc.lower()
    
    # Diccionario de dominios y categorías
    media_map = {
        "elmundo.es": 1,
        "elpais.com": 2,
        "eldiario.es": 3,
        "20minutos.es": 4,
        "articulo14.com": 5,
        "infolibre.es": 6,
        "lavanguardia.com": 7
    }
    
    # Buscar el dominio en el mapa
    for key, value in media_map.items():
        if key in domain:
            return key, value
    
    # Si no coincide con ninguno, retornar None o 0
    return 0


# =====================================================================================
# 3. Fecha
# =====================================================================================
def clasificar_var_fecha(articulo: Article) -> Optional[str]:
    """
    Recibe un objeto Article.
    Devuelve la fecha como texto en formato 'dd/mm/aa' o None.
    """
    # Verificamos si existe la fecha
    if articulo.publish_date:
        # %d = día (01-31)
        # %m = mes (01-12)
        # %y = año (dos últimos dígitos, ej: 26)
        return articulo.publish_date.strftime("%d/%m/%y")
    
    return None

# =====================================================================================
# 3a. Mes
# =====================================================================================
def clasificar_var_mes(articulo: Article) -> Optional[int]:
    """
    Recibe un objeto Article procesado.
    Devuelve el mes (MM) como entero o None si no tiene fecha.
    """
    if articulo.publish_date:
        return articulo.publish_date.month
    
    return None


# =====================================================================================
# 4. Año
# =====================================================================================
def clasificar_var_año(articulo: Article) -> Optional[int]:
    """
    Recibe un objeto Article procesado.
    Devuelve el año (YYYY) como entero o None si no tiene fecha.
    """
    if articulo.publish_date:
        return articulo.publish_date.year
    
    return None


# =====================================================================================
# 5. Número de Caracteres
# =====================================================================================
def clasificar_var_caracteres(articulo: Article) -> int:
    """
    Devuelve la cantidad de caracteres del cuerpo del artículo.
    Si no hay texto, devuelve 0.
    """
    if articulo.text:
        # Retorna la longitud del texto
        return len(articulo.text)
    
    return 0

# =====================================================================================
# 6. Titular
# =====================================================================================
def clasificar_var_titular(articulo: Article) -> Optional[str]:
    """
    Recibe un objeto Article procesado.
    Devuelve el titular en minúsculas o None si no existe.
    """
    if articulo.title:
        # .strip() quita espacios extra y .lower() pasa a minúsculas
        return articulo.title.strip().lower()
    
    return None


# =====================================================================================
# 7a. Nombre Propio Titular
# =====================================================================================
from utils import  NombresDetectados

def clasificar_var_nombre_propio_titular_list(titulo: str) -> NombresDetectados:
    """
    Extrae nombres propios del titular y los clasifica según su género o tipo.
    Devuelve dos listas sincronizadas: nombres y valores.
    """
    
    # Si el título es muy corto, devolvemos listas vacías
    if not titulo or len(titulo) < 3:
        return NombresDetectados(nombres=[], valores=[])

    prompt = f"""
    Analiza el siguiente TITULAR y extrae los nombres propios, entidades, lugares o tecnologías.
    Asigna a cada uno su código numérico correspondiente.

    TITULAR: "{titulo}"

    TABLA DE CÓDIGOS:
    1  = Hombre (Nombre individual)
    2  = Mujer (Nombre individual)
    3  = Grupo Mixto (ej: "Los Reyes", "La pareja", "Padres")
    32 = Grupo Mixto (Mayoría hombres)
    33 = Grupo Mixto (Mayoría mujeres)
    4  = Institución, Organización, Empresa, Partido Político (ej: "Google", "PSOE", "La ONU")
    41 = Lugares: Países, Regiones, Ciudades (ej: "España", "Madrid", "Europa")
    42 = Tecnología: Apps, Modelos de IA, Robots, Software (ej: "ChatGPT", "Gemini", "Sora", "TikTok")

    INSTRUCCIONES:
    - Ignora sustantivos comunes que no sean entidades (ej: no extraigas "la policía" si es genérico, pero sí "Mossos d'Esquadra").
    - Distingue bien entre la EMPRESA (OpenAI -> 4) y el PRODUCTO (ChatGPT -> 42).
    - Si no hay nombres propios, devuelve listas vacías.

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON con esta estructura exacta:
    {{
        "nombres": ["Nombre1", "Nombre2"],
        "valores": [Codigo1, Codigo2]
    }}
    """

    # --- LLAMADA AL MODELO ---
    respuesta_texto = consultar_ollama(prompt)

    # --- PARSEO Y VALIDACIÓN ---
    try:
        # Limpieza básica para encontrar el JSON
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1:
            return NombresDetectados(nombres=[], valores=[])
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)
        
        # Validamos con Pydantic
        resultado = NombresDetectados(**data)
        
        # Validación de seguridad: Las listas deben tener el mismo tamaño
        if len(resultado.nombres) != len(resultado.valores):
            # Si hay desajuste, cortamos a la longitud del más corto
            min_len = min(len(resultado.nombres), len(resultado.valores))
            resultado.nombres = resultado.nombres[:min_len]
            resultado.valores = resultado.valores[:min_len]
            
        return resultado

    except (json.JSONDecodeError, ValidationError):
        return NombresDetectados(nombres=[], valores=[])



# =====================================================================================
# 7b. Género Nombre Propio Titular
# =====================================================================================
def clasificar_var_nombre_propio_titular(valores: list[int]) -> int:
    """
    Calcula el valor único de protagonismo basado en la lista de entidades detectadas.
    Prioridad absoluta a las personas sobre entidades/lugares.
    """
    
    if not valores:
        return 0

    # 1. CONTADORES DE PERSONAS
    cnt_hombres = valores.count(1)
    cnt_mujeres = valores.count(2)
    
    # Contamos también si el LLM ya detectó grupos mixtos explícitos
    cnt_mixtos_neutro = valores.count(3)
    cnt_mixtos_hombres = valores.count(32)
    cnt_mixtos_mujeres = valores.count(33)

    # Suma total de indicadores humanos
    total_humanos = cnt_hombres + cnt_mujeres + cnt_mixtos_neutro + cnt_mixtos_hombres + cnt_mixtos_mujeres

    # --- FASE 1: FILTRO HUMANO (Prioridad Absoluta) ---
    if total_humanos > 0:
        
        # CASO A: Solo Hombres (Sin mujeres ni grupos mixtos)
        if cnt_hombres > 0 and cnt_mujeres == 0 and cnt_mixtos_neutro == 0 and cnt_mixtos_hombres == 0 and cnt_mixtos_mujeres == 0:
            return 1
            
        # CASO B: Solo Mujeres (Sin hombres ni grupos mixtos)
        if cnt_mujeres > 0 and cnt_hombres == 0 and cnt_mixtos_neutro == 0 and cnt_mixtos_hombres == 0 and cnt_mixtos_mujeres == 0:
            return 2

        # CASO C: Mixto (Hay presencia de ambos o indicadores de grupo)
        # Aquí decidimos si es 3, 32 o 33 contando individuos
        
        if cnt_hombres > cnt_mujeres:
            return 32  # Mixto mayoritariamente masculino
        elif cnt_mujeres > cnt_hombres:
            return 33  # Mixto mayoritariamente femenino
        else:
            # Empate técnico (ej: 1 hombre y 1 mujer) o solo había un [3] genérico
            # Si el empate viene de counts explícitos (1 vs 1), es 3.
            # Si viene de grupos (ej: un [32] detectado), respetamos ese matiz.
            if cnt_mixtos_hombres > cnt_mixtos_mujeres:
                return 32
            elif cnt_mixtos_mujeres > cnt_mixtos_hombres:
                return 33
            else:
                return 3 # Empate total o mixto neutro

    # --- FASE 2: NO HUMANOS (Solo si total_humanos == 0) ---
    # Establecemos jerarquía de interés: IA > Entidad > Lugar
    
    if 42 in valores:
        return 42 # Prioridad a Tecnología/IA
    
    if 4 in valores:
        return 4  # Empresas / Instituciones
        
    if 41 in valores:
        return 41 # Lugares (es lo menos informativo)

    # Si todo falla (ej: llegó un código desconocido)
    return 0


# =====================================================================================
# 8. Cita Titular
# =====================================================================================
from utils import CitaTitularValidada

def clasificar_var_cita_titular(titulo: str) -> CitaTitularValidada:
    """
    Analiza si el titular contiene una cita o declaración de alguien.
    Distingue entre directa (textual) e indirecta (parafraseada).
    """
    
    if not titulo:
        return CitaTitularValidada(cita="N/A", tipo=1)

    prompt = f"""
    Analiza el siguiente TITULAR de noticia y detecta si contiene una CITA o DECLARACIÓN de una persona o entidad.

    TITULAR: "{titulo}"

    CLASIFICACIÓN:
    1 = No hay cita. Es un hecho, una descripción del periodista o usa comillas solo para resaltar una palabra (ej: El "caso Koldo").
    2 = Cita Directa. Reproduce palabras textuales. 
        - Pistas: Usa comillas para una frase completa ("Me voy", dijo X) o dos puntos (Sánchez: No dimitiré).
    3 = Cita Indirecta. Parafrasea lo que alguien dijo sin usar comillas para la frase entera.
        - Pistas: Usa verbos de habla (asegura que, dice que, pide, advierte, niega) seguidos de la idea.

    INSTRUCCIONES:
    - En el campo "cita", extrae SOLO el contenido de lo que se ha dicho. Si es tipo 1, pon "N/A".
    - Sé estricto: "Madrid aprueba la ley" es 1 (hecho). "Ayuso dice que Madrid aprobará la ley" es 3 (indirecta).

    Responde SOLO con este JSON:
    {{
        "cita": "texto de la declaración",
        "tipo": numero
    }}
    """

    # Llamada al LLM (asumiendo que tienes tu función consultar_ollama)
    respuesta = consultar_ollama(prompt)
    
    try:
        # Limpieza básica para extraer JSON
        json_str = respuesta[respuesta.find('{'):respuesta.rfind('}')+1]
        data = json.loads(json_str)
        
        return CitaTitularValidada(**data)
        
    except Exception:
        # Fallback en caso de error
        return CitaTitularValidada(cita="Error al procesar", tipo=1)


# =====================================================================================
# 9a. Protagonistas que aparecen en la información
# =====================================================================================
from utils import ProtagonistasDetectados

def clasificar_var_cla_genero_prota_list(texto_noticia: str) -> ProtagonistasDetectados:
    """
    Analiza el cuerpo de la noticia para extraer los protagonistas principales.
    Devuelve listas sincronizadas de nombres únicos y sus códigos.
    """
    
    # Validación básica
    if not texto_noticia or len(texto_noticia) < 10:
        return ProtagonistasDetectados(nombres=[], valores=[])

    # --- PROMPT ---
    prompt = f"""
    Analiza el siguiente TEXTO DE NOTICIA y extrae los protagonistas (personas, entidades, lugares clave).
    
    TEXTO (Fragmento):
    "{texto_noticia}..."

    TABLA DE CÓDIGOS ESTRICTA:
    1  = Hombre (Ej: "Pedro Sánchez", "El Papa", "Carlos")
    2  = Mujer (Ej: "María", "Isabel Díaz Ayuso", "Alessandra")
    3  = Grupo Mixto (Ej: "La pareja", "Los vecinos", "Padres")
    4  = Institución/Empresa (Ej: "Gobierno", "Reuters", "Vatican Media", "PSOE")
    41 = Lugares: Países, Regiones, Ciudades (ej: "España", "Madrid", "Europa")
    42 = Tecnología: Apps, Modelos de IA, Robots, Software (ej: "ChatGPT", "Gemini", "Sora", "TikTok")

    REGLAS OBLIGATORIAS:
    1. Si el nombre es de una sola persona, JAMÁS uses códigos 3, 32 o 33.
    2. Agencias de noticias (Reuters, EFE, Europa Press) son SIEMPRE código 4.
    3. Nombres femeninos (María, Alessandra) son código 2.
    4. Nombres masculinos (Pedro, Francisco) son código 1.
    
    FORMATO JSON EXACTO:
    {{
        "nombres": ["Nombre1", "Nombre2"],
        "valores": [Codigo1, Codigo2]
    }}
    """

    # --- LLAMADA AL MODELO ---
    # Asumiendo que usas tu función 'consultar_ollama'
    # Se recomienda un modelo con buena capacidad de contexto (ej: llama3, mistral, gemma:7b)
    respuesta_texto = consultar_ollama(prompt)

    # --- PARSEO Y VALIDACIÓN ---
    try:
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1:
            return ProtagonistasDetectados(nombres=[], valores=[])
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)
        
        resultado = ProtagonistasDetectados(**data)
        
        # Sincronización de seguridad
        min_len = min(len(resultado.nombres), len(resultado.valores))
        resultado.nombres = resultado.nombres[:min_len]
        resultado.valores = resultado.valores[:min_len]
            
        return resultado

    except (json.JSONDecodeError, ValidationError):
        return ProtagonistasDetectados(nombres=[], valores=[])


# =====================================================================================
# 9b. Génerop Protagonistas que aparecen en la información
# =====================================================================================
def clasificar_var_cla_genero_prota(valores: list[int]) -> int:
    """
    Calcula el valor único de los protagonistas en el CUERPO de la noticia.
    Toma la lista de códigos detectados y decide el código dominante.
    """
    
    if not valores:
        return 0

    # 1. CONTEO DE CÓDIGOS HUMANOS
    cnt_hombres = valores.count(1)
    cnt_mujeres = valores.count(2)
    
    # Contamos grupos mixtos detectados por el LLM
    cnt_mixtos_neutro = valores.count(3)
    cnt_mixtos_hombres = valores.count(32)
    cnt_mixtos_mujeres = valores.count(33)

    # Suma total de indicadores humanos (Individuales + Grupos)
    total_humanos = cnt_hombres + cnt_mujeres + cnt_mixtos_neutro + cnt_mixtos_hombres + cnt_mixtos_mujeres

    # --- FASE 1: FILTRO HUMANO (Prioridad Absoluta) ---
    if total_humanos > 0:
        
        # CASO A: Solo Hombres (Sin mujeres individuales NI grupos mixtos)
        # Es estricto: si aparece un grupo "33" o "3", ya no es solo hombres.
        if (cnt_hombres > 0 and cnt_mujeres == 0 and 
            cnt_mixtos_neutro == 0 and cnt_mixtos_hombres == 0 and cnt_mixtos_mujeres == 0):
            return 1
            
        # CASO B: Solo Mujeres (Sin hombres individuales NI grupos mixtos)
        if (cnt_mujeres > 0 and cnt_hombres == 0 and 
            cnt_mixtos_neutro == 0 and cnt_mixtos_hombres == 0 and cnt_mixtos_mujeres == 0):
            return 2

        # CASO C: Mixto (Hay presencia de ambos o hay grupos mixtos)
        # Comparamos cantidades para determinar la mayoría
        
        # C1. Comparación directa de individuos
        if cnt_hombres > cnt_mujeres:
            return 32  # Mixto más hombres
        elif cnt_mujeres > cnt_hombres:
            return 33  # Mixto más mujeres
        
        # C2. Empate de individuos (ej: 0 vs 0, o 2 vs 2). Desempate por grupos.
        else:
            if cnt_mixtos_hombres > cnt_mixtos_mujeres:
                return 32
            elif cnt_mixtos_mujeres > cnt_mixtos_hombres:
                return 33
            else:
                # Empate total (ej: 1 hombre, 1 mujer) o solo grupos neutros (3)
                return 3

    # --- FASE 2: NO HUMANOS (Solo si no hay NINGÚN humano) ---
    
    # Prioridad: Tecnología (42) > Institución (4) > Lugar (41)
    
    if 42 in valores:
        return 42 # IA, Robots, Apps
    
    if 4 in valores:
        return 4  # Empresas, Partidos, Organismos
        
    if 41 in valores:
        return 41 # Países, Ciudades (Fondo)

    # Si llega aquí, es un código desconocido o lista vacía
    return 0


# =====================================================================================
# 10. Nombre Periodista
# =====================================================================================
def clasificar_var_nombre_periodista(articulo: Article) -> str:
    """
    Extrae autores y limpia textos basura como 'Ver Biografía', 'Redacción', etc.
    """
    
    # Lista de palabras/frases que NO queremos en el nombre
    palabras_basura = [
        "ver biografía", "biografía", "ver perfil", "perfil", 
        "ver más", "see profile", "read more", "twitter", 
        "email", "follow", "redacción", "agencia"
    ]

    autores_detectados = []

    # 1. Intentar obtener autores desde el parser
    if articulo.authors:
        for autor in articulo.authors:
            autor_limpio = autor.strip()
            
            # Verificamos si el texto (en minúsculas) contiene alguna palabra basura
            if any(basura in autor_limpio.lower() for basura in palabras_basura):
                # Si es basura pura (ej: "Ver Biografía"), lo ignoramos
                # Pero si es "Lorena Pacho, Ver Biografía", intentamos limpiarlo
                
                # Caso específico que te pasó: eliminar "Ver Biografía" del string
                for basura in palabras_basura:
                    # Reemplazamos la basura por vacío, ignorando mayúsculas/minúsculas es complejo,
                    # así que hacemos un replace simple de las variantes comunes:
                    autor_limpio = autor_limpio.replace("Ver Biografía", "")
                    autor_limpio = autor_limpio.replace("Ver biografía", "")
                
                autor_limpio = autor_limpio.strip(" ,|") # Limpiamos comas o barras sobrantes

            # Si después de limpiar queda algo y no es demasiado corto, lo guardamos
            if len(autor_limpio) > 2:
                autores_detectados.append(autor_limpio)

    # 2. Si encontramos autores limpios, los devolvemos
    if autores_detectados:
        # Eliminamos duplicados usando set() y mantenemos orden
        return ", ".join(list(dict.fromkeys(autores_detectados)))

    # 3. Fallback: Metadatos (si la lista authors falló o se limpió todo)
    meta = articulo.meta_data
    claves_meta = ['author', 'og:author', 'dc.creator', 'byl']
    
    for clave in claves_meta:
        valor = meta.get(clave)
        if valor:
            # A veces los metadatos también traen basura, podrías aplicar limpieza aquí también
            return str(valor).strip()

    # 4. Defecto
    return "Redacción / Otros"


# =====================================================================================
# 11. Género Periodista (Autoría)
# =====================================================================================
from utils import GeneroPeriodistaValidado
def clasificar_var_genero_periodista(nombre_periodista: str, nombre_medio:str) -> int:
    """
    Clasifica la autoría considerando el contexto del medio.
    Recibe:
      - nombre_periodista: El texto de la firma (ej: "Redacción", "Juan Pérez")
      - nombre_medio: El nombre del periódico donde se publica (ej: "El País")
    """
    
    # Validación inicial rápida
    if not nombre_periodista or len(nombre_periodista) < 2:
        return 0

    # Prompt enriquecido con el contexto del medio y definiciones
    prompt = f"""
    Contexto:
    Noticia publicada en el medio: "{nombre_medio}".
    Autor/Firma a analizar: "{nombre_periodista}".
    
    Tu misión es clasificar la autoría (0-7) siguiendo estrictamente estas definiciones:

    0 = Ns/Nc: Desconocido, ambiguo o iniciales.
    1 = Hombre: Nombre de persona masculino.
    2 = Mujer: Nombre de persona femenino.
    3 = Mixto: Varios autores de distinto género.
    4 = Otros medios: El autor es otro medio de comunicación (ej: "The New York Times", "Revista Hola").
    5 = Agencia: Agencias de noticias puras (EFE, Europa Press, Reuters, AFP).
    
    6 = Redacción (Periodística): 
        - Firma genérica del propio medio "{nombre_medio}" (ej: "Redacción", "El País", "Editorial").
        - IMPORTANTE: Si el autor es una empresa comercial que NO es un medio de noticias (como Ford, Apple), NO ES 6.
    
    7 = Corporativo (Comercial / Institucional): 
        - Firmado por una empresa comercial, marca de tecnología, coches, banco, etc. (ej: Ford, Meta, Google, BBVA, Zara).
        - Firmado por instituciones gubernamentales o ONGs (ej: Gobierno de España, Greenpeace).
        - Notas de prensa firmadas por la marca.

    INSTRUCCIONES DE PRIORIDAD:
    1. Si "{nombre_periodista}" es una marca conocida (coches, tech, ropa) -> ELIGE 7.
    2. Si "{nombre_periodista}" es igual a "{nombre_medio}" -> ELIGE 6.
    3. Si "{nombre_periodista}" es una Agencia conocida -> ELIGE 5.

    Responde ÚNICAMENTE con el número dígito (0-7).
    """
    # 1. Llamada al modelo (tu función externa)
    respuesta_texto = consultar_ollama(prompt)

    # 2. Extracción del número (Pre-procesamiento)
    # Buscamos el primer dígito que aparezca en el texto
    match = re.search(r'\d+', respuesta_texto)
    
    numero_detectado = int(match.group()) if match else 0

    # 3. Validación con Pydantic
    try:
        # Instanciamos el modelo con el dato detectado
        # Si numero_detectado es 9 (alucinación), Pydantic dará error aquí
        resultado = GeneroPeriodistaValidado(codigo=numero_detectado)
        
        # Si llegamos aquí, es un int válido entre 0 y 5
        return resultado.codigo

    except ValidationError as e:
        print(f"Error de validación Pydantic (Dato inválido: {numero_detectado}): {e}")
        return 0 # Default seguro si la IA alucina un número fuera de rango


# =====================================================================================
# 12. Tema
# =====================================================================================
from utils import TemaConExplicacion

def clasificar_var_tema(titulo: str, texto_cuerpo: str) -> TemaConExplicacion:
    """
    Clasifica el tema y da una explicación.
    Devuelve un objeto Pydantic con .codigo (int) y .explicacion (str).
    """
    
    # 1. Validación rápida
    full_text = f"{titulo} {texto_cuerpo}"
    if not full_text or len(full_text) < 10:
        # Devolvemos objeto vacío/error
        return TemaConExplicacion(codigo=0, explicacion="Texto insuficiente para clasificar.")

    # Recorte para optimizar velocidad
    texto_recortado = texto_cuerpo[:1500]
    
    # 2. Prompt diseñado para JSON
    prompt = f"""
    Analiza la noticia:
    Título: "{titulo}"
    Extracto: "{texto_recortado}..."

    Tu tarea es clasificarla en UNA categoría (1-17) y explicar por qué.
    
    Categorías:
    1 = Científica / Investigación
    2 = Comunicación
    3 = De farándula o espectáculo
    4 = Deportiva
    5 = Economía (Mercados, inflación, consumo)
    6 = Educación/cultura
    7 = Empleo/Trabajo
    8 = Empresa (Corporativo, negocios)
    9 = Judicial
    10 = Medioambiente
    11 = Policial
    12 = Política
    13 = Salud
    14 = Social
    15 = Tecnología
    16 = Transporte
    17 = Otros

    FORMATO DE RESPUESTA OBLIGATORIO (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido con este formato:
    {{
        "codigo": (número entero del 1 al 17),
        "explicacion": "(frase breve justificando tu elección)"
    }}
    """

    # 3. Llamada al modelo
    respuesta_texto = consultar_ollama(prompt)

    # 4. Limpieza y Parsing de JSON
    # A veces los modelos envuelven el JSON en markdown ```json ... ```
    # Buscamos donde empieza '{' y termina '}'
    try:
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            raise ValueError("No se encontró JSON en la respuesta")
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str) # Convertimos texto a diccionario Python

        # 5. Validación Pydantic
        resultado = TemaConExplicacion(**data)
        return resultado

    except (json.JSONDecodeError, ValueError, ValidationError) as e:
        print(f"Error parseando respuesta del modelo: {e}")
        # Retorno de seguridad en caso de fallo
        return TemaConExplicacion(codigo=0, explicacion=f"Error técnico: {str(e)}")


# =====================================================================================
# 12b. Sección
# =====================================================================================
def clasificar_var_seccion(articulo: Article) -> str:
    """
    Extrae la sección del periódico basándose en metadatos y la URL.
    No usa IA (es más rápido y exacto para este dato estructural).
    """
    
    # --- 1. BUSCAR EN METADATOS (La fuente más fiable) ---
    meta = articulo.meta_data
    
    # Lista de claves comunes donde los medios guardan la sección
    claves_seccion = [
        'section',           # Estándar simple
        'article:section',   # Protocolo Open Graph (Facebook/LinkedIn)
        'og:section',        # Variación Open Graph
        'category',          # WordPress y CMS comunes
        'dc.subject',        # Dublin Core standard
        'ut.section'         # Algunos medios custom
    ]

    for clave in claves_seccion:
        valor = meta.get(clave)
        # A veces el valor es una lista, tomamos el primero
        if isinstance(valor, list):
            valor = valor[0]
        
        if valor and isinstance(valor, str) and len(valor) > 2:
            return valor.strip().title() # Ej: "DEPORTES " -> "Deportes"

    # --- 2. BUSCAR EN LA URL (Si no hay metadatos) ---
    # Ejemplo: https://www.elmundo.es/economia/2024/02/10/noticia.html
    # Queremos extraer "economia"
    
    path = urlparse(articulo.url).path
    segmentos = path.split('/')
    
    # Filtramos segmentos vacíos
    segmentos = [s for s in segmentos if s]

    for segmento in segmentos:
        # Ignoramos segmentos que son años (4 dígitos) o muy cortos (idiomas 'es', 'en')
        if re.match(r'^\d{4}$', segmento): # Es un año (2024)
            continue
        if re.match(r'^\d{1,2}$', segmento): # Es un día o mes (10, 02)
            continue
        if len(segmento) <= 2: # Es un código de idioma (es, en, cat)
            continue
        if segmento in ['noticia', 'articulo', 'story', 'news']: # Palabras genéricas
            continue
            
        # Si pasa los filtros, asumimos que es la sección
        # Reemplazamos guiones por espacios (ej: "ciencia-y-salud" -> "Ciencia Y Salud")
        return segmento.replace('-', ' ').title()

    # --- 3. DEFECTO ---
    return "General"


########################################################################################################################
########################################################################################################################

# =====================================================================================
# 13. IA Tema Central
# =====================================================================================
from utils import IaTemaCentralConExplicacion

def clasificar_var_ia_tema_central(titulo: str, texto_cuerpo: str) -> IaTemaCentralConExplicacion:
    """
    Determina si la Inteligencia Artificial es el TEMA CENTRAL de la noticia.
    Devuelve objeto con .codigo y .explicacion.
    """
    
    # Unificamos texto
    texto_completo = (titulo + " " + texto_cuerpo).lower()

    # --- 1. FILTRO DE EFICIENCIA (Heurística) ---
    palabras_clave_ia = [
        "inteligencia artificial", "artificial intelligence", "ia ", "ai ", 
        "chatgpt", "gpt", "llm", "machine learning", "aprendizaje automático",
        "red neuronal", "deep learning", "midjourney", "dall-e", "bard", "gemini",
        "copilot", "algoritmo generativo", "sam altman", "openai", "nvidia"
    ]
    
    # Si ninguna palabra clave está presente, asumimos directamente que NO (1)
    if not any(palabra in texto_completo for palabra in palabras_clave_ia):
        return IaTemaCentralConExplicacion(
            codigo=1,
            explicacion="El texto no contiene términos relacionados con la Inteligencia Artificial."
        )

    # --- 2. PROMPT CON SOLICITUD DE JSON ---
    # Recortamos texto para centrar la atención en el inicio (donde suele estar el tema central)
    texto_recortado = texto_cuerpo[:2500]
    
    prompt = f"""
    Analiza la jerarquía de la información en esta noticia:
    
    TÍTULO: "{titulo}"
    TEXTO: "{texto_recortado}..."

    Objetivo: Determinar si la Inteligencia Artificial (IA) es el TEMA PRINCIPAL y PROTAGONISTA.
    
    Criterios de clasificación:

    1 = No (Mención secundaria / Otro tema):
        - La IA se menciona al final o de pasada.
        - Es un discurso (Papa, Políticos) sobre varios temas y la IA es solo uno más.
        - La IA es una herramienta secundaria (ej: "Policía usa IA para un robo", el tema es el robo).
        - El Título NO menciona tecnología o IA.

    2 = Sí (Tema Central):
        - La noticia gira completamente en torno a la IA (avances, regulación, peligros, inversiones).
        - La IA es el sujeto principal del Título.

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido:
    {{
        "codigo": (1 o 2),
        "explicacion": "(Breve justificación analizando si la IA es protagonista o secundaria)"
    }}
    """

    # --- 3. LLAMADA AL MODELO ---
    respuesta_texto = consultar_ollama(prompt)

    # --- 4. EXTRACCIÓN Y VALIDACIÓN ---
    try:
        # Buscamos el bloque JSON
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            return IaTemaCentralConExplicacion(
                codigo=1, 
                explicacion="Error: El modelo no devolvió un formato JSON válido."
            )
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)

        # Validación Pydantic
        return IaTemaCentralConExplicacion(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        # Fallback seguro
        return IaTemaCentralConExplicacion(
            codigo=1, 
            explicacion=f"Error técnico al procesar la respuesta: {str(e)}"
        )


# =====================================================================================
# 14. Significado IA
# =====================================================================================
from utils import IaSignificadoConExplicacion

def clasificar_var_significado_ia(titulo: str, texto_cuerpo: str) -> IaSignificadoConExplicacion:
    """
    Detecta si el artículo explica o define QUÉ ES la IA o CÓMO FUNCIONA.
    Devuelve un objeto con .codigo (1/2) y .explicacion (str).
    """
    
    # Unificamos texto
    texto_completo = (titulo + " " + texto_cuerpo).lower()

    # --- 1. FILTRO DE EFICIENCIA (Heurística) ---
    # Si no hay palabras "inteligencia" o "algoritmo", difícilmente explicará qué son.
    keywords_tecnicas = ["inteligencia", "ia ", "ai ", "algoritmo", "red neuronal", "modelo de lenguaje"]
    
    if not any(k in texto_completo for k in keywords_tecnicas):
        return IaSignificadoConExplicacion(
            codigo=1,
            explicacion="El texto no contiene términos técnicos básicos para ofrecer una definición de IA."
        )

    # --- 2. PROMPT EN FORMATO JSON ---
    # Recortamos el texto (buscamos definiciones, suelen estar al principio)
    texto_recortado = texto_cuerpo[:2000]
    
    prompt = f"""
    Analiza el siguiente texto periodístico con enfoque pedagógico:
    Título: "{titulo}"
    Extracto: "{texto_recortado}..."

    Tu tarea: Determinar si el artículo contiene una EXPLICACIÓN o DEFINICIÓN sobre qué es la Inteligencia Artificial (IA) o cómo funciona técnicamente.

    Criterios de clasificación:
    
    1 = No (Mera mención o uso):
        - El artículo habla de herramientas (ChatGPT, Bard) o noticias de empresas sin explicar qué son.
        - Habla de "la IA" como un sujeto abstracto ("la IA cambiará el mundo") sin definirla.
        - Ej: "Google lanzó su nueva IA ayer". Aquí NO se aprende qué es la tecnología.

    2 = Sí (Didáctico / Definitorio):
        - El texto tiene intención educativa.
        - Contiene frases tipo: "La IA generativa funciona prediciendo el siguiente token...", "Los LLM son modelos entrenados con...".
        - Explica la diferencia técnica entre tipos de IA.

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido:
    {{
        "codigo": (1 o 2),
        "explicacion": "(Breve frase justificando si hay definiciones técnicas o solo menciones)"
    }}
    """

    # --- 3. LLAMADA AL MODELO ---
    respuesta_texto = consultar_ollama(prompt)

    # --- 4. EXTRACCIÓN Y VALIDACIÓN ---
    try:
        # Buscamos el bloque JSON por si el modelo añade texto extra
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            return IaSignificadoConExplicacion(
                codigo=1, 
                explicacion="Error: El modelo no devolvió un formato JSON válido."
            )
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)

        # Validación con Pydantic
        return IaSignificadoConExplicacion(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        return IaSignificadoConExplicacion(
            codigo=1, 
            explicacion=f"Error técnico al procesar la respuesta: {str(e)}"
        )

# =====================================================================================
# 15. Menciona IA
# =====================================================================================
from utils import MencionIaConExplicacion

def clasificar_var_menciona_ia(titulo: str, texto_cuerpo: str) -> MencionIaConExplicacion:
    """
    Detecta si se menciona la IA y lista TODAS las palabras clave encontradas.
    Usa Regex y palabras clave (No requiere Ollama).
    """
    
    # Unificamos texto
    texto_completo = f"{titulo} \n {texto_cuerpo}"
    texto_lower = texto_completo.lower()
    
    # Creamos un conjunto (set) para evitar palabras repetidas
    palabras_encontradas = set()
    
    # --- 1. BÚSQUEDA DE CONCEPTOS CLAROS ---
    palabras_clave = [
        "inteligencia artificial", "artificial intelligence",
        "machine learning", "aprendizaje automático", "deep learning",
        "redes neuronales", "chatgpt", "generative ai", "ia generativa",
        "openai", "midjourney", "dall-e", "bard", "gemini", "copilot",
        "large language model", " llm ", "algoritmos generativos",
        "sam altman", "sora", "claude 3", "llama 3", "mistral",
        "vision pro", "neuralink"
    ]
    
    # Iteramos sobre todas las palabras y si están, las añadimos al set
    for frase in palabras_clave:
        if frase in texto_lower:
            palabras_encontradas.add(frase.strip()) # strip para quitar espacios de " llm "

    # --- 2. BÚSQUEDA DE SIGLAS "IA" o "AI" (Case Sensitive) ---
    # Usamos re.findall para encontrar TODAS las ocurrencias, no solo la primera
    patron_siglas = r'\b(IA|AI|I\.A\.|A\.I\.)\b'
    coincidencias_siglas = re.findall(patron_siglas, texto_completo)
    
    # Añadimos las siglas encontradas al conjunto
    for sigla in coincidencias_siglas:
        palabras_encontradas.add(sigla)

    # --- 3. CONSTRUCCIÓN DE RESPUESTA ---
    
    # Si el conjunto tiene elementos, es un SÍ (2)
    if palabras_encontradas:
        # Convertimos el set a una lista ordenada y la unimos con comas
        lista_final = ", ".join(sorted(palabras_encontradas))
        return MencionIaConExplicacion(
            codigo=2,
            explicacion=lista_final
        )

    # Si el conjunto está vacío, es un NO (1)
    return MencionIaConExplicacion(
        codigo=1,
        explicacion="No se encontraron términos, siglas ni conceptos relacionados con la Inteligencia Artificial en el texto."
    )

# =====================================================================================
# 16. Referencia a políticas en materia de género e igualdad
# =====================================================================================
from utils import ReferenciaPoliticasGeneroConExplicacion

def clasificar_var_referencia_politicas_genero(titulo: str, texto_cuerpo: str) -> ReferenciaPoliticasGeneroConExplicacion:
    """
    Detecta si la noticia hace referencia a POLÍTICAS, LEYES o DEBATES sobre 
    género, igualdad, feminismo o violencia machista.
    Devuelve objeto con .codigo y .explicacion.
    """
    
    # Unificamos y limpiamos texto para el filtro
    texto_completo = (titulo + " " + texto_cuerpo).lower()
    
    # --- 1. FILTRO DE EFICIENCIA (Heurística) ---
    raices_clave = [
        "igualdad", "género", "genero", "femin", "mujer", "machis", 
        "brecha", "paridad", "sexis", "patriarca", "trans ", "lgtbi",
        "conciliación", "techo de cristal", "violencia vicaria", "víctima",
        "ley", "ministerio", "protesta", "derechos" # Añadido contexto político/legal
    ]
    
    # Verificamos si hay al menos una palabra de género Y contexto político/social
    # (Simplificado: si no hay ninguna raíz clave, descartamos).
    if not any(raiz in texto_completo for raiz in raices_clave):
        return ReferenciaPoliticasGeneroConExplicacion(
            codigo=1,
            explicacion="El texto no contiene términos clave relacionados con género, igualdad o políticas sociales."
        )

    # --- 2. PROMPT ESPECÍFICO (Modo JSON) ---
    # Recortamos el texto para no saturar el contexto
    texto_recortado = texto_cuerpo[:2000]
    
    prompt = f"""
    Analiza la siguiente noticia:
    Título: "{titulo}"
    Extracto: "{texto_recortado}..."

    Tu tarea: Determinar si el texto hace referencia a POLÍTICAS DE GÉNERO, LEYES DE IGUALDAD o DEBATES SOBRE DERECHOS DE LA MUJER.

    Criterios de clasificación:
    1 = No:
        - La mujer es mencionada solo como protagonista de un hecho (ej: "La alcaldesa inauguró la feria").
        - Sucesos o crímenes sin contexto social/legal.
    
    2 = Sí:
        - Menciona leyes, cuotas, paridad o medidas gubernamentales sobre igualdad.
        - Habla de violencia machista/género como problema estructural o legal.
        - Trata sobre feminismo, 8M, brecha salarial o discriminación laboral.

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido:
    {{
        "codigo": (1 o 2),
        "explicacion": "(Frase breve justificando si se trata de política/derechos o es solo una mención circunstancial)"
    }}
    """

    # --- 3. Llamada al modelo ---
    respuesta_texto = consultar_ollama(prompt)

    # --- 4. Procesamiento JSON ---
    try:
        # Buscamos el JSON dentro de la respuesta (por si el modelo añade texto extra)
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            # Fallback si no hay JSON
            return ReferenciaPoliticasGeneroConExplicacion(
                codigo=1, 
                explicacion="Error: El modelo no devolvió un formato válido."
            )
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)

        # Validación final con Pydantic
        return ReferenciaPoliticasGeneroConExplicacion(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        return ReferenciaPoliticasGeneroConExplicacion(
            codigo=1, 
            explicacion=f"Error técnico al procesar la respuesta: {str(e)}"
        )


# =====================================================================================
# 17. Denuncia a la desigualdad de género
# =====================================================================================
from utils import DenunciaDesigualdadConExplicacion

def clasificar_var_denuncia_desigualdad_genero(titulo: str, texto_cuerpo: str) -> DenunciaDesigualdadConExplicacion:
    """
    Detecta si la noticia DENUNCIA desigualdad y explica por qué.
    Devuelve objeto con .codigo (1/2) y .explicacion (str).
    """
    
    # Unificamos texto
    texto_completo = (titulo + " " + texto_cuerpo).lower()
    
    # --- 1. FILTRO DE EFICIENCIA (Heurística) ---
    palabras_activadoras = [
        "desigualdad", "discriminaci", "brecha", "violencia", "machis", 
        "patriarca", "acos", "abus", "víctima", "feminici", "sexismo",
        "techo de cristal", "precariedad", "injusticia", "derechos de las mujeres",
        "igualdad real", "conciliación", "paridad"
    ]
    
    # Si NO hay palabras clave, retornamos 1 directamente con explicación automática
    if not any(p in texto_completo for p in palabras_activadoras):
        return DenunciaDesigualdadConExplicacion(
            codigo=1, 
            explicacion="El texto no contiene términos relacionados con género, desigualdad o violencia machista."
        )

    # --- 2. PROMPT PARA JSON ---
    texto_recortado = texto_cuerpo[:2000]
    
    prompt = f"""
    Analiza el enfoque de esta noticia:
    Título: "{titulo}"
    Extracto: "{texto_recortado}..."

    Tu tarea es determinar si el texto DENUNCIA o VISIBILIZA un problema de desigualdad de género.

    Criterios:
    1 = No (Neutro/Sucesos): Solo narra hechos sin crítica social, o habla de mujeres exitosas sin mencionar dificultades de género.
    2 = Sí (Denuncia/Crítica): Critica el machismo, aporta datos de brechas, denuncia violencia sistémica o cubre protestas feministas.

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido:
    {{
        "codigo": (1 o 2),
        "explicacion": "(Frase breve justificando si hay denuncia social o es meramente informativo)"
    }}
    """

    # --- 3. LLAMADA A OLLAMA ---
    respuesta_texto = consultar_ollama(prompt)

    # --- 4. PROCESAMIENTO DE RESPUESTA ---
    try:
        # Limpieza de bloques de código markdown si los hay
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            # Fallback si el modelo no devuelve JSON
            return DenunciaDesigualdadConExplicacion(
                codigo=1, 
                explicacion="Error: El modelo no devolvió un formato válido."
            )
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)

        # Validación Pydantic
        return DenunciaDesigualdadConExplicacion(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        # Si algo falla en el parseo, devolvemos un objeto seguro
        return DenunciaDesigualdadConExplicacion(
            codigo=1, 
            explicacion=f"Error técnico al procesar la respuesta: {str(e)}"
        )


# =====================================================================================
# 18. Presencia de mujeres racializadas en la noticia
# =====================================================================================
from utils import MujeresRacializadasConExplicacion

def clasificar_var_mujeres_racializadas_noticias(titulo: str, texto_cuerpo: str) -> MujeresRacializadasConExplicacion:
    """
    Detecta la presencia o mención de mujeres racializadas (no blancas/caucásicas) en la noticia.
    Devuelve objeto con .codigo (1/2) y .explicacion (str).
    """
    
    # Unificamos texto
    texto_completo = (titulo + " " + texto_cuerpo).lower()

    # --- 1. FILTRO DE EFICIENCIA (Heurística) ---
    # Buscamos términos que sugieran diversidad étnica, racial o contextos migratorios.
    # Si no aparece NADA de esto, asumimos que se habla de mujeres blancas o el tema no es racial.
    
    terminos_clave = [
        "racializada", "negra", "afro", "etnia", "raza", "indígena", 
        "gitana", "romaní", "latina", "hispana", "asiática", "árabe", 
        "musulmana", "morocc", "marroquí", "subsahariana", "migrante", 
        "refugiada", "islam", "velo", "hijab", "mestiza", "mulata",
        "origen", "nacionalidad", "extranjera", "diversidad"
    ]
    
    # Nota: Este filtro es laxo para no descartar falsos negativos, 
    # pero ayuda a limpiar noticias de política nacional estándar (ej: Ayuso, Montero).
    if not any(t in texto_completo for t in terminos_clave):
        return MujeresRacializadasConExplicacion(
            codigo=1,
            explicacion="El texto no contiene marcadores explícitos de diversidad étnica o racial."
        )

    # --- 2. PROMPT CON SOLICITUD DE JSON ---
    texto_recortado = texto_cuerpo[:2000]
    
    prompt = f"""
    Analiza la representación de las personas en esta noticia:
    Título: "{titulo}"
    Extracto: "{texto_recortado}..."

    Tu tarea: Determinar si en la noticia aparecen, se mencionan o protagonizan **MUJERES RACIALIZADAS**.
    
    Definición de "Mujer Racializada" para este análisis:
    Mujeres que son percibidas socialmente como no blancas en un contexto occidental. Incluye:
    - Mujeres negras / afrodescendientes.
    - Mujeres latinas / sudamericanas.
    - Mujeres asiáticas.
    - Mujeres árabes / magrebíes / musulmanas (contexto cultural-étnico).
    - Mujeres indígenas.
    - Mujeres gitanas / romaníes.

    Criterios de clasificación:
    
    1 = No:
        - Solo aparecen mujeres blancas / caucásicas (ej: políticas europeas, actrices de Hollywood blancas).
        - No se menciona el origen étnico y por el contexto se asume hegemonía blanca.
        - Se habla de "inmigrantes" en general sin especificar mujeres.

    2 = Sí:
        - Aparece explícitamente una mujer descrita por su etnia u origen (ej: "la activista afroamericana", "la cantante colombiana").
        - Se menciona a una figura pública conocida por ser racializada (ej: Kamala Harris, Rihanna, Salma Hayek, Zendaya) aunque no se diga su raza explícitamente en el texto.
        - Noticias sobre colectivos específicos (ej: "Las mujeres afganas", "Las temporeras marroquíes").

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido:
    {{
        "codigo": (1 o 2),
        "explicacion": "(Indica qué mujer o colectivo racializado se ha detectado y por qué)"
    }}
    """

    # --- 3. LLAMADA AL MODELO ---
    respuesta_texto = consultar_ollama(prompt)

    # --- 4. EXTRACCIÓN Y VALIDACIÓN ---
    try:
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            return MujeresRacializadasConExplicacion(
                codigo=1, 
                explicacion="Error: El modelo no devolvió un formato JSON válido."
            )
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)

        return MujeresRacializadasConExplicacion(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        return MujeresRacializadasConExplicacion(
            codigo=1, 
            explicacion=f"Error técnico al procesar la respuesta: {str(e)}"
        )


# =====================================================================================
# 19. Presencia de mujeres con discapacidad en la noticia
# =====================================================================================
from utils import MujeresConDiscapacidadConExplicacion

def clasificar_var_mujeres_con_discapacidad_noticias(titulo: str, texto_cuerpo: str) -> MujeresConDiscapacidadConExplicacion:
    """
    Detecta la presencia o mención explícita de mujeres con discapacidad o diversidad funcional.
    Devuelve objeto con .codigo (1/2) y .explicacion (str).
    """
    
    # Unificamos texto
    texto_completo = (titulo + " " + texto_cuerpo).lower()

    # --- 1. FILTRO DE EFICIENCIA (Heurística) ---
    # Palabras clave que sugieren discapacidad, diversidad funcional o condiciones específicas.
    # Si no aparece ninguna, descartamos la noticia.
    
    terminos_clave = [
        "discapacidad", "diversidad funcional", "silla de ruedas", "movilidad reducida",
        "ciega", "sorda", "sordomuda", "invidente", "autis", " tea ", "asperger",
        "síndrome de down", "parálisis", "cerebral", "amputada", "prótesis",
        "salud mental", "trastorno", "bipolar", "esquizofren", "depresio", # En contextos de discapacidad psicosocial
        "dependencia", "capacitism", "paralímpic", "once", "cermi"
    ]
    
    if not any(t in texto_completo for t in terminos_clave):
        return MujeresConDiscapacidadConExplicacion(
            codigo=1,
            explicacion="El texto no contiene términos relacionados con la discapacidad o diversidad funcional."
        )

    # --- 2. PROMPT CON SOLICITUD DE JSON ---
    texto_recortado = texto_cuerpo[:2000]
    
    prompt = f"""
    Analiza la representación de las personas en esta noticia:
    Título: "{titulo}"
    Extracto: "{texto_recortado}..."

    Tu tarea: Determinar si en la noticia aparecen, se mencionan o protagonizan **MUJERES CON DISCAPACIDAD**.

    Criterios de clasificación:
    
    1 = No:
        - Se menciona discapacidad, pero en HOMBRES (ej: "El atleta paralímpico ganó el oro").
        - Se usan términos metafóricos (ej: "La justicia es ciega", "parálisis política").
        - Son lesiones temporales (ej: "La jugadora se rompió la pierna y estará baja un mes").
        - Se habla de discapacidad en general (leyes, barreras) sin mencionar a ninguna mujer o colectivo femenino específico.

    2 = Sí:
        - Aparece una mujer (o niña) con discapacidad física, sensorial, intelectual o psicosocial.
        - Se habla de colectivos específicos (ej: "Las mujeres con discapacidad sufren más violencia").
        - Se menciona a deportistas paralímpicas, activistas con diversidad funcional, etc.

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido:
    {{
        "codigo": (1 o 2),
        "explicacion": "(Indica quién es la mujer y cuál es su discapacidad o contexto)"
    }}
    """

    # --- 3. LLAMADA AL MODELO ---
    # Gemma 4b suele ser bueno distinguiendo género en estos contextos
    respuesta_texto = consultar_ollama(prompt)

    # --- 4. EXTRACCIÓN Y VALIDACIÓN ---
    try:
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            return MujeresConDiscapacidadConExplicacion(
                codigo=1, 
                explicacion="Error: El modelo no devolvió un formato JSON válido."
            )
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)

        return MujeresConDiscapacidadConExplicacion(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        return MujeresConDiscapacidadConExplicacion(
            codigo=1, 
            explicacion=f"Error técnico al procesar la respuesta: {str(e)}"
        )


# =====================================================================================
# 20. Presencia de diversidad generacional en las mujeres que aparecen
# =====================================================================================
from utils import MujeresGeneracionalidadConExplicacion

def clasificar_var_mujeres_generacionalidad_noticias(titulo: str, texto_cuerpo: str) -> MujeresGeneracionalidadConExplicacion:
    """
    Detecta si en la noticia aparecen mujeres de **distintas generaciones** o de 
    **edades no hegemónicas** (niñas, adolescentes o ancianas).
    Devuelve objeto con .codigo (1/2) y .explicacion (str).
    """
    
    # Unificamos texto
    texto_completo = (titulo + " " + texto_cuerpo).lower()

    # --- 1. FILTRO DE EFICIENCIA (Heurística) ---
    # Buscamos marcadores de edad extremos o relaciones intergeneracionales.
    # Si no aparecen, asumimos que son adultos estándar (lo más común en noticias).
    
    terminos_edad = [
        # Infancia / Juventud
        "niña", "adolescente", "joven", "menor", "escolar", "alumna", "estudiante", 
        "chica", "hija", "infantil", "bebé", "generación z",
        # Vejez / Tercera Edad
        "anciana", "abuela", "jubilada", "mayor", "tercera edad", "senior", 
        "vejez", "pensionista", "octogenaria", "nonagenaria", "vieja", "residencia",
        # Relacional
        "madre", "nieta", "familia", "generaciones", "intergeneracional"
    ]
    
    if not any(t in texto_completo for t in terminos_edad):
        return MujeresGeneracionalidadConExplicacion(
            codigo=1,
            explicacion="El texto no contiene términos que sugieran diversidad de edades (niñas, ancianas) o relaciones intergeneracionales."
        )

    # --- 2. PROMPT CON SOLICITUD DE JSON ---
    texto_recortado = texto_cuerpo[:2000]
    
    prompt = f"""
    Analiza la edad y las generaciones de las mujeres en esta noticia:
    Título: "{titulo}"
    Extracto: "{texto_recortado}..."

    Tu tarea: Determinar si hay **DIVERSIDAD GENERACIONAL** en la representación femenina.

    Criterios de clasificación:
    
    1 = No (Representación Estándar):
        - Solo aparecen mujeres adultas en edad laboral típica (aprox 25-60 años). Ej: Políticas, profesionales, empresarias.
        - Se menciona "madre" solo como dato biográfico sin relevancia en la historia (ej: "es madre de dos hijos").
        - No se especifica la edad y se asume adultez.

    2 = Sí (Diversidad / Edades no hegemónicas):
        - Aparecen **Niñas o Adolescentes** con voz propia o como protagonistas.
        - Aparecen **Mujeres Mayores / Ancianas / Jubiladas** (Visibilidad de la tercera edad).
        - Hay un enfoque **Intergeneracional**: Se habla de madres e hijas, abuelas y nietas, o el impacto de un tema en distintas generaciones de mujeres.

    FORMATO DE RESPUESTA (JSON):
    Responde ÚNICAMENTE con un objeto JSON válido:
    {{
        "codigo": (1 o 2),
        "explicacion": "(Indica qué edades o relación generacional se ha detectado)"
    }}
    """

    # --- 3. LLAMADA AL MODELO ---
    # Usamos Gemma 4b (o tu modelo preferido)
    respuesta_texto = consultar_ollama(prompt)

    # --- 4. EXTRACCIÓN Y VALIDACIÓN ---
    try:
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1 or fin == 0:
            return MujeresGeneracionalidadConExplicacion(
                codigo=1, 
                explicacion="Error: El modelo no devolvió un formato JSON válido."
            )
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)

        return MujeresGeneracionalidadConExplicacion(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        return MujeresGeneracionalidadConExplicacion(
            codigo=1, 
            explicacion=f"Error técnico al procesar la respuesta: {str(e)}"
        )


# =====================================================================================
# 21. Tiene Fotografías y 22. Número de fotografías
# =====================================================================================
from utils import FotografiasValidadas

def clasificar_var_fotografias(articulo: Any) -> FotografiasValidadas:
    """
    Analiza las imágenes del artículo (Top Image + Cuerpo).
    Filtra iconos, basura y publicidad.
    
    Args:
        articulo: Objeto 'Article' de la librería newspaper3k ya descargado y parseado.
    """
    
    # Usamos un set para evitar duplicados (ej: si la top_image también sale en el texto)
    imagenes_reales = set()
    
    # --- 1. IMAGEN DE PORTADA (TOP IMAGE) ---
    if articulo.top_image and len(articulo.top_image) > 10:
        # Filtro básico: que sea una URL válida y no vacía
        imagenes_reales.add(articulo.top_image)

    # --- 2. IMÁGENES DEL CUERPO ---
    # Usamos clean_top_node (lxml object) que contiene solo el texto principal limpio
    nodo_texto = articulo.clean_top_node 
    
    if nodo_texto is not None:
        # Buscamos todas las etiquetas <img>
        imgs_en_texto = nodo_texto.xpath('.//img')
        
        for img in imgs_en_texto:
            src = img.get('src')
            if not src:
                continue
            
            # Normalizamos a minúsculas para chequear
            src_lower = src.lower()
            
            # --- FILTROS ANTI-BASURA (Heurística) ---
            
            # A. Descartar formatos que suelen ser de interfaz (iconos, spacers)
            if src_lower.endswith(('.svg', '.gif', '.ico')):
                continue
                
            # B. Palabras prohibidas (indican publicidad, tracking o diseño web)
            palabras_prohibidas = [
                'logo', 'icon', 'avatar', 'profile', 'pixel', 'spacer', 
                'doubleclick', 'adserver', 'banner', 'button', 'social',
                'facebook', 'twitter', 'whatsapp', 'share', 'sprite',
                'author', 'comment'
            ]
            
            if any(palabra in src_lower for palabra in palabras_prohibidas):
                continue

            # C. Descartar por dimensiones diminutas (si el HTML las tiene)
            # Muchos "pixels" de tracking son de 1x1
            width = img.get('width')
            height = img.get('height')
            
            # Si tiene width/height y es menor a 100px, seguramente no es una foto editorial
            if width and width.isdigit() and int(width) < 100:
                continue
            if height and height.isdigit() and int(height) < 100:
                continue

            # Si pasa los filtros, añadimos la URL
            imagenes_reales.add(src)

    # --- 3. CONSTRUCCIÓN DE RESPUESTA ---
    lista_urls = list(imagenes_reales)
    cantidad_final = len(lista_urls)
    
    # Lógica de código: 2 si hay fotos, 1 si no
    codigo_final = 2 if cantidad_final > 0 else 1

    return FotografiasValidadas(
        codigo=codigo_final,
        cantidad=cantidad_final,
        evidencias=lista_urls
    )


# =====================================================================================
# 23. Tiene Fuentes y 24. Número de Fuentes
# =====================================================================================
from utils import FuentesValidadas

def clasificar_var_tiene_fuentes(texto_noticia: str) -> FuentesValidadas:
    """
    Determina si la noticia tiene fuentes.
    Retorna codigo=2 si encuentra al menos una, codigo=1 si no encuentra nada.
    """
    
    # 1. Validación inicial
    if not texto_noticia or len(texto_noticia) < 50:
        return FuentesValidadas(codigo=1, evidencias=[], cantidad=0)

    # 2. Recorte (Analizamos el principio del texto donde se atribuyen las fuentes)
    texto_analisis = texto_noticia[:3500]

    # 3. Prompt: Solo pedimos la lista de nombres
    prompt = f"""
    Analiza el texto y extrae una lista de las FUENTES de información explícitas (personas, entidades, documentos) a las que se atribuyen los datos.

    TEXTO: "{texto_analisis}..."

    INSTRUCCIONES:
    - Busca verbos de atribución: "según X", "dijo Y", "informó Z", "fuentes de...", "el informe de...".
    - Si es un artículo de opinión sin datos externos, la lista debe estar vacía.

    Responde SOLO con este JSON:
    {{
        "fuentes": ["Nombre Fuente 1", "Nombre Fuente 2"]
    }}
    """

    # 4. Llamada al LLM
    respuesta_texto = consultar_ollama(prompt)

    # 5. Lógica Python (Determinista)
    try:
        inicio = respuesta_texto.find('{')
        fin = respuesta_texto.rfind('}') + 1
        
        if inicio == -1:
            return FuentesValidadas(codigo=1, evidencias=[], cantidad=0)
            
        json_str = respuesta_texto[inicio:fin]
        data = json.loads(json_str)
        
        # Extraemos la lista limpia
        lista_fuentes = data.get("fuentes", [])
        cantidad = len(lista_fuentes)
        
        # --- AQUÍ ESTÁ EL CAMBIO ---
        if cantidad > 0:
            # Si hay elementos -> Código 2 (Sí)
            return FuentesValidadas(
                codigo=2,
                evidencias=lista_fuentes,
                cantidad=cantidad
            )
        else:
            # Si la lista está vacía -> Código 1 (No)
            return FuentesValidadas(
                codigo=1,
                evidencias=[],
                cantidad=0
            )

    except Exception:
        return FuentesValidadas(codigo=1, evidencias=[], cantidad=0)


