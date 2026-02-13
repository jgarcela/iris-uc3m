from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from models import AnalisisContenidoGeneral, AnalisisLenguaje, AnalisisFuentes, AgentState
from utils import cargar_config_ini, generar_instrucciones_codebook
from typing import Optional

CONFIG = cargar_config_ini()
MODEL_NAME = "gemma3:4b"

def formatear_codigos_para_prompt(codigos_dict):
    """Formatea un diccionario de códigos como texto plano para evitar que LangChain lo interprete como variables"""
    if not codigos_dict:
        return "No disponible"
    lines = []
    for codigo, descripcion in sorted(codigos_dict.items()):
        lines.append(f"  - Código '{codigo}': {descripcion}")
    return "\n".join(lines)

def nodo_analista_contenido(state: AgentState):
    """Analiza variables de CONTENIDO_GENERAL"""
    llm = ChatOllama(model=MODEL_NAME, temperature=0).with_structured_output(AnalisisContenidoGeneral)
    
    instrucciones = generar_instrucciones_codebook(CONFIG, "CONTENIDO_GENERAL")
    contenido_general = CONFIG.get("CONTENIDO_GENERAL", {})
    
    tema_codes_str = formatear_codigos_para_prompt(contenido_general.get("TEMA", {}))
    genero_periodista_codes_str = formatear_codigos_para_prompt(contenido_general.get("GENERO_PERIODISTA", {}))
    cita_titular_codes_str = formatear_codigos_para_prompt(contenido_general.get("CITA_TITULAR", {}))
    genero_personas_codes_str = formatear_codigos_para_prompt(contenido_general.get("GENERO_PERSONAS_MENCIONADAS", {}))
    genero_nombre_propio_codes_str = formatear_codigos_para_prompt(contenido_general.get("GENERO_NOMBRE_PROPIO_TITULAR", {}))
    personas_mencionadas_codes_str = formatear_codigos_para_prompt(contenido_general.get("PERSONAS_MENCIONADAS", {}))
    menciona_ia_codes_str = formatear_codigos_para_prompt(contenido_general.get("MENCIONA_IA", {}))
    ia_tema_central_codes_str = formatear_codigos_para_prompt(contenido_general.get("IA_TEMA_CENTRAL", {}))
    significado_ia_codes_str = formatear_codigos_para_prompt(contenido_general.get("SIGNIFICADO_IA", {}))
    lenguaje_vars_codes_str = formatear_codigos_para_prompt(CONFIG.get("LENGUAJE", {}).get("LENGUAJE_VARS", {}))

    prompt_txt = f"""Eres un clasificador experto de noticias especializado en CONTENIDO GENERAL.
    
    {instrucciones}
    
    ### INSTRUCCIONES ESPECÍFICAS - CRÍTICO
    
    ⚠️ REGLA ABSOLUTA: Debes devolver SOLO el código numérico (ID) como STRING. 
    ❌ NO devuelvas texto descriptivo como "Sí", "No", "Masculino", etc.
    ✅ SÍ devuelve códigos numéricos como "1", "2", "3", "0", etc.
    
    **VARIABLES DE CONTENIDO GENERAL:**
    
    TEMA:
    {tema_codes_str}
    
    GENERO_PERIODISTA:
    {genero_periodista_codes_str}
    
    CITA_TITULAR:
    {cita_titular_codes_str}
    
    **ORDEN LÓGICO - VARIABLES DE NOMBRE PROPIO EN TITULAR:**
    
    ⚠️ IMPORTANTE: Debes seguir este orden lógico:
    
    1. PRIMERO: Determina si hay nombre propio (nombre de persona) en el titular (nombre_propio_titular)
       - Si hay nombre propio: escribe SOLO el nombre de la persona (ej: 'María García', 'Pedro Sánchez', 'Luther King')
       - Si NO hay nombre propio: escribe 'No aplica'
       - ⚠️ CRÍTICO: NO escribas el titular completo. Solo el NOMBRE de la persona.
       - Ejemplo: Si el titular es "¿Dónde están los Luther King de hoy?", el nombre propio es "Luther King" (o "Martin Luther King"), NO el titular completo.
    
    2. LUEGO: SOLO SI hay nombre propio (nombre_propio_titular != 'No aplica'), determina el género:
       GENERO_NOMBRE_PROPIO_TITULAR:
       {genero_nombre_propio_codes_str}
       - Si nombre_propio_titular = 'No aplica', entonces genero_nombre_propio_titular_id DEBE ser '1' (No hay)
       - Si nombre_propio_titular tiene un nombre real, entonces determina el género ('2', '3' o '4')
    
    **ORDEN LÓGICO - VARIABLES DE PERSONAS MENCIONADAS:**
    
    ⚠️ IMPORTANTE: Debes seguir este orden lógico:
    
    1. PRIMERO: Determina si hay personas mencionadas (personas_mencionadas_id)
       PERSONAS_MENCIONADAS:
       {personas_mencionadas_codes_str}
       - '1' = No hay personas mencionadas
       - '2' = Sí hay personas mencionadas
    
    2. LUEGO: SOLO SI hay personas mencionadas (personas_mencionadas_id = '2'), determina el género:
       GENERO_PERSONAS_MENCIONADAS:
       {genero_personas_codes_str}
       - Si personas_mencionadas_id = '1' (No), entonces genero_personas_mencionadas_id DEBE ser '1' (No hay)
       - Si personas_mencionadas_id = '2' (Sí), entonces determina el género ('2', '3' o '4')
    
    **VARIABLES DE IA:**
    
    MENCIONA_IA:
    {menciona_ia_codes_str}
    
    IA_TEMA_CENTRAL (IA principal):
    {ia_tema_central_codes_str}
    
    SIGNIFICADO_IA:
    {significado_ia_codes_str}
    
    **VARIABLES DE IGUALDAD Y DIVERSIDAD:**
    
    IMPORTANTE: Estas variables SOLO aceptan los códigos '1' (No) o '2' (Sí). NO uses otros códigos.
    - Referencias políticas igualdad: '1' = No, '2' = Sí
    - Denuncia desigualdad género: '1' = No, '2' = Sí
    - Presencia mujeres racializadas: '1' = No, '2' = Sí
    - Presencia mujeres discapacidad: '1' = No, '2' = Sí
    - Presencia diversidad generacional: '1' = No, '2' = Sí
    
    ### TU TAREA
    Analiza SOLO las variables de CONTENIDO GENERAL y devuelve SOLO los códigos numéricos según el libro de códigos.
    
    ⚠️ RECUERDA EL ORDEN LÓGICO:
    - PRIMERO determina si hay nombre propio / personas mencionadas
    - LUEGO, SOLO SI hay, determina el género correspondiente
    - NO tiene sentido determinar el género si no hay nombre/personas
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_txt),
        ("human", "TITULAR:\n{titular}\n\nAUTOR:\n{autor}\n\nCONTENIDO:\n{texto_noticia}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({
        "titular": state.get('titular', ''),
        "autor": state.get('autor', ''),
        "texto_noticia": state.get('texto_noticia', '')
    })
    
    return {"resultado_contenido": res}

def nodo_analista_lenguaje(state: AgentState):
    """Analiza variables de LENGUAJE"""
    llm = ChatOllama(model=MODEL_NAME, temperature=0).with_structured_output(AnalisisLenguaje)
    
    instrucciones = generar_instrucciones_codebook(CONFIG, "LENGUAJE")
    lenguaje = CONFIG.get("LENGUAJE", {})
    
    lenguaje_sexista_codes_str = formatear_codigos_para_prompt(lenguaje.get("LENGUAJE_SEXISTA", {}))
    lenguaje_vars_codes_str = formatear_codigos_para_prompt(lenguaje.get("LENGUAJE_VARS", {}))

    prompt_txt = f"""Eres un clasificador experto de noticias especializado en LENGUAJE SEXISTA.
    
    {instrucciones}
    
    ### INSTRUCCIONES ESPECÍFICAS - CRÍTICO
    
    ⚠️ REGLA ABSOLUTA: Debes devolver SOLO el código numérico (ID) como STRING. 
    ❌ NO devuelvas texto descriptivo como "Sí", "No", etc.
    ✅ SÍ devuelve códigos numéricos como "1", "2", "3", etc.
    
    **VARIABLES DE LENGUAJE:**
    
    LENGUAJE_SEXISTA:
    {lenguaje_sexista_codes_str}
    
    IMPORTANTE: Las siguientes variables SOLO aceptan los códigos '1' (No) o '2' (Sí). NO uses otros códigos.
    - Masculino genérico: '1' = No, '2' = Sí
    - Hombre humanidad: '1' = No, '2' = Sí
    - Uso dual zorra: '1' = No, '2' = Sí
    - No uso cargos mujeres: '1' = No, '2' = Sí
    - Sexismo social: '1' = No, '2' = Sí
    - Androcentrismo: '1' = No, '2' = Sí
    - Mujeres sin nombre: '1' = No, '2' = Sí
    - Asimetría mujer/hombre: '1' = No, '2' = Sí
    - Infantilización mujeres: '1' = No, '2' = Sí
    - Denominación sexualizada: '1' = No, '2' = Sí
    - Denominación redundante: '1' = No, '2' = Sí
    - Denominación dependiente: '1' = No, '2' = Sí
    - Criterios excepción: '1' = No, '2' = Sí
    - Comparación mujeres/hombres: '1' = No, '2' = Sí
    
    ### TU TAREA
    Analiza SOLO las variables de LENGUAJE y devuelve SOLO los códigos numéricos según el libro de códigos.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_txt),
        ("human", "TITULAR:\n{titular}\n\nAUTOR:\n{autor}\n\nCONTENIDO:\n{texto_noticia}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({
        "titular": state.get('titular', ''),
        "autor": state.get('autor', ''),
        "texto_noticia": state.get('texto_noticia', '')
    })
    
    return {"resultado_lenguaje": res}

def nodo_analista_fuentes(state: AgentState):
    """Analiza variables de FUENTES"""
    llm = ChatOllama(model=MODEL_NAME, temperature=0).with_structured_output(AnalisisFuentes)
    
    instrucciones = generar_instrucciones_codebook(CONFIG, "FUENTES")
    lenguaje = CONFIG.get("LENGUAJE", {})
    lenguaje_vars_codes_str = formatear_codigos_para_prompt(lenguaje.get("LENGUAJE_VARS", {}))

    prompt_txt = f"""Eres un clasificador experto de noticias especializado en FUENTES.
    
    {instrucciones}
    
    ### INSTRUCCIONES ESPECÍFICAS - CRÍTICO
    
    ⚠️ REGLA ABSOLUTA: Debes devolver SOLO el código numérico (ID) como STRING. 
    ❌ NO devuelvas texto descriptivo como "Sí", "No", etc.
    ✅ SÍ devuelve códigos numéricos como "1", "2", etc.
    
    **VARIABLES DE FUENTES:**
    
    IMPORTANTE: Esta variable SOLO acepta los códigos '1' (No) o '2' (Sí). NO uses otros códigos.
    - Tiene fuente: '1' = No, '2' = Sí
    
    Número declaraciones: Devuelve el número total de declaraciones/fuentes como string (ej: "0", "1", "3")
    
    ### TU TAREA
    Analiza SOLO las variables de FUENTES y devuelve SOLO los códigos numéricos según el libro de códigos.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_txt),
        ("human", "TITULAR:\n{titular}\n\nAUTOR:\n{autor}\n\nCONTENIDO:\n{texto_noticia}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({
        "titular": state.get('titular', ''),
        "autor": state.get('autor', ''),
        "texto_noticia": state.get('texto_noticia', '')
    })
    
    return {"resultado_fuentes": res}

# Crear el workflow con 3 nodos independientes
workflow = StateGraph(AgentState)
workflow.add_node("analista_contenido", nodo_analista_contenido)
workflow.add_node("analista_lenguaje", nodo_analista_lenguaje)
workflow.add_node("analista_fuentes", nodo_analista_fuentes)
workflow.set_entry_point("analista_contenido")
workflow.add_edge("analista_contenido", "analista_lenguaje")
workflow.add_edge("analista_lenguaje", "analista_fuentes")
workflow.add_edge("analista_fuentes", END)
app_graph = workflow.compile()
