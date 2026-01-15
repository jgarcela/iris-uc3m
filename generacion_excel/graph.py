from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from models import AnalisisContenidoGeneral, AgentState
from utils import cargar_config_ini, generar_instrucciones_codebook
from typing import Optional

CONFIG = cargar_config_ini()
MODEL_NAME = "gemma3:4b"
llm = ChatOllama(model=MODEL_NAME, temperature=0).with_structured_output(AnalisisContenidoGeneral)

def formatear_codigos_para_prompt(codigos_dict):
    """Formatea un diccionario de códigos como texto plano para evitar que LangChain lo interprete como variables"""
    if not codigos_dict:
        return "No disponible"
    lines = []
    for codigo, descripcion in sorted(codigos_dict.items()):
        lines.append(f"  - Código '{codigo}': {descripcion}")
    return "\n".join(lines)

def nodo_analista(state: AgentState):
    # Codebook (Siempre presente)
    instrucciones = generar_instrucciones_codebook(CONFIG, "CONTENIDO_GENERAL")
    
    # Obtener los códigos específicos del config y formatearlos como texto
    contenido_general = CONFIG.get("CONTENIDO_GENERAL", {})
    tema_codes_str = formatear_codigos_para_prompt(contenido_general.get("TEMA", {}))
    genero_periodista_codes_str = formatear_codigos_para_prompt(contenido_general.get("GENERO_PERIODISTA", {}))
    cita_titular_codes_str = formatear_codigos_para_prompt(contenido_general.get("CITA_TITULAR", {}))
    genero_personas_codes_str = formatear_codigos_para_prompt(contenido_general.get("GENERO_PERSONAS_MENCIONADAS", {}))
    genero_nombre_propio_codes_str = formatear_codigos_para_prompt(contenido_general.get("GENERO_NOMBRE_PROPIO_TITULAR", {}))
    personas_mencionadas_codes_str = formatear_codigos_para_prompt(contenido_general.get("PERSONAS_MENCIONADAS", {}))

    prompt_txt = f"""Eres un clasificador experto de noticias.
    
    {instrucciones}
    
    ### INSTRUCCIONES ESPECÍFICAS
    
    IMPORTANTE: Debes devolver SOLO el código numérico (ID) que corresponda. NO devuelvas el texto descriptivo.
    
    Para cada variable, usa estos códigos exactos:
    
    **CITA_TITULAR**:
    {cita_titular_codes_str}
    
    **GENERO_PERIODISTA**:
    {genero_periodista_codes_str}
    
    **GENERO_PERSONAS_MENCIONADAS**:
    {genero_personas_codes_str}
    
    **GENERO_NOMBRE_PROPIO_TITULAR**:
    {genero_nombre_propio_codes_str}
    
    **PERSONAS_MENCIONADAS**:
    {personas_mencionadas_codes_str}
    
    **TEMA**:
    {tema_codes_str}
    
    ### TU TAREA
    Analiza la noticia y devuelve SOLO los códigos numéricos según el libro de códigos.
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
    
    return {"resultado": res}

workflow = StateGraph(AgentState)
workflow.add_node("analista", nodo_analista)
workflow.set_entry_point("analista")
workflow.add_edge("analista", END)
app_graph = workflow.compile()