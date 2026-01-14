from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from models import AnalisisContenidoGeneral, AgentState
from utils import cargar_config_ini, generar_instrucciones_codebook
from typing import Optional

CONFIG = cargar_config_ini()
MODEL_NAME = "gemma3:4b"
llm = ChatOllama(model=MODEL_NAME, temperature=0).with_structured_output(AnalisisContenidoGeneral)

def nodo_analista(state: AgentState):
    # Codebook (Siempre presente)
    instrucciones = generar_instrucciones_codebook(CONFIG, "CONTENIDO_GENERAL")

    prompt_txt = f"""Eres un clasificador experto de noticias.
    
    {instrucciones}
    
    ### TU TAREA
    Analiza la noticia actual y extrae los códigos.
    
    Tienes la siguiente información:
    - TITULAR: {state.get('titular', '')}
    - AUTOR: {state.get('autor', '')}
    - CONTENIDO: {state.get('texto_noticia', '')}
    
    Extrae:
    1. Nombre propio titular: El nombre propio que aparece en el titular (si no hay, pon 'No aplica')
    2. Cita titular: Si hay cita en el titular (0 = No, 1 = Sí)
    3. Género periodista: El género del periodista basado en el autor
    4. Género personas noticias: El género de las personas mencionadas en la noticia
    5. Temática noticias: La temática principal de la noticia según el libro de códigos
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