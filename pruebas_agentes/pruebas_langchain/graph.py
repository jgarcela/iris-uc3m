from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from models import AnalisisContenidoGeneral
from utils import cargar_config_ini, generar_instrucciones_codebook
from rag_engine import buscar_contexto_k_shot
from typing import TypedDict, Optional

CONFIG = cargar_config_ini()
MODEL_NAME = "gemma3:4b" # "llama3.1:8b"
llm = ChatOllama(model=MODEL_NAME, temperature=0).with_structured_output(AnalisisContenidoGeneral)

# Añadimos 'use_rag' al estado
class AgentState(TypedDict):
    id_noticia: str
    texto_noticia: str
    use_rag: bool  # <--- INTERRUPTOR
    resultado: Optional[AnalisisContenidoGeneral]

def nodo_analista(state: AgentState):
    # 1. Codebook (Siempre presente)
    instrucciones = generar_instrucciones_codebook(CONFIG, "CONTENIDO_GENERAL")
    
    # 2. Lógica Condicional del RAG
    contexto_rag = ""
    if state.get('use_rag', False):
        # Solo buscamos si el interruptor está encendido
        ejemplos = buscar_contexto_k_shot(state['texto_noticia'], state['id_noticia'], k=3)
        contexto_rag = f"""
        ### EJEMPLOS DE REFERENCIA (MEMORIA)
        Mira cómo se etiquetaron estas noticias similares y SIGUE SU LÓGICA:
        {ejemplos}
        """
    else:
        contexto_rag = "No tienes ejemplos previos. Bása tu análisis únicamente en el Libro de Códigos."

    prompt_txt = f"""Eres un clasificador experto de noticias.
    
    {instrucciones}
    
    {contexto_rag}
    
    ### TU TAREA
    Analiza la noticia actual y extrae los códigos.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_txt),
        ("human", "NOTICIA ACTUAL:\n{texto_noticia}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({"texto_noticia": state['texto_noticia']})
    
    return {"resultado": res}

workflow = StateGraph(AgentState)
workflow.add_node("analista", nodo_analista)
workflow.set_entry_point("analista")
workflow.add_edge("analista", END)
app_graph = workflow.compile()