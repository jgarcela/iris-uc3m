import json
import os
import re

# Librerías Core
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# Librerías RAG (Carga de PDFs y VectorStore)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# 1. CONFIGURACIÓN DE LA BASE DE DATOS
# ==========================================

CARPETA_DB = "chroma_db"
# Asegúrate de usar el mismo modelo que usaste en indexer.py
MODELO_EMBEDDING = "embeddinggemma:300m" 

def cargar_retriever():
    if not os.path.exists(CARPETA_DB):
        print(f"❌ ERROR: No existe la carpeta '{CARPETA_DB}'.")
        print("   Ejecuta primero el script 'indexer.py' para procesar los PDFs.")
        return None
    
    print("💾 Cargando memoria desde disco...")
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    
    # Cargamos Chroma apuntando al directorio persistente
    vector_db = Chroma(
        persist_directory=CARPETA_DB,
        embedding_function=embeddings,
        collection_name="manual_estilo"
    )
    
    # k=3 recupera los 3 fragmentos más relevantes
    return vector_db.as_retriever(search_kwargs={"k": 3})

# Inicializamos el retriever globalmente
retriever = cargar_retriever()

# ==========================================
# 2. DEFINICIÓN DE LA TOOL
# ==========================================

@tool
def consultar_manual_estilo(consulta: str) -> str:
    """
    Busca normas en el Manual de Estilo (PDFs indexados). 
    Úsalo para verificar violencia, género, menores, fuentes, siglas, etc.
    Devuelve: Texto de la norma y Fuente (Nombre archivo / Página).
    """
    if not retriever:
        return "Error: Base de conocimientos no disponible. Ejecuta indexer.py."

    print(f"   [TOOL] Consultando DB: '{consulta}'")
    try:
        docs_rel = retriever.invoke(consulta)
    except Exception as e:
        return f"Error técnico en la búsqueda: {str(e)}"
    
    if not docs_rel:
        return "No se encontró información relevante en los manuales para esa consulta."
    
    resultado = []
    for d in docs_rel:
        # Extraemos metadatos guardados por el indexador
        fuente = os.path.basename(d.metadata.get("source", "doc_desconocido"))
        pag = d.metadata.get("page", 0)
        # Formato claro para que el LLM lo pueda citar en el JSON final
        resultado.append(f"FUENTE: {fuente} (Pág {pag}) | CONTENIDO: {d.page_content}")
        
    return "\n\n".join(resultado)

tools = [consultar_manual_estilo]

# ==========================================
# 3. CONFIGURACIÓN DEL AGENTE Y PROMPT
# ==========================================

llm = ChatOllama(model="qwen3:8b", temperature=0.4)

template = """
Eres un auditor de calidad periodística.

VARIABLES:
{variables}

DEFINICIONES DE VARIABLES:
{definiciones}

TIENES UNA HERRAMIENTA CRÍTICA:
- 'consultar_manual_estilo': Busca en los PDFs cargados. ÚSALA SIEMPRE para verificar normas.

OBJETIVO:
Tu objetivo es analizar la noticia sobre las VARIABLES con las DEFINICIONES DE VARIABLES que se te proporcionan y devolver un JSON con las variables analizadas.

FORMATO DE SALIDA:
{{ "variable": {{ "codigo": "...", "evidencia": ["..."], "fuente": ["..."] }} }}
{{ "variable": {{ "codigo": "...", "evidencia": ["..."], "fuente": ["..."] }} }}

PASOS PARA RAZONAR:
1. Lee el texto.
2. Si la tool dice que algo es incorrecto, marca la variable como "Sí" (tiene error/sesgo).
3. Devuelve JSON con el formato de salida.
4. Si no hay evidencia, la evidencia debe ser una lista vacía [].
5. Si no hay fuente, la fuente debe ser una lista vacía [].
6. Si no hay etiqueta, la etiqueta debe ser una lista vacía [].
7. Si no hay variable, la variable debe ser una lista vacía [].
8. Sustituye "variable" por el nombre de la variable que se está analizando.
9. "codigo" es la etiqueta que se ha asignado a la variable.
10. "evidencia" es la evidencia literal del texto que se ha encontrado para la variable.
11. "fuente" es la página o las páginas precisas de los archivos que respaldan de la evidencia.

NOTICIA:
Titular: {titular}
Texto: {texto}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

def limpiar_json(texto):
    """Extrae el JSON válido de la respuesta del agente."""
    if isinstance(texto, dict): return texto
    
    # Intento 1: Bloque Markdown
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL | re.IGNORECASE)
    if match: 
        try: return json.loads(match.group(1))
        except: pass
    
    # Intento 2: Llaves puras
    start = texto.find("{")
    end = texto.rfind("}")
    if start != -1 and end != -1: 
        try: return json.loads(texto[start:end+1])
        except: pass
        
    return {"raw": texto, "error": "No se pudo parsear el JSON"}

def analizar_noticia(titular, texto, defs):
    if not retriever: return {"error": "DB no cargada"}
    
    # Preparamos las strings para el prompt
    defs_str = json.dumps(defs, indent=2, ensure_ascii=False)
    vars_list_str = ", ".join(list(defs.keys()))
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    res = executor.invoke({
        "variables": vars_list_str,      # Inyectamos la lista de nombres
        "definiciones": defs_str,        # Inyectamos el diccionario completo
        "titular": titular,
        "texto": texto,
        "input": "Analiza la noticia y busca infracciones."
    })
    
    return limpiar_json(res["output"])

# ==========================================
# 4. EJECUCIÓN
# ==========================================

if __name__ == "__main__":
    # Asegúrate de poner un PDF en la carpeta 'conocimiento' antes de ejecutar
    
    titular = "Trágico final en el centro: un nuevo crimen pasional sacude la capital por culpa de los celos"

    texto = """
    La policía investiga el crimen pasional ocurrido ayer en el domicilio conyugal. 
    La víctima, conocida por ser la mujer del famoso arquitecto Pedro Ruiz, fue hallada sin vida 
    tras una fuerte discusión. El agresor, un vecino ejemplar según el barrio, habría actuado 
    cegado por la pasión y el miedo al abandono. Algunos testigos comentan que la chica solía 
    vestir de forma muy provocativa y tenía muchas amistades masculinas, lo que provocaba 
    constantes tensiones en la pareja. Los jueces ya se han hecho cargo del caso.
    """

    defs = {"lenguaje_sexista": {"1": "No", "2": "Sí"}}

    if retriever:
        print("\n--- INICIANDO AGENTE ---")
        resultado = analizar_noticia(titular, texto, defs)
        
        print("\n" + "="*40)
        print("RESULTADO JSON FINAL")
        print("="*40)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    else:
        print("❌ Error: No se pudo cargar la base de datos (retriever es None).")
        print("   Asegúrate de haber ejecutado 'indexer.py' primero.")