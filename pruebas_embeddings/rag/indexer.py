import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# CONFIGURACIÓN
CARPETA_PDFS = "conocimiento"
CARPETA_DB = "chroma_db"  # Aquí se guardará la "memoria" del agente
MODELO_EMBEDDING = "embeddinggemma:300m" # Asegúrate de tenerlo: ollama pull embeddinggemma:300m

def indexar_pdfs():
    # 1. Verificar PDFs
    if not os.path.exists(CARPETA_PDFS):
        os.makedirs(CARPETA_PDFS)
        print(f"❌ La carpeta '{CARPETA_PDFS}' no existe. Créala y mete PDFs dentro.")
        return

    archivos = glob.glob(os.path.join(CARPETA_PDFS, "*.pdf"))
    if not archivos:
        print(f"❌ No hay PDFs en '{CARPETA_PDFS}'.")
        return

    print(f"📚 Encontrados {len(archivos)} PDFs. Iniciando carga...")

    # 2. Cargar y Trocear
    docs_totales = []
    for pdf in archivos:
        print(f"   - Procesando: {os.path.basename(pdf)}")
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        docs_totales.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = splitter.split_documents(docs_totales)
    print(f"✂️  Generados {len(splits)} fragmentos de texto.")

    # 3. Crear y Persistir la Base de Datos
    print(f"🧠 Generando embeddings ({MODELO_EMBEDDING}) y guardando en disco...")
    
    # Si la carpeta DB ya existe, la borramos para re-indexar limpio (opcional)
    if os.path.exists(CARPETA_DB):
        import shutil
        shutil.rmtree(CARPETA_DB)

    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    
    # Al pasar persist_directory, Chroma guarda todo en disco automáticamente
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="manual_estilo",
        persist_directory=CARPETA_DB
    )
    
    print(f"✅ ¡ÉXITO! Base de datos guardada en folder './{CARPETA_DB}'")

if __name__ == "__main__":
    indexar_pdfs()