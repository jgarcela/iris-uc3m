import os
import shutil
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Ruta donde se guarda la memoria
DB_DIR = "./data/vector_db_benchmark"

EMBEDDING_MODEL = "embeddinggemma:300m" 

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def inicializar_vector_db(df_datos, columnas_etiquetas, forzar_reinicio=False):
    """
    Indexa el dataset. 
    Si la base de datos ya existe, LA REUTILIZA para ahorrar tiempo.
    """
    
    # 1. COMPROBAR SI YA EXISTE
    # Si la carpeta existe, tiene archivos dentro y NO hemos pedido forzar reinicio...
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0 and not forzar_reinicio:
        print(f"\n💾 Base de datos encontrada en '{DB_DIR}'.")
        print("⏩ SALTANDO indexación. Usando la memoria ya existente.")
        return # <--- AQUÍ SE DETIENE Y NO VUELVE A CREARLA

    # 2. SI NO EXISTE (O FORZAMOS), LIMPIAMOS Y CREAMOS
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR) # Borrar la vieja si está corrupta o forzamos
    
    print("\n🔄 Creando nueva base de datos vectorial (esto tardará un poco)...")
    
    docs = []
    for idx, row in df_datos.iterrows():
        # Construimos la "Solución Correcta" para que sirva de ejemplo
        info_etiquetas = "ETIQUETADO CORRECTO (GROUND TRUTH):\n"
        for col_csv, nombre_var in columnas_etiquetas.items():
            valor = row.get(col_csv, 'Ns/Nc')
            info_etiquetas += f"- {nombre_var}: {valor}\n"

        # Metadatos para filtrar (evitar data leakage)
        meta = {
            "original_id": str(row.get('id', idx)), 
            "info_ground_truth": info_etiquetas
        }
        
        # El contenido es el texto completo
        doc = Document(page_content=str(row.get('full_text', '')), metadata=meta)
        docs.append(doc)
    
    # Guardar en lotes para no saturar memoria
    batch_size = 100
    print(f"📥 Indexando {len(docs)} documentos...")
    
    for i in range(0, len(docs), batch_size):
        subset = docs[i:i+batch_size]
        Chroma.from_documents(
            documents=subset,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        print(f"   - Lote {i} a {i+len(subset)} guardado.", end="\r")
        
    print(f"\n✅ Base de datos creada exitosamente en {DB_DIR}.")

def buscar_contexto_k_shot(texto_input, id_actual, k=3):
    """
    Busca k ejemplos similares, asegurándose de NO devolver la misma noticia actual.
    """
    # Cargamos la DB existente
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Pedimos k+1 por si el más parecido es él mismo
    try:
        results = vectorstore.similarity_search(texto_input, k=k+1)
    except Exception as e:
        return "Error recuperando contexto (DB vacía o corrupta)."
    
    contexto = ""
    count = 0
    for res in results:
        # Filtro: Si el ID coincide con la noticia actual, saltar (Data Leakage)
        if res.metadata.get("original_id") == str(id_actual):
            continue
        
        count += 1
        if count > k: break
        
        contexto += f"\n--- EJEMPLO SIMILAR {count} ---\n"
        # Recortamos un poco el texto para no gastar tanto token
        contexto += f"Texto Noticia: {res.page_content[:400]}...\n" 
        contexto += f"{res.metadata['info_ground_truth']}\n"
    
    return contexto