import configparser
import json
import ast
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- 1. FUNCIÓN DE CARGA DE CONFIGURACIÓN (INI) ---
def cargar_definiciones_de_ini(ruta_ini, nombre_seccion):
    if not os.path.exists(ruta_ini): return {}
    parser = configparser.ConfigParser()
    parser.read(ruta_ini, encoding="utf-8")
    if nombre_seccion not in parser: return {}
    items = dict(parser.items(nombre_seccion))
    try: variables = ast.literal_eval(items.get('variables', '[]'))
    except: return {}

    mapa_default = {'1': 'No', '2': 'Sí'}
    for k, v in items.items():
        if k.upper().endswith("_VARS"):
            try: mapa_default = ast.literal_eval(v)
            except: pass; break

    defs = {}
    for var in variables:
        mapa = None
        # Búsqueda exacta y aproximada
        for k, v in items.items():
            if k.upper() == var.upper() or (k.upper() != "VARIABLES" and var.upper() in k.upper()):
                try: 
                    clean = v.replace("’", "'").replace("“", '"').replace("”", '"')
                    mapa = ast.literal_eval(clean); break
                except: pass
        defs[var] = mapa if mapa else mapa_default
    return defs

# --- 2. CARGAR PROMPT DESDE TXT ---
def cargar_prompt(ruta_txt):
    if not os.path.exists(ruta_txt):
        raise FileNotFoundError(f"Falta el archivo: {ruta_txt}")
    with open(ruta_txt, "r", encoding="utf-8") as f:
        return f.read()

# --- 3. CONFIGURACIÓN LLM ---
llm = ChatOllama(model="gemma3:4b", temperature=0, format="json")

# --- 4. FUNCIÓN NUCLEAR DE ANÁLISIS ---
def ejecutar_analisis(titular, texto, seccion, ruta_prompt, ruta_ini="config.ini"):
    # A. Cargar datos
    defs = cargar_definiciones_de_ini(ruta_ini, seccion)
    if not defs: return {"error": "Sin definiciones"}
    
    plantilla = cargar_prompt(ruta_prompt)
    prompt = ChatPromptTemplate.from_template(plantilla)
    chain = prompt | llm
    
    # B. Ejecutar
    # print(f"   > Ejecutando con: {os.path.basename(ruta_prompt)}...")
    try:
        res = chain.invoke({
            "nombre_seccion": seccion,
            "definiciones": json.dumps(defs, indent=2, ensure_ascii=False),
            "titular": titular,
            "texto": texto
        })
        json_raw = json.loads(res.content)
    except Exception as e:
        return {"error": str(e), "raw": res.content if 'res' in locals() else ""}

    # C. Post-Procesado (Etiquetas)
    resultado = {}
    for var, data in json_raw.items():
        if not isinstance(data, dict): continue
        codigo = str(data.get("codigo", "")).strip()
        mapa = defs.get(var, {})
        etiqueta = mapa.get(codigo, "Desconocido")
        
        resultado[var] = {
            "codigo": codigo,
            "etiqueta": etiqueta,
            "evidencia": data.get("evidencia", [])
        }
    return resultado

# --- 5. FUNCIÓN PARA EL TEST A/B/C ---
def comparar_prompts(titular, texto, seccion):
    estrategias = [
        ("Zero-Shot", "prompts/1_zeroshot.txt"),
        ("Definiciones", "prompts/2_definiciones.txt"),
        ("Expertise (Rol)", "prompts/3_expertise.txt")
    ]
    
    print(f"\n{'='*60}")
    print(f"COMPARATIVA DE PROMPTS PARA SECCIÓN: {seccion}")
    print(f"{'='*60}")

    for nombre, ruta in estrategias:
        print(f"\n--- Estrategia: {nombre} ---")
        try:
            res = ejecutar_analisis(titular, texto, seccion, ruta)
            # Imprimimos solo una variable clave para no saturar la pantalla
            # O el JSON completo si prefieres
            print(json.dumps(res, indent=2, ensure_ascii=False))
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {ruta}")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    titular_test = "Estudio internacional evidencia sesgos de género en IA: la investigadora María López critica el olvido"
    texto_test = """
    La investigadora principal María López (Universidad Complutense) presentó ayer en Madrid los resultados.
    "Los datos indican que las referencias a mujeres aparecen con menos frecuencia", explicó López.
    """

    # Ejecutar la comparativa
    comparar_prompts(titular_test, texto_test, "LENGUAJE")