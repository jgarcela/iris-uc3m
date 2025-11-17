import json
import re
import time
import pandas as pd
from litellm import completion

# === 1. Configuración ===
API_BASE = "http://localhost:11434"
MODELS = [
    "ollama/qwen3:8b",
    "ollama/deepseek-r1:8b",
    "ollama/gemma3:12b",
    "ollama/gemma3:4b",
    "ollama/llama3.1:8b",
    "ollama/mistral:7b"
]

# === 2. Dataset de ejemplo ===
texts = [
    "María López presentó un estudio. \"Hemos encontrado resultados prometedores\" dijo la investigadora. El reportaje fue redactado por Laura García.",
    "El entrenador aseguró que el equipo trabajará duro. El columnista Juan Pérez destacó el esfuerzo de los jugadores.",
    "La presidenta anunció nuevas medidas económicas junto a su gabinete, conformado por hombres y mujeres."
]

# === 3. Variables ===
VARIABLES = [
    'cita_textual_titular', 'genero_nombre_propio_titular', 'genero_periodista',
    'genero_personas_mencionadas', 'nombre_propio_titular', 'personas_mencionadas', 'tema'
]

CITA_TITULAR = {'0': 'No', '1': 'Sí'}
GENERO_NOMBRE_PROPIO_TITULAR = {'1': 'No hay', '2': 'Sí, hombre', '3': 'Sí, mujer', '4': 'Sí, mujer y hombre'}
GENERO_PERIODISTA = {'1': 'Masculino', '2': 'Femenino', '3': 'Mixto', '4': 'Ns/Nc', '5': 'Agencia/otros medios', '6': 'Redacción', '7': 'Corporativo'}
GENERO_PERSONAS_MENCIONADAS = {'1': 'No hay', '2': 'Sí, hombre', '3': 'Sí, mujer', '4': 'Sí, mujer y hombre'}
TEMA = {'1': 'Científica/Investigación', '2': 'Comunicación', '3': 'De farándula o espectáculo', '4': 'Deportiva', '5': 'Economía', '6': 'Educación/cultura', '7': 'Empleo/Trabajo', '8': 'Empresa', '9': 'Judicial', '10': 'Medioambiente', '11': 'Policial', '12': 'Política', '13': 'Salud', '14': 'Social', '15': 'Tecnología', '16': 'Transporte', '17': 'Otros'}

# Validaciones por variable
VALID_CODES = {
    'cita_textual_titular': set(CITA_TITULAR.keys()),
    'genero_nombre_propio_titular': set(GENERO_NOMBRE_PROPIO_TITULAR.keys()),
    'genero_periodista': set(GENERO_PERIODISTA.keys()),
    'genero_personas_mencionadas': set(GENERO_PERSONAS_MENCIONADAS.keys()),
    'nombre_propio_titular': set(['1','2','3','4']),
    'personas_mencionadas': set(['1','2','3','4']),
    'tema': set(TEMA.keys())
}

# === 4. Prompt (corregido, con comillas simples y llaves escapadas) ===
format_instructions = (
    'Devuelve exclusivamente un objeto JSON con las claves: '
    + ', '.join(VARIABLES)
    + ".\nCada clave debe mapear a un objeto con:\n"
    "- 'codigo': el número (string) correspondiente según las tablas, EXACTAMENTE tal como aparece.\n"
    "- 'evidencia': lista (array) de fragmentos textuales extraídos del texto que justifican la etiqueta.\n"
    "Si no hay evidencia, devuelve [].\n\n"
    "Usa *exclusivamente* los siguientes valores:\n"
    f"CITA_TITULAR = {list(CITA_TITULAR.keys())}\n"
    f"GENERO_NOMBRE_PROPIO_TITULAR = {list(GENERO_NOMBRE_PROPIO_TITULAR.keys())}\n"
    f"GENERO_PERIODISTA = {list(GENERO_PERIODISTA.keys())}\n"
    f"GENERO_PERSONAS_MENCIONADAS = {list(GENERO_PERSONAS_MENCIONADAS.keys())}\n"
    f"TEMA = {list(TEMA.keys())}\n\n"
    "Ejemplo (formato exacto esperado; NO añadas texto fuera del JSON):\n"
    '{{\n'
    "  'cita_textual_titular': {{'codigo': '1', 'evidencia': ['María López presentó un estudio']}},\n"
    "  'genero_personas_mencionadas': {{'codigo': '4', 'evidencia': ['María López', 'Juan Pérez']}}\n"
    '}}'
)

PROMPT_TEMPLATE = (
    'Analiza el siguiente texto periodístico y clasifícalo según las variables de CONTENIDO_GENERAL.\n\n'
    'Texto:\n{texto}\n\n'
    "Para cada variable, devuelve un objeto con 'codigo' y 'evidencia'.\n\n"
    + format_instructions
)



# === 5. Utilidades ===
def extract_json(text):
    """Extrae el primer objeto JSON válido de la respuesta del modelo."""
    matches = re.findall(r'\{(?:.|\n)*\}', text)
    for m in matches:
        try:
            return json.loads(m)
        except Exception:
            try:
                cleaned = m.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
                return json.loads(cleaned)
            except Exception:
                continue
    raise ValueError("No se pudo extraer JSON válido del texto de salida.")

def validate_and_normalize(output_obj):
    """Valida estructura y códigos, normaliza resultados."""
    normalized = {}
    errors = []
    for var in VARIABLES:
        if var not in output_obj:
            normalized[var] = {"codigo": None, "evidencia": []}
            errors.append(f"Falta {var}")
            continue
        entry = output_obj[var]
        codigo = str(entry.get("codigo")) if isinstance(entry.get("codigo"), (str, int)) else None
        evidencia = entry.get("evidencia") if isinstance(entry.get("evidencia"), list) else []
        allowed = VALID_CODES.get(var)
        if allowed and codigo not in allowed:
            errors.append(f"Código inválido {codigo} en {var}")
        normalized[var] = {"codigo": codigo, "evidencia": evidencia}
    return normalized, errors

def analyze_text_with_model(text, model):
    """Llama al modelo con LiteLLM y procesa salida."""
    prompt = PROMPT_TEMPLATE.format(texto=text)
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_base=API_BASE
        )
        raw = resp.choices[0].message.content
    except Exception as e:
        return {"error": f"Error al llamar al modelo {model}: {e}"}

    try:
        parsed = extract_json(raw)
    except Exception as e:
        return {"error": f"No JSON válido en {model}", "raw_response": raw}

    normalized, errors = validate_and_normalize(parsed)
    return {"parsed": normalized, "errors": errors, "raw_response": raw}

# === 6. Ejecución de la evaluación ===
results = []
for model in MODELS:
    print(f"\n🔍 Evaluando con modelo: {model}")
    for i, text in enumerate(texts):
        print(f"  → Texto {i+1}/{len(texts)}...")
        out = analyze_text_with_model(text, model)
        row = {
            "modelo": model,
            "texto": text,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": out.get("error"),
            "model_errors": out.get("errors"),
            "raw_response": out.get("raw_response"),
        }
        parsed = out.get("parsed")
        if parsed:
            for var in VARIABLES:
                row[f"{var}__codigo"] = parsed[var]["codigo"]
                row[f"{var}__evidencia"] = json.dumps(parsed[var]["evidencia"], ensure_ascii=False)
        results.append(row)
        time.sleep(0.3)  # pausa para evitar saturar Ollama

# === 7. Guardar resultados ===
df = pd.DataFrame(results)
df.to_csv("evaluacion_multimodelo.csv", index=False, encoding="utf-8-sig")
print("\n✅ Evaluación terminada. Resultados guardados en 'evaluacion_multimodelo.csv'.")
