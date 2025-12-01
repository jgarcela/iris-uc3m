# agents.py
"""
Agentes configurables - Versión Final + FIX LLM LITERALITY.

CORRECCIÓN CRÍTICA:
- El ejemplo JSON del prompt ahora se construye DINÁMICAMENTE con los nombres
  reales de las variables (ej: "tema", "genero...").
- Esto evita que el LLM devuelva claves genéricas como "variable_1" o "variable_A".
"""

import configparser
import ast
import json
import os
import re
import string
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Prefer the newer Ollama adapter if available
try:
    from langchain_ollama import OllamaLLM as OllamaClient  # type: ignore
except Exception:
    try:
        from langchain_community.llms import Ollama as OllamaClient  # type: ignore
    except Exception:
        OllamaClient = None

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

# -----------------------
# DEPENDENCIAS (Padre -> Hijos)
# -----------------------
DEPENDENCY_MAP = {
    "nombre_propio_titular": ["genero_nombre_propio_titular"],
    "personas_mencionadas": ["genero_personas_mencionadas"],
    "declaracion_fuente": ["nombre_fuente"], 
    "nombre_fuente": ["genero_fuente", "tipo_fuente"]
}

# -----------------------
# Config loader
# -----------------------
def load_config(path: str = "config.ini") -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")
    out: Dict[str, Dict[str, Any]] = {}
    
    for section in cp.sections():
        out[section] = {}
        for k, v in cp.items(section):
            v_str = v.strip()
            try:
                val = ast.literal_eval(v_str)
            except Exception:
                try:
                    clean_str = v_str.replace("’", "'").replace("“", '"').replace("”", '"')
                    val = ast.literal_eval(clean_str)
                except Exception:
                    val = v_str
            out[section][k.upper()] = val 
    return out

# -----------------------
# Helper: Detect Negative Key
# -----------------------
def get_negative_key_for_map(mapping: Dict[str, str]) -> str:
    """Busca la clave que significa 'No', 'False', 'Ninguno', '0', etc."""
    NEGATIVE_TERMS = {"no", "no hay", "none", "falso", "false", "no aplica", "ninguno", "nan"}
    for k, v in mapping.items():
        if str(v).lower().strip() in NEGATIVE_TERMS:
            return k
    if '0' in mapping: return '0'
    if '1' in mapping: return '1'
    return '1'

# -----------------------
# Helper: Text Cleaning for Matching
# -----------------------
def clean_string_for_matching(text: str) -> str:
    """Elimina puntuación y espacios, y convierte a minúsculas."""
    if not text: return ""
    return re.sub(r'[\W_]+', '', text.lower())

# -----------------------
# Schema & Parser
# -----------------------
def create_output_parser(variable_names: List[str]) -> Tuple[StructuredOutputParser, str]:
    schemas = [
        ResponseSchema(
            name=var,
            description="Objeto con 'codigo' (string) y 'evidencia' (lista de strings)."
        )
        for var in variable_names
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)
    return parser, parser.get_format_instructions()

# -----------------------
# Prompt Builder (DYNAMIC EXAMPLE)
# -----------------------
def build_prompt_template_for_section(
    section_name: str,
    variable_names: List[str],
    variable_maps: Dict[str, Dict[str, str]], 
    format_instructions: str,
    titular_only_vars: Optional[List[str]] = None,
    custom_instructions: Optional[str] = None
) -> PromptTemplate:
    
    lines: List[str] = []

    if custom_instructions:
        lines.append(custom_instructions)
    else:
        lines.append(f"Actúa como un analista de datos periodísticos riguroso. CLASIFICA las variables de la sección: {section_name}.")
        lines.append("Analiza el texto buscando objetividad y precisión.")
    
    if "FUENTES" in section_name.upper():
        lines.append("")
        lines.append("=== LÓGICA PARA FUENTES ===")
        lines.append("1. ¿Hay cita textual? -> 'declaracion_fuente'.")
        lines.append("2. ¿Quién lo dice? -> 'nombre_fuente'.")
        lines.append("3. Género y Tipo de esa persona -> 'genero_fuente', 'tipo_fuente'.")
        lines.append("NOTA: Si no hay cita, todo es 'No hay'/'No aplica'.")

    if titular_only_vars:
        lines.append("")
        lines.append("ATENCIÓN: Variables EXCLUSIVAS del TITULAR:")
        lines.append(f"{str(titular_only_vars)}")
    
    lines.append("")
    lines.append("=== CÓDIGOS DISPONIBLES ===")
    
    sorted_vars = list(variable_names)
    sorted_vars.sort(key=lambda x: 0 if not x.startswith("genero") else 1)
    
    for var in sorted_vars:
        v_map = variable_maps.get(var, {'1': 'No', '2': 'Sí'})
        json_str = json.dumps(v_map, ensure_ascii=False)
        json_escaped = json_str.replace("{", "{{").replace("}", "}}")
        lines.append(f"- Variable '{var}': {json_escaped}")

    lines.append("")
    lines.append("=== REGLAS ===")
    lines.append("1. **EXHAUSTIVIDAD**: Busca todas las frases relevantes.")
    lines.append("2. **LITERALIDAD**: Copia texto del artículo. NO uses las etiquetas.")
    lines.append("3. **NEGATIVOS**: Si es 'No', evidencia = [].")
    lines.append("4. **CLAVES EXACTAS**: Usa SOLAMENTE los nombres de variables listados arriba.")
    
    lines.append("")
    # --- FIX: Generar ejemplo DINÁMICO con las variables reales ---
    lines.append("JSON de salida (Formato esperado):")
    lines.append("{{")
    
    # Tomamos las primeras variables reales para el ejemplo
    example_vars = variable_names[:3] if len(variable_names) > 3 else variable_names
    
    for i, var in enumerate(example_vars):
        comma = "," if i < len(example_vars) - 1 else ("," if len(variable_names) > len(example_vars) else "")
        # Usamos 4 llaves {{{{ para que salgan 2 {{ en el prompt final (que es lo que Langchain necesita para imprimir 1 { real)
        # Es confuso, pero necesario: PromptTemplate -> Literal text
        lines.append(f'  "{var}": {{{{ "codigo": "...", "evidencia": [...] }}}}{comma}')
    
    if len(variable_names) > len(example_vars):
        lines.append('  ... resto de variables ...')
        
    lines.append("}}")
    # --------------------------------------------------------------
    
    lines += [
        "",
        "--- INICIO ARTÍCULO ---",
        "Titular: {titular}",
        "Texto: {texto}",
        "--- FIN ARTÍCULO ---",
        "",
        "Respuesta JSON:"
    ]

    return PromptTemplate(
        template="\n".join(lines), 
        input_variables=["titular", "texto"], 
        partial_variables={"format_instructions": format_instructions}
    )

# -----------------------
# Utils: JSON Extraction & Text Matching
# -----------------------
def extract_json_from_agent_output(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m: return m.group(1)
    m2 = re.search(r"```[\s\S]*?(\{[\s\S]*?\})[\s\S]*?```", text)
    if m2: return m2.group(1)
    start = text.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[start:i+1]
    return None

def normalize_evidence_list(evidence_raw: Any, source_text: str) -> List[str]:
    if not evidence_raw: return []
    if isinstance(evidence_raw, str):
        try: evidence_raw = json.loads(evidence_raw)
        except: evidence_raw = [evidence_raw]
    if not isinstance(evidence_raw, list): return []

    final_list = []
    source_lower = (source_text or "").lower()
    source_clean = clean_string_for_matching(source_text)
    
    for snippet in evidence_raw:
        if not isinstance(snippet, str) or not snippet.strip(): continue
        clean_snippet = snippet.strip().strip('"').strip("'").strip("...")
        if not clean_snippet: continue

        # 1. Match Exacto
        if clean_snippet in source_text:
            final_list.append(clean_snippet)
            continue
        
        # 2. Match Case Insensitive
        snippet_lower = clean_snippet.lower()
        if snippet_lower in source_lower:
            start_idx = source_lower.find(snippet_lower)
            if start_idx != -1:
                final_list.append(source_text[start_idx : start_idx + len(clean_snippet)])
                continue
        
        # 3. Smart Fuzzy Match
        snippet_super_clean = clean_string_for_matching(clean_snippet)
        if snippet_super_clean and snippet_super_clean in source_clean:
            final_list.append(clean_snippet)
            continue
        
        pass

    return list(set(final_list))

# -----------------------
# AgentAnalyzer
# -----------------------
@dataclass
class AgentAnalyzer:
    name: str
    variables: List[str]
    variable_maps: Dict[str, Dict[str, Any]]
    parser: StructuredOutputParser
    prompt: PromptTemplate
    model: Any
    titular_only_vars: List[str]

    def _invoke_model(self, prompt_text: str) -> str:
        if not self.model: raise RuntimeError("No LLM client.")
        try:
            if hasattr(self.model, "invoke"):
                resp = self.model.invoke(prompt_text)
                return getattr(resp, "content", None) or str(resp)
            if hasattr(self.model, "run"): return self.model.run(prompt_text)
            return self.model(prompt_text)
        except Exception as e: return f"__LLM_INVOCATION_ERROR__:{e}"

    def analyze_text(self, titular: str, texto: str, return_raw: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {"parsed": None, "errors": [], "raw_output": None}
        
        try:
            prompt_value = self.prompt.format_prompt(titular=(titular or ""), texto=(texto or ""))
        except KeyError as e:
            out["errors"].append(f"Prompt error: {e}")
            return out

        agent_response = self._invoke_model(prompt_value.to_string())
        out["raw_output"] = agent_response
        if return_raw: return out

        json_str = extract_json_from_agent_output(agent_response)
        if not json_str:
            out["errors"].append("No JSON found.")
            return out

        try:
            parsed_obj = json.loads(json_str)
            # Auto-unwrap logic: Si la respuesta es {"nombre_variable": {...}}, entramos un nivel.
            if len(parsed_obj) == 1:
                first_key = list(parsed_obj.keys())[0]
                if isinstance(parsed_obj[first_key], dict) and first_key not in self.variables:
                    parsed_obj = parsed_obj[first_key]
        except Exception as e:
            out["errors"].append(f"Invalid JSON: {e}")
            return out

        parsed_clean: Dict[str, Dict[str, Any]] = {}
        NEGATIVE_LABEL_TERMS = {"no", "no hay", "none", "falso", "false", "0", "nan", "no aplica", "ninguno"}
        full_context = f"{titular or ''}\n{texto or ''}"

        # 1. Primera pasada
        for var in self.variables:
            raw_val = parsed_obj.get(var, {})
            # Fix: si el LLM usó claves inventadas, raw_val será {} y el código abajo se encarga.
            if not isinstance(raw_val, dict): raw_val = {"codigo": str(raw_val), "evidencia": []}

            code = str(raw_val.get("codigo", "")).strip().replace('"', '').replace("'", "")
            allowed_map = self.variable_maps.get(var, {})
            neg_key = get_negative_key_for_map(allowed_map)

            if allowed_map and code not in allowed_map:
                found = next((k for k, v in allowed_map.items() if str(v).lower() == code.lower()), None)
                code = found if found else neg_key
            if not code: code = neg_key

            raw_evid = raw_val.get("evidencia")
            scope = titular if var in self.titular_only_vars else full_context
            
            label_text = str(allowed_map.get(code, "")).lower()
            if isinstance(raw_evid, list):
                raw_evid = [e for e in raw_evid if isinstance(e, str) and clean_string_for_matching(e) != clean_string_for_matching(label_text)]

            final_evid = normalize_evidence_list(raw_evid, scope)

            label = str(allowed_map.get(code, "Unknown"))
            is_neg_label = label.lower().strip() in NEGATIVE_LABEL_TERMS

            if is_neg_label:
                final_evid = []
            elif not final_evid:
                 is_categorical = len(allowed_map) > 5
                 if not is_categorical:
                    code = neg_key
                    label = str(allowed_map.get(code, "No"))

            parsed_clean[var] = {
                "codigo": code,
                "evidencia": final_evid,
                "etiqueta": label
            }

        # 2. Dependencias
        for _ in range(2): 
            for parent, children in DEPENDENCY_MAP.items():
                if parent not in parsed_clean: continue
                parent_label = parsed_clean[parent]["etiqueta"]
                
                if parent_label.lower().strip() in NEGATIVE_LABEL_TERMS:
                    for child in children:
                        if child in parsed_clean:
                            child_map = self.variable_maps.get(child, {})
                            child_neg_key = get_negative_key_for_map(child_map)
                            
                            parsed_clean[child]["codigo"] = child_neg_key
                            parsed_clean[child]["etiqueta"] = str(child_map.get(child_neg_key, "No"))
                            parsed_clean[child]["evidencia"] = []

        out["parsed"] = parsed_clean
        return out 

    def analyze_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        return [self.analyze_text(t, x) for t, x in pairs]

# -----------------------
# Factory
# -----------------------
def create_agents_from_config(
    config_path: str = "config.ini", 
    model_kwargs: Optional[Dict[str, Any]] = None,
    prompt_overrides: Optional[Dict[str, str]] = None
) -> Dict[str, AgentAnalyzer]:
    
    cfg = load_config(config_path)
    model_kwargs = model_kwargs or {"model": "gemma2:9b", "temperature": 0.0}
    prompt_overrides = prompt_overrides or {}

    if OllamaClient is None: raise RuntimeError("No Ollama client.")
    base_llm = OllamaClient(**model_kwargs)
    agents = {}

    for section, items in cfg.items():
        variables = items.get("VARIABLES")
        if not isinstance(variables, (list, tuple)): continue
        
        var_maps = {}
        section_default_map = next((v for k, v in items.items() if k.endswith("_VARS") and isinstance(v, dict)), None)

        for var in variables:
            var_upper = var.upper()
            if var_upper in items and isinstance(items[var_upper], dict):
                var_maps[var] = items[var_upper]
            elif section_default_map:
                var_maps[var] = section_default_map
            else:
                var_maps[var] = {'1': 'No', '2': 'Sí'}

        parser, fmt = create_output_parser(list(variables))
        titular_only = [v for v in variables if "titular" in v.lower()]
        custom_instr = prompt_overrides.get(section, None)

        prompt = build_prompt_template_for_section(
            section_name=section,
            variable_names=list(variables),
            variable_maps=var_maps,
            format_instructions=fmt,
            titular_only_vars=titular_only,
            custom_instructions=custom_instr
        )

        agents[section] = AgentAnalyzer(
            name=section, variables=list(variables), variable_maps=var_maps,
            parser=parser, prompt=prompt, model=base_llm, titular_only_vars=titular_only,
        )
    return agents