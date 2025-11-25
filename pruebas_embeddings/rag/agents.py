# agents.py
"""
Agentes configurables desde config.ini — versión SIN tools / SIN ReAct.
Importar desde notebook:
    from agents import create_agents_from_config
    agents = create_agents_from_config("config.ini")
"""

import configparser
import ast
import json
import os
import re
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
                    val = ast.literal_eval(v_str.replace("’", "'").replace("“", '"').replace("”", '"'))
                except Exception:
                    val = v_str
            out[section][k] = val
    return out

# -----------------------
# Schema & parser builders
# -----------------------
def make_response_schemas(variable_names: List[str]) -> List[ResponseSchema]:
    return [
        ResponseSchema(
            name=var,
            description="Objeto con 'evidencia' (lista de fragmentos textuales que justifican el valor)."
        )
        for var in variable_names
    ]

def create_output_parser(variable_names: List[str]) -> Tuple[StructuredOutputParser, str]:
    schemas = make_response_schemas(variable_names)
    parser = StructuredOutputParser.from_response_schemas(schemas)
    return parser, parser.get_format_instructions()

# -----------------------
# Prompt builder — NO tools, agent must return final JSON directly
# -----------------------
def build_prompt_template_for_section(
    section_name: str,
    variable_names: List[str],
    allowed_maps: Dict[str, Any],
    format_instructions: str,
    titular_only_vars: Optional[List[str]] = None,
) -> PromptTemplate:
    lines: List[str] = []
    lines.append(f"Analiza el siguiente artículo y extrae las EVIDENCIAS necesarias para clasificar las variables de la sección: {section_name}.")
    if titular_only_vars:
        lines.append("")
        lines.append("Variables que deben valorarse exclusivamente desde el TITULAR (no usar el cuerpo para estas variables):")
        lines.append(str(titular_only_vars))
    lines.append("")
    lines.append("NO se usarán herramientas externas. Devuelve al final UN ÚNICO OBJETO JSON con las variables y sus evidencias.")
    lines.append("La salida NO debe contener 'Thought', 'Action', 'Observation' ni texto explicativo — SOLO el JSON.")
    lines += [
        "",
        "Titular:",
        "{titular}",
        "",
        "Texto de la noticia:",
        "{texto}",
        "",
        "Devuelve SOLO el JSON final con la forma (ejemplo):",
        ""
    ]

    # Example JSON with escaped braces
    example_lines = ["{{"]
    for i, var in enumerate(variable_names):
        comma = "," if i < len(variable_names) - 1 else ""
        example_lines.append(f'  "{var}": {{{{ "evidencia": ["..."] }}}}{comma}')
    example_lines.append("}}")
    lines += example_lines

    lines += [
        "",
        "RECUERDA: la última respuesta debe ser exactamente el JSON (ningún 'Thought', 'Action', 'Observation' ni texto explicativo).",
        "",
        "{format_instructions}"
    ]

    template = "\n".join(lines)
    return PromptTemplate(template=template, input_variables=["titular", "texto"], partial_variables={"format_instructions": format_instructions})

# -----------------------
# Utilities
# -----------------------
def _normalize_evidence_field(evid_any):
    if evid_any is None:
        return []
    if isinstance(evid_any, list):
        return evid_any
    if isinstance(evid_any, str):
        try:
            parsed = json.loads(evid_any)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [evid_any]
    return []

def evidence_in_text(evidence: List[str], source_text: str) -> List[str]:
    if not evidence:
        return []
    src = (source_text or "").lower()
    return [e for e in evidence if e and e.strip() and e.lower() in src]

# -----------------------
# Simple name extraction & heuristics (no external APIs)
# -----------------------
def extract_name_candidates(evidence: List[str]) -> List[str]:
    candidates = []
    name_re = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)\b")
    for ev in evidence or []:
        if not ev:
            continue
        for m in name_re.findall(ev):
            tok = m.strip()
            if tok and tok not in candidates:
                candidates.append(tok)
            if len(candidates) >= 6:
                break
        if len(candidates) >= 6:
            break
    return candidates

def infer_gender_code_heuristic(evidence: List[str]) -> str:
    """
    Heurística simple basada en pronombres y cargos:
      '1' = No hay info
      '2' = Masculino
      '3' = Femenino
      '4' = Mixto
    """
    txt = " ".join(evidence or []).lower()
    female_markers = [" ella ", "ella,", "investigadora", "coautora", "autora", "profesora", "sra.", "señora", "la investigadora", "la profesora"]
    male_markers = [" él ", "el investigador", "coautor", "autor", "profesor", "sr.", "señor", "el investigador"]
    f = any(m in txt for m in female_markers)
    m = any(m in txt for m in male_markers)

    if f and m:
        return "4"
    if f:
        return "3"
    if m:
        return "2"

    # fallback: if names exist, try naive last-letter heuristic on first names (not reliable)
    names = extract_name_candidates(evidence)
    if names:
        male = 0
        female = 0
        for n in names:
            n0 = n.split()[0].lower()
            if n0.endswith("a") or n0.endswith("á") or n0.endswith("e"):  # generous guess
                female += 1
            else:
                male += 1
        if male > 0 and female > 0:
            return "4"
        if female > male:
            return "3"
        if male > female:
            return "2"

    return "1"

# -----------------------
# Extract JSON helper
# -----------------------
def extract_json_from_agent_output(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"```[\s\S]*?(\{[\s\S]*?\})[\s\S]*?```", text)
    if m2:
        return m2.group(1)
    m3 = re.search(r"Final Answer\s*[:\-]*\s*(\{[\s\S]*?\})", text, re.IGNORECASE)
    if m3:
        return m3.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

# -----------------------
# AgentAnalyzer (no tools)
# -----------------------
@dataclass
class AgentAnalyzer:
    name: str
    variables: List[str]
    allowed_maps: Dict[str, Any]
    parser: StructuredOutputParser
    prompt: PromptTemplate
    model: Any  # Ollama client
    titular_only_vars: List[str]

    def _invoke_model(self, prompt_text: str) -> str:
        """
        Robust invocation of the LLM client. Tries .invoke, .run, or call.
        Returns the raw textual response.
        """
        if not self.model:
            raise RuntimeError("No LLM client available on this agent.")
        # Try various invocation styles
        try:
            # prefer .invoke if available
            if hasattr(self.model, "invoke"):
                resp = self.model.invoke(prompt_text)
                content = getattr(resp, "content", None) or str(resp)
                return content
            # fallback to .run
            if hasattr(self.model, "run"):
                return self.model.run(prompt_text)
            # last fallback: call directly
            return self.model(prompt_text)
        except Exception as e:
            # wrap exception as string to be consumed by analyze_text
            return f"__LLM_INVOCATION_ERROR__:{e}"

    def analyze_text(self, titular: str, texto: str, return_raw: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el LLM (sin tools), extrae el JSON con evidencias que devuelva el modelo,
        normaliza evidencias (aplica enforcement titular-only) y devuelve para cada variable:
            { "codigo": "<codigo>", "evidencia": [ ... ] }
        También rellena out['errors'] y out['raw_output'].
        """
        out: Dict[str, Any] = {"parsed": None, "errors": [], "raw_output": None}

        # 1) Formatear prompt
        prompt_value = self.prompt.format_prompt(titular=(titular or ""), texto=(texto or ""))

        # 2) Invocar LLM de forma robusta
        agent_response = self._invoke_model(prompt_value.to_string())
        out["raw_output"] = agent_response

        if return_raw:
            return out

        # 3) Extraer el JSON (el LLM debe devolver evidencias por variable)
        json_str = extract_json_from_agent_output(agent_response)
        if not json_str:
            out["errors"].append("No JSON found in model output.")
            return out

        try:
            parsed_obj = json.loads(json_str)
        except Exception as e:
            out["errors"].append(f"Invalid JSON extracted: {e}")
            out["errors"].append(f"Extracted string: {json_str}")
            return out

        # 4) Normalizar evidencias y aplicar enforcement de titular-only
        parsed_with_codes: Dict[str, Dict[str, Any]] = {}
        for var in self.variables:
            raw_val = parsed_obj.get(var, {}) or {}
            evid = _normalize_evidence_field(raw_val.get("evidencia"))
            if var in self.titular_only_vars:
                evid = evidence_in_text(evid, titular or "")

            # 5) Inferir código según tipo de variable
            if "genero" in var.lower() or "género" in var.lower():
                codigo = infer_gender_code_heuristic(evid)
            else:
                # regla binaria: 2 = Sí (hay evidencia), 1 = No (vacío)
                codigo = "2" if evid else "1"

            parsed_with_codes[var] = {
                "codigo": str(codigo),
                "evidencia": evid
            }

        out["parsed"] = parsed_with_codes
        return out 


    def analyze_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        return [self.analyze_text(t, x) for t, x in pairs]

# -----------------------
# Factory: create agents from config.ini (no tools)
# -----------------------
def create_agents_from_config(
    config_path: str = "config.ini",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, AgentAnalyzer]:
    cfg = load_config(config_path)
    model_kwargs = model_kwargs or {"model": "gemma3:4b", "base_url": "http://localhost:11434"}

    if OllamaClient is None:
        raise RuntimeError("No Ollama client available. Install langchain-ollama or langchain_community.llms.")

    base_llm = OllamaClient(**model_kwargs)

    agents: Dict[str, AgentAnalyzer] = {}
    for section, items in cfg.items():
        variables = items.get("variables")
        if not isinstance(variables, (list, tuple)):
            continue
        parser, fmt = create_output_parser(list(variables))
        allowed_maps = {k.upper(): v for k, v in items.items() if isinstance(v, dict)}
        titular_only = [v for v in variables if "titular" in v.lower()]

        prompt = build_prompt_template_for_section(
            section_name=section,
            variable_names=list(variables),
            allowed_maps=allowed_maps,
            format_instructions=fmt,
            titular_only_vars=titular_only,
        )

        agents[section] = AgentAnalyzer(
            name=section,
            variables=list(variables),
            allowed_maps=allowed_maps,
            parser=parser,
            prompt=prompt,
            model=base_llm,
            titular_only_vars=titular_only,
        )

    return agents

