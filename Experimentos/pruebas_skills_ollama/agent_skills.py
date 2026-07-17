"""
Single LangChain agent (Ollama) with on-demand skills from `skills/<id>/SKILL.md`.

Setup
-----
1. Ollama running locally (`ollama serve`) and the model pulled, e.g. `ollama pull gemma4:e4b`.
2. From this directory::

    python3 -m venv .venv
    source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt

Run
---
::

    python agent_skills.py "Resume en tres frases: <tu texto>"
    python agent_skills.py --model gemma3:4b "Lista los skills disponibles y dime cuál usar para JSON"

Environment: ``OLLAMA_MODEL`` overrides the default model if ``--model`` is omitted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

_SKILLS_ROOT = Path(__file__).resolve().parent / "skills"


def _allowed_skill_ids() -> list[str]:
    if not _SKILLS_ROOT.is_dir():
        return []
    out: list[str] = []
    for p in _SKILLS_ROOT.iterdir():
        if p.is_dir() and (p / "SKILL.md").is_file():
            out.append(p.name)
    return sorted(out)


@tool
def list_skills() -> str:
    """Lista los identificadores de skills disponibles (carpetas bajo skills/ con SKILL.md). Úsalo primero si no sabes qué skill existe."""
    ids = _allowed_skill_ids()
    if not ids:
        return "(ningún skill: añade carpetas skills/<nombre>/SKILL.md)"
    return "Skills disponibles (skill_id):\n" + "\n".join(f"- {s}" for s in ids)


@tool
def read_skill(skill_id: str) -> str:
    """Lee el contenido completo de skills/<skill_id>/SKILL.md. skill_id debe ser uno de los listados por list_skills."""
    allowed = _allowed_skill_ids()
    if skill_id not in allowed:
        return (
            "Error: skill_id no permitido. Usa list_skills. "
            f"Permitidos: {', '.join(allowed) if allowed else '(ninguno)'}"
        )
    path = _SKILLS_ROOT / skill_id / "SKILL.md"
    return path.read_text(encoding="utf-8")


SYSTEM_PROMPT = """Eres un asistente con acceso a habilidades documentadas en archivos SKILL.md del proyecto.

Reglas:
- Las instrucciones detalladas no están en este mensaje: están en disco.
- Para tareas que encajen con un skill, llama primero a list_skills si no sabes los ids, luego read_skill con el id adecuado y sigue ese SKILL.md.
- No inventes el contenido de un SKILL.md sin haberlo leído con read_skill.
- Si ningún skill aplica, responde con buen criterio general."""


def build_executor(model: str, temperature: float, verbose: bool) -> AgentExecutor:
    llm = ChatOllama(model=model, temperature=temperature)
    tools = [list_skills, read_skill]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=15,
        handle_parsing_errors=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente Ollama con skills SKILL.md bajo skills/")
    parser.add_argument(
        "task",
        nargs="*",
        help="Instrucción o pregunta para el agente (une todos los argumentos)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:e4b"),
        help="Nombre del modelo en Ollama (por defecto: env OLLAMA_MODEL o gemma4:e4b)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    task = " ".join(args.task).strip()
    if not task:
        parser.error("Indica una tarea como argumentos, p. ej. python agent_skills.py \"Resume: ...\"")

    executor = build_executor(args.model, args.temperature, args.verbose)
    result = executor.invoke({"input": task})
    print(result.get("output", result))


if __name__ == "__main__":
    main()
