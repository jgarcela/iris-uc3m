#!/usr/bin/env python3
"""
Genera skills-resumen de las guías de lenguaje (Experimento 21).

Para cada guía de `Experimentos/methodology/` relevante al sexismo lingüístico:
  1. recupera pasajes representativos con el índice TF-IDF de guias.py,
  2. los resume con un LLM a un SKILL.md auxiliar (reglas + ejemplos + recomendaciones),
  3. lo escribe en skills/<slug>/SKILL.md con frontmatter (name+description).

Estos resúmenes son cargables por el agente vía read_skill (orientación rápida),
complementarios a la tool CONSULTAR_GUIA (recuperación literal en vivo).

Uso:
    python3 generar_resumenes_guias.py [--modelo gpt-4o-mini]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXPERIMENTOS_DIR = EXP_DIR.parent.parent
for p in (str(EXPERIMENTOS_DIR), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import consultar_ollama  # noqa: E402
import guias  # noqa: E402
from tools import SKILLS_DIR  # noqa: E402

# Guías centradas en lenguaje no sexista (las de violencia/infancia se excluyen).
GUIAS_RESUMIR = [
    ("CSD_GUIA_PER_DEPOR.md", "guia_resumen_csd_deporte",
     "Resumen de la guía CSD de lenguaje no sexista en periodismo deportivo (masculino genérico, feminización, denominación de deportistas)."),
    ("Guia lenguaje inclusivo (1).md", "guia_resumen_inclusivo_justicia",
     "Resumen de la guía del Ministerio de Justicia para un lenguaje más inclusivo e igualitario."),
    ("Guiaslenguajenosexista_.md", "guia_resumen_no_sexista",
     "Resumen de guía general de lenguaje no sexista (recomendaciones y alternativas al masculino genérico)."),
    ("Guia_orientativa_para_el_uso_igualitario_del_lengu_ACTUALIZADA.md", "guia_resumen_uso_igualitario",
     "Resumen de guía orientativa para el uso igualitario del lenguaje (cargos, profesiones, tratamientos)."),
    ("Guia-Igualdad-castellano.md", "guia_resumen_igualdad",
     "Resumen de guía de igualdad en el uso del lenguaje en castellano."),
]

# Consultas que definen qué es relevante recuperar de cada guía para nuestras 5 variables.
QUERIES = [
    "masculino generico alternativas desdoblamiento",
    "feminizacion cargos profesiones presidenta jueza",
    "denominacion mujeres rol familiar aspecto fisico",
    "asimetria tratamiento mujeres hombres nombre apellido",
    "recomendaciones lenguaje no sexista periodistas ejemplos",
]

MAX_CTX_CHARS = 6000


def _pasajes_de_guia(fuente: str) -> str:
    """Recupera pasajes representativos de una guía concreta (dedup, cap de chars)."""
    vistos: set[str] = set()
    bloques: list[str] = []
    total = 0
    for q in QUERIES:
        for r in guias.buscar(q, k=3):
            if r["fuente"] != fuente:
                continue
            clave = r["texto"][:80]
            if clave in vistos:
                continue
            vistos.add(clave)
            bloques.append(f"[{r['titulo']}] {r['texto']}")
            total += len(r["texto"])
            if total >= MAX_CTX_CHARS:
                return "\n\n".join(bloques)
    return "\n\n".join(bloques)


PROMPT_RESUMEN = """Eres experta en lenguaje no sexista. A partir de estos EXTRACTOS de una guía real,
redacta un resumen operativo en español (markdown, 250-450 palabras) con esta estructura:

## Enfoque de la guía
(1-2 frases sobre qué defiende)

## Reglas clave
(4-8 reglas concretas de lenguaje no sexista aplicables al analizar un texto periodístico)

## Ejemplos
(2-4 pares «uso sexista → alternativa», tomados o inspirados en los extractos)

## Al clasificar
(2-3 recomendaciones de cómo usar esta guía para decidir si un texto tiene sexismo lingüístico)

No inventes datos que contradigan los extractos. No incluyas portada, créditos ni bibliografía.
Devuelve SOLO el markdown, sin preámbulo.

=== EXTRACTOS ===
{contexto}
=== FIN ===
"""


def render_skill(nombre: str, descripcion: str, cuerpo: str) -> str:
    return (f"---\nname: {nombre}\ndescription: {descripcion}\n---\n\n"
            f"# {descripcion}\n\n"
            "> Resumen auto-generado de una guía real de `Experimentos/methodology/` "
            "(revisar antes de citar). Para texto literal usa la acción `CONSULTAR_GUIA`.\n\n"
            f"{cuerpo.strip()}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", default="gpt-4o-mini")
    ap.add_argument("--solo", help="Generar solo esta fuente (nombre de fichero)")
    args = ap.parse_args()

    for fuente, slug, desc in GUIAS_RESUMIR:
        if args.solo and args.solo != fuente:
            continue
        ctx = _pasajes_de_guia(fuente)
        if not ctx.strip():
            print(f"⚠️  Sin pasajes para {fuente}, salto.")
            continue
        prompt = PROMPT_RESUMEN.format(contexto=ctx[:MAX_CTX_CHARS])
        cuerpo = consultar_ollama(prompt, modelo=args.modelo, temperature=0.2).strip()
        destino = SKILLS_DIR / slug / "SKILL.md"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(render_skill(slug, desc, cuerpo), encoding="utf-8")
        print(f"  · {destino.relative_to(EXP_DIR)}  ({len(cuerpo)} chars)")
    print("Recuerda: añade los slugs a SKILLS_AUXILIARES en tools.py si quieres que el agente los vea.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
