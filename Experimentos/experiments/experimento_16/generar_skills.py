#!/usr/bin/env python3
"""
Genera skills/<variable>/SKILL.md desde variables.json.

La metodología operativa del JSON se inspira en las guías de
Experimentos/pruebas_skills_ollama/methodology/ (referencia bibliográfica).

Uso:
    cd Experimentos/experiments/experimento_16
    python3 generar_skills.py
    python3 generar_skills.py --json ../../variables.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTO_DIR = Path(__file__).resolve().parent
EXPERIMENTOS_DIR = EXPERIMENTO_DIR.parent.parent
sys.path.insert(0, str(EXPERIMENTOS_DIR))
sys.path.insert(0, str(EXPERIMENTO_DIR))

from skill_content import generar_skills  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=EXPERIMENTOS_DIR / "variables.json",
        help="Ruta a variables.json",
    )
    args = parser.parse_args()
    if not args.json.is_file():
        print(f"No se encuentra {args.json}", file=sys.stderr)
        return 1
    paths = generar_skills(str(args.json))
    print(f"Generadas {len(paths)} skills:")
    for p in paths:
        print(f"  · {p.relative_to(EXPERIMENTO_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
