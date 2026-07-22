#!/usr/bin/env python3
"""
Genera skills/<variable>/SKILL.md desde variables.json (Experimento 21).

Independiente del Exp 16 (descartado): reutiliza sólo los helpers de aplanado de
utils.py. Las skills auxiliares (guia_*, verificar_evidencias) se escriben a mano
y NO las toca este script.

Uso:
    cd Experimentos/experiments/experimento_21_agentskills
    python3 generar_skills.py [--json ../../variables.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXPERIMENTOS_DIR = EXP_DIR.parent.parent
sys.path.insert(0, str(EXPERIMENTOS_DIR))

from utils import (  # noqa: E402
    _aplanar_caso_limite,
    _aplanar_definicion,
    _aplanar_ejemplos_negativos,
    _aplanar_ejemplos_positivos,
    _aplanar_metodologia,
    cargar_variables_desde_json,
    obtener_config_variable,
)
from tools import SKILLS_DIR, SKILLS_VARIABLE  # noqa: E402


def _lista_codigos(valores: list) -> str:
    return "\n".join(f"{i + 1} = {v}" for i, v in enumerate(valores))


def _frontera(config: dict) -> str:
    fr = config.get("frontera_con_otras")
    if not fr:
        return ""
    if isinstance(fr, dict):
        return "\n".join(f"- **{k}:** {v}" for k, v in fr.items())
    return str(fr)


def render_skill_md(config: dict, origen: str = "variables.json") -> str:
    nombre = config["nombre"]
    codigo_v = config.get("codigo", "")
    valores = config["valores_posibles"]
    desc = (
        f"Metodología para clasificar «{nombre}» (V{codigo_v}) en un artículo. "
        "Cárgala antes de emitir el veredicto de esta variable."
    )
    L = [
        "---", f"name: {nombre}", f"description: {desc}",
        f"origen_json: {origen}", "---", "",
        f"# Variable {nombre} (V{codigo_v})", "",
        f"> Generada desde `{origen}`. Regenerar con `generar_skills.py --json <ruta>`.", "",
        f"Clasifica **`{nombre}`** a nivel de documento.", "",
        "## Definición", _aplanar_definicion(config.get("definicion", "")), "",
        "## Metodología (ejecutar en orden, sin saltar pasos)",
        _aplanar_metodologia(config.get("metodologia", "")), "",
        "## Códigos posibles", _lista_codigos(valores),
    ]
    ejemplos = config.get("ejemplos_positivos") or config.get("ejemplos") or []
    if ejemplos:
        L += ["", "## Ejemplos donde SÍ aplica", _aplanar_ejemplos_positivos(ejemplos)]
    if config.get("ejemplos_negativos"):
        L += ["", "## Contraejemplos (NO marcar)",
              _aplanar_ejemplos_negativos(config["ejemplos_negativos"])]
    if config.get("caso_limite_documentado"):
        L += ["", "## Caso límite documentado",
              _aplanar_caso_limite(config["caso_limite_documentado"])]
    fr = _frontera(config)
    if fr:
        L += ["", "## Fronteras con otras variables", fr]
    L += [
        "", "## Salida",
        "Tras aplicar la metodología emite: "
        "`FINAL: {\"codigo\": <n>, \"explicacion\": \"...cita pasos...\", \"evidencias\": [literales o []]}`.",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=EXPERIMENTOS_DIR / "variables.json")
    args = ap.parse_args()
    if not args.json.is_file():
        print(f"No se encuentra {args.json}", file=sys.stderr)
        return 1
    datos = cargar_variables_desde_json(str(args.json))
    for nombre, codigo in SKILLS_VARIABLE.items():
        config = obtener_config_variable(datos, codigo)
        destino = SKILLS_DIR / nombre / "SKILL.md"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(render_skill_md(config, args.json.name), encoding="utf-8")
        print(f"  · {destino.relative_to(EXP_DIR)}")
    print("Skills auxiliares (guia_*, verificar_evidencias) NO se tocan (manuales).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
