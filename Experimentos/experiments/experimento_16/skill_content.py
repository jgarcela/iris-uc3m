"""Contenido de skills por variable (generación y lectura). Sin dependencia de ollama."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

EXPERIMENTO_DIR = Path(__file__).resolve().parent
SKILLS_DIR = EXPERIMENTO_DIR / "skills"

VARIABLES_EXP16_CODIGOS = ("25", "26", "30", "33", "35")

INSPIRACION_METODOLOGICA = (
    "La metodología operativa de esta skill está codificada en `variables.json` y se "
    "**inspira** en las guías de `Experimentos/pruebas_skills_ollama/methodology/` "
    "(tesis Sainz de Baranda, guías de lenguaje inclusivo y no sexista, CSD, etc.). "
    "Esas guías son referencia bibliográfica; en clasificación se ejecutan los pasos "
    "declarados en la sección Metodología de esta skill."
)


def _cargar_variables(ruta: str) -> Any:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _obtener_config(datos: Any, codigo: str) -> dict:
    if isinstance(datos, dict) and codigo in datos:
        return datos[codigo]
    if isinstance(datos, list):
        for v in datos:
            if v.get("codigo") == codigo:
                return v
    raise ValueError(f"Variable {codigo!r} no encontrada")


def _aplanar_definicion(definicion: Union[str, dict]) -> str:
    if isinstance(definicion, str):
        return definicion
    if not isinstance(definicion, dict):
        return str(definicion)
    partes = []
    if "concepto" in definicion:
        partes.append(definicion["concepto"])
    if "criterio_operativo" in definicion:
        partes.append("")
        partes.append("**Criterio operativo:** " + definicion["criterio_operativo"])
    for clave, valor in definicion.items():
        if clave in ("concepto", "criterio_operativo"):
            continue
        partes.extend(["", f"**{clave.replace('_', ' ').capitalize()}:** {valor}"])
    return "\n".join(partes).strip()


def _aplanar_metodologia(metodologia: Union[str, dict]) -> str:
    if isinstance(metodologia, str):
        return metodologia
    if not isinstance(metodologia, dict):
        return str(metodologia)
    lineas = []
    for clave, valor in metodologia.items():
        partes = clave.split("_", 2)
        if len(partes) == 3 and partes[0] == "paso":
            etiqueta = f"Paso {partes[1]} — {partes[2].replace('_', ' ').capitalize()}"
        else:
            etiqueta = clave.replace("_", " ").capitalize()
        lineas.extend([f"**{etiqueta}**", valor, ""])
    return "\n".join(lineas).strip()


def _aplanar_ejemplos_positivos(ejemplos: Union[str, list]) -> str:
    if isinstance(ejemplos, str):
        return ejemplos
    lineas = []
    for i, ej in enumerate(ejemplos, 1):
        if isinstance(ej, str):
            lineas.append(f"{i}. {ej}")
            continue
        texto = ej.get("texto", "")
        razon = ej.get("razon", "") or ej.get("regla_inversion", "")
        etq = ej.get("etiqueta")
        cab = f"{i}. «{texto}»" + (f" → etiqueta {etq}" if etq is not None else "")
        lineas.append(cab)
        if razon:
            lineas.append(f"   Razón: {razon}")
    return "\n".join(lineas)


def _aplanar_ejemplos_negativos(ejemplos: list) -> str:
    if not ejemplos:
        return "(Sin contraejemplos documentados.)"
    lineas = []
    for i, ej in enumerate(ejemplos, 1):
        lineas.append(f"{i}. «{ej.get('texto', '')}»")
        if ej.get("razon_no_aplica"):
            lineas.append(f"   No aplica: {ej['razon_no_aplica']}")
    return "\n".join(lineas)


def _lista_codigos(valores: list) -> str:
    return "\n".join(f"{i + 1} = {v}" for i, v in enumerate(valores))


def render_skill_md(config: dict[str, Any]) -> str:
    """Genera el contenido completo de skills/<nombre>/SKILL.md."""
    nombre = config["nombre"]
    codigo_v = config.get("codigo", "")
    valores = config["valores_posibles"]
    descripcion = (
        f"Clasifica la variable {nombre} (V{codigo_v}) en artículos periodísticos. "
        "Ejecuta la metodología paso a paso antes de asignar codigo."
    )
    lineas = [
        "---",
        f"name: {nombre}",
        f"description: {descripcion}",
        "---",
        "",
        f"# Variable {nombre} (V{codigo_v})",
        "",
        f"Skill del Experimento 16. Clasifica **`{nombre}`** a nivel de documento.",
        "",
        "## Inspiración metodológica",
        INSPIRACION_METODOLOGICA,
        "",
        "## Definición",
        _aplanar_definicion(config.get("definicion", "")),
        "",
        "## Metodología (ejecutar en orden, sin saltar pasos)",
        _aplanar_metodologia(config.get("metodologia", "")),
        "",
        "## Códigos posibles",
        _lista_codigos(valores),
    ]
    ejemplos = config.get("ejemplos_positivos") or config.get("ejemplos") or []
    if ejemplos:
        lineas.extend(["", "## Ejemplos donde SÍ aplica", _aplanar_ejemplos_positivos(ejemplos)])
    if config.get("ejemplos_negativos"):
        lineas.extend([
            "",
            "## Contraejemplos (NO marcar)",
            _aplanar_ejemplos_negativos(config["ejemplos_negativos"]),
        ])
    if config.get("caso_limite_documentado"):
        caso = config["caso_limite_documentado"]
        if isinstance(caso, dict):
            lineas.extend([
                "",
                "## Caso límite documentado",
                f"Texto: «{caso.get('texto', '')}»",
                f"Decisión: {caso.get('decision', '')}",
                f"Explicación: {caso.get('explicacion', '')}",
            ])
    if config.get("frontera_con_otras"):
        lineas.extend(["", "## Fronteras con otras variables", str(config["frontera_con_otras"])])
    lineas.extend([
        "",
        "## Salida esperada para esta variable",
        "",
        "Tras aplicar la metodología, aporta en el JSON unificado:",
        "",
        f"- Clave: `{nombre}`",
        "- Campos: `codigo`, `explicacion` (citando pasos aplicados), `evidencias` (literales o `[]` si codigo=1)",
    ])
    return "\n".join(lineas) + "\n"


def ruta_skill(nombre: str) -> Path:
    return SKILLS_DIR / nombre / "SKILL.md"


def generar_skills(ruta_json: str) -> list[Path]:
    """Escribe skills/<nombre>/SKILL.md para cada variable del experimento 16."""
    datos = _cargar_variables(ruta_json)
    escritos: list[Path] = []
    for codigo in VARIABLES_EXP16_CODIGOS:
        config = _obtener_config(datos, codigo)
        nombre = config["nombre"]
        destino = ruta_skill(nombre)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(render_skill_md(config), encoding="utf-8")
        escritos.append(destino)
    return escritos


def cargar_skill_variable(nombre: str) -> str:
    """Lee skills/<nombre>/SKILL.md y lo envuelve como bloque anexo al system prompt."""
    path = ruta_skill(nombre)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta: python3 generar_skills.py"
        )
    contenido = path.read_text(encoding="utf-8").strip()
    return f"--- SKILL: {nombre} ---\n\n{contenido}"
