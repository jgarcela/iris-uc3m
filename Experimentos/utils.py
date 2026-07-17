
import json
import configparser
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Any, List, Dict, Union
import re
import ollama
from pathlib import Path

# Ruta al config.ini (junto a este utils.py) con la sección [API-KEYS].
_CONFIG_PATH = Path(__file__).resolve().parent / "config.ini"


def _get_api_key(nombre: str) -> Optional[str]:
    """
    Devuelve la API key `nombre` (p. ej. 'openai_api_key') desde config.ini
    [API-KEYS]. Si no existe o está vacía, devuelve None y deja que el SDK
    use la variable de entorno correspondiente como fallback.
    """
    try:
        parser = configparser.ConfigParser()
        if not parser.read(_CONFIG_PATH, encoding="utf-8"):
            return None
        valor = parser.get("API-KEYS", nombre, fallback="").strip()
        return valor or None
    except Exception:
        return None


def _con_reintentos(fn, *, intentos: int = 5, base: float = 1.0, proveedor: str = ""):
    """
    Ejecuta `fn()` reintentando ante errores transitorios (rate limit / 5xx /
    timeouts) con backoff exponencial + jitter. Pensado para saturar las APIs con
    muchos workers sin que un 429 tumbe el artículo.

    - `intentos`: nº máximo de intentos totales.
    - `base`: segundos base del backoff (espera ~ base * 2**n + jitter).
    Reeleva la excepción si se agotan los intentos.
    """
    import time
    import random

    for n in range(intentos):
        try:
            return fn()
        except Exception as e:
            # ¿Es un error que merece reintento? (rate limit, sobrecarga, red)
            nombre_err = type(e).__name__.lower()
            texto_err = str(e).lower()
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            transitorio = (
                status in (408, 409, 425, 429, 500, 502, 503, 504)
                or any(t in nombre_err for t in ("ratelimit", "timeout", "connection", "overloaded", "apierror", "serviceunavailable", "internalserver"))
                or any(t in texto_err for t in ("rate limit", "overloaded", "timeout", "temporarily", "try again", "503", "429"))
            )
            if not transitorio or n == intentos - 1:
                raise
            espera = base * (2 ** n) + random.uniform(0, base)
            print(f"[reintento {n+1}/{intentos-1}] {proveedor} {type(e).__name__}: espero {espera:.1f}s")
            time.sleep(espera)

CLAUDE_API_MODEL_ID = "claude-haiku-4-5-20251001"

# IDs que deben ir a la API de Anthropic (no a Ollama). Ampliar aquí si añades más variantes Claude vía API.
CLAUDE_API_MODEL_IDS = frozenset(
    {
        CLAUDE_API_MODEL_ID,
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-7",
    }
)

# Modelos de razonamiento de OpenAI (familia GPT-5 / o*): NO aceptan `temperature`
# distinto del valor por defecto y usan `max_completion_tokens`.
OPENAI_REASONING_MODEL_IDS = frozenset(
    {
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    }
)

# IDs que deben ir a la API de OpenAI (no a Ollama). Ampliar aquí si añades más variantes.
OPENAI_API_MODEL_IDS = frozenset(
    {
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    }
) | OPENAI_REASONING_MODEL_IDS

# IDs que deben ir a la API de Google Gemini (no a Ollama). Ampliar aquí si añades más variantes.
GEMINI_API_MODEL_IDS = frozenset(
    {
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    }
)

# =====================================================================================
# 0. Ollama / Anthropic (Claude API)
# =====================================================================================
# def consultar_ollama(prompt: str, modelo: str = "gemma3:4b") -> str:
#     """
#     Función genérica para enviar cualquier prompt a Ollama.
#     Devuelve la respuesta del modelo como texto limpio.
#     """
#     try:
#         response = ollama.chat(model=modelo, messages=[
#             {
#                 'role': 'user',
#                 'content': prompt,
#             },
#         ])
#         return response['message']['content'].strip()
    
#     except Exception as e:
#         print(f"Error conectando con el modelo {modelo}: {e}")
#         return ""



def consultar_ollama(prompt: str, modelo: str = "gemma3:4b", temperature: float = 0) -> str:
    """
    Envía el prompt a Ollama salvo que `modelo` sea una API externa:
      - Claude (Anthropic): usa `ANTHROPIC_API_KEY` en el entorno.
      - GPT (OpenAI): usa `OPENAI_API_KEY` en el entorno.
      - Gemini (Google): usa `GEMINI_API_KEY` (o `GOOGLE_API_KEY`) en el entorno.
    Los modelos locales (p. ej. `gemma3n:e4b`) caen en la rama Ollama por defecto.
    Devuelve la respuesta del modelo como texto limpio.
    """
    if modelo in GEMINI_API_MODEL_IDS:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            print(f"Para usar {modelo} instala google-genai: pip install google-genai ({e})")
            return ""
        try:
            client = genai.Client(api_key=_get_api_key("gemini_api_key"))
            response = _con_reintentos(
                lambda: client.models.generate_content(
                    model=modelo,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature),
                ),
                proveedor="gemini",
            )
            return (response.text or "").strip()
        except Exception as e:
            print(f"Error conectando con el modelo {modelo}: {e}")
            return ""

    if modelo in OPENAI_API_MODEL_IDS:
        try:
            from openai import OpenAI
        except ImportError as e:
            print(f"Para usar {modelo} instala openai: pip install openai ({e})")
            return ""
        try:
            client = OpenAI(api_key=_get_api_key("openai_api_key"))
            kwargs = {
                "model": modelo,
                "messages": [{"role": "user", "content": prompt}],
            }
            # Los modelos de razonamiento (GPT-5) solo admiten la temperatura por
            # defecto; en su lugar controlamos el coste/latencia con reasoning_effort.
            if modelo in OPENAI_REASONING_MODEL_IDS:
                kwargs["reasoning_effort"] = "low"
            else:
                kwargs["temperature"] = temperature
            response = _con_reintentos(
                lambda: client.chat.completions.create(**kwargs),
                proveedor="openai",
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"Error conectando con el modelo {modelo}: {e}")
            return ""

    if modelo in CLAUDE_API_MODEL_IDS:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            print(f"Para usar {modelo} instala anthropic: pip install anthropic ({e})")
            return ""
        try:
            client = Anthropic(api_key=_get_api_key("anthropic_api_key"))
            response = _con_reintentos(
                lambda: client.messages.create(
                    model=modelo,
                    max_tokens=8192,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                ),
                proveedor="anthropic",
            )
            texto = "".join(
                bloque.text for bloque in response.content if bloque.type == "text"
            )
            return texto.strip()
        except Exception as e:
            print(f"Error conectando con el modelo {modelo}: {e}")
            return ""

    try:
        response = ollama.chat(
            model=modelo,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature},
        )
        return response['message']['content'].strip()

    except Exception as e:
        print(f"Error conectando con el modelo {modelo}: {e}")
        return ""


# =====================================================================================
# 7a. Nombre Propio Titular
# =====================================================================================
class NombresDetectados(BaseModel):
    # Lista de cadenas con los nombres extraídos
    nombres: List[str] = Field(default_factory=list, description="Lista de nombres propios detectados")
    # Lista de enteros con los códigos correspondientes
    valores: List[int] = Field(default_factory=list, description="Lista de valores clasificados según la tabla")


# =====================================================================================
# 8. Cita Titular
# =====================================================================================
class CitaTitularValidada(BaseModel):
    # La parte del texto que corresponde a la cita
    cita: str = Field(..., description="El fragmento exacto de la declaración. Si es 1 (No), dejar vacío o poner 'N/A'.")
    
    # El código de clasificación
    tipo: int = Field(..., description="1=No, 2=Directa, 3=Indirecta")


# =====================================================================================
# 9a. Protagonistas que aparecen en la información
# =====================================================================================
class ProtagonistasDetectados(BaseModel):
    # Lista de cadenas con los nombres extraídos
    nombres: List[str] = Field(default_factory=list, description="Lista de nombres únicos detectados en la noticia")
    # Lista de enteros con los códigos correspondientes
    valores: List[int] = Field(default_factory=list, description="Lista de valores clasificados según la tabla")


# =====================================================================================
# 11. Género Periodista (Autoría)
# =====================================================================================
class GeneroPeriodistaValidado(BaseModel):
    # Field(...) hace el campo obligatorio
    # ge=0: Greater or equal to 0
    # le=5: Less or equal to 5
    codigo: int = Field(..., ge=0, le=7, description="Código de clasificación de autoría (0-7)")


# =====================================================================================
# 12. Tema
# =====================================================================================
class TemaConExplicacion(BaseModel):
    # Validamos que sea un entero entre 0 y 17
    codigo: int = Field(..., ge=0, le=17, description="Código numérico del tema")
    # Añadimos el campo de explicación
    explicacion: str = Field(..., description="Breve justificación de por qué se eligió este tema")


########################################################################################################################
########################################################################################################################


# =====================================================================================
# 13. IA Tema Central
# =====================================================================================
class IaTemaCentralConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No es tema central, 2=Sí es tema central")
    # Campo nuevo
    explicacion: str = Field(..., description="Justificación de la jerarquía de la información")


# =====================================================================================
# 14. Significado IA
# =====================================================================================
class IaSignificadoConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No explica significado, 2=Sí explica significado")
    # Campo nuevo
    explicacion: str = Field(..., description="Justificación: ¿Hay definiciones técnicas o es solo mención?")


# =====================================================================================
# 15. Menciona IA
# =====================================================================================
class MencionIaConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No menciona IA, 2=Sí menciona IA")
    # Explicación generada automáticamente por Python
    explicacion: str = Field(..., description="Justificación exacta (qué palabra o sigla se encontró)")


# =====================================================================================
# 16. Referencia a políticas en materia de género e igualdad
# =====================================================================================
class ReferenciaPoliticasGeneroConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No referencia políticas, 2=Sí referencia políticas de género")
    # Campo nuevo para el razonamiento
    explicacion: str = Field(..., description="Justificación de la decisión")


# =====================================================================================
# 17. Denuncia a la desigualdad de género
# =====================================================================================
class DenunciaDesigualdadConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No denuncia, 2=Sí denuncia desigualdad")
    # Nueva explicación
    explicacion: str = Field(..., description="Justificación de por qué se considera denuncia o no")


# =====================================================================================
# 18. Presencia de mujeres racializadas en la noticia
# =====================================================================================
class MujeresRacializadasConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No aparecen, 2=Sí aparecen mujeres racializadas")
    # Justificación
    explicacion: str = Field(..., description="Detalle sobre quiénes son las mujeres detectadas y su contexto étnico")


# =====================================================================================
# 19. Presencia de mujeres con discapacidad en la noticia
# =====================================================================================
class MujeresConDiscapacidadConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No aparecen, 2=Sí aparecen mujeres con discapacidad")
    # Justificación
    explicacion: str = Field(..., description="Detalle sobre quiénes son las mujeres detectadas y su contexto de discapacidad")


# =====================================================================================
# 20. Presencia de diversidad generacional en las mujeres que aparecen
# =====================================================================================
class MujeresGeneracionalidadConExplicacion(BaseModel):
    # 1 = No, 2 = Sí
    codigo: int = Field(..., ge=1, le=2, description="1=No hay diversidad generacional, 2=Sí hay diversidad (niñas, ancianas o mezcla)")
    # Justificación
    explicacion: str = Field(..., description="Detalle de las edades o generaciones identificadas en la noticia")


# =====================================================================================
# 21. Tiene Fotografías y 22. Número de fotografías
# =====================================================================================
class FotografiasValidadas(BaseModel):
    # Código: 1=No, 2=Sí
    codigo: int = Field(..., description="1 = No tiene fotos, 2 = Sí tiene fotos.")
    
    # Cantidad total
    cantidad: int = Field(..., ge=0, description="Número total de fotografías editoriales detectadas.")
    
    # Lista de links (URLs)
    evidencias: List[str] = Field(default_factory=list, description="Lista de URLs de las imágenes encontradas.")

# =====================================================================================
# 23. Tiene Fuentes y 24. Número de Fuentes
# =====================================================================================
class FuentesValidadas(BaseModel):
    # Un solo número entero: 2 si hay fuentes, 1 si no hay
    codigo: int = Field(..., description="1 = No tiene fuentes, 2 = Sí tiene fuentes.")
    
    # La lista de evidencias (nombres de las fuentes)
    evidencias: List[str] = Field(default_factory=list, description="Lista de nombres de las fuentes detectadas.")
    
    # Cantidad total
    cantidad: int = Field(..., description="Número total de fuentes.")


########################################################################################################################
########################################################################################################################

# =====================================================================================
# Bloque II. Lenguaje (25-39)
# =====================================================================================
def cargar_variables_desde_json(ruta_archivo: str = "variables.json") -> list:
    """Carga el archivo JSON completo desde el disco."""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo '{ruta_archivo}'. Asegúrate de crearlo con los datos del prompt anterior.")
    except json.JSONDecodeError:
        raise ValueError(f"El archivo '{ruta_archivo}' no tiene un formato JSON válido.")

def obtener_config_variable(datos_json: Any, codigo_buscado: str) -> dict:
    """
    Filtra el JSON para encontrar la configuración de una variable.
    Soporta lista de dicts O dict indexado por código.
    """
    # Formato v2: dict indexado por código
    if isinstance(datos_json, dict):
        # Saltar metadatos del documento
        if codigo_buscado in datos_json:
            return datos_json[codigo_buscado]
        # Variantes con prefijo
        for clave in [codigo_buscado, f"V{codigo_buscado}", f"v{codigo_buscado}"]:
            if clave in datos_json:
                return datos_json[clave]
        raise ValueError(f"Variable {codigo_buscado!r} no existe en el JSON")
 
    # Formato legacy: lista de dicts
    if isinstance(datos_json, list):
        for variable in datos_json:
            if variable.get("codigo") == codigo_buscado:
                return variable
        raise ValueError(f"Variable {codigo_buscado!r} no existe en la lista JSON")
 
    raise TypeError(f"Estructura JSON no reconocida: {type(datos_json).__name__}")

def cargar_texto_template(ruta: Union[str, Path]) -> str:
    """Lee el template .md desde disco."""
    return Path(ruta).read_text(encoding="utf-8")


# ============================================================================
# Aplanadores: convierten campos anidados a texto legible
# ============================================================================
 
def _aplanar_definicion(definicion: Union[str, dict]) -> str:
    """
    Convierte el campo `definicion` a texto.
      - Si es string (legacy): lo devuelve tal cual.
      - Si es dict (v2): formatea concepto + criterio_operativo + extras.
    """
    if isinstance(definicion, str):
        return definicion
 
    if not isinstance(definicion, dict):
        return str(definicion)
 
    partes = []
    # Orden recomendado: concepto primero, criterio operativo después
    if "concepto" in definicion:
        partes.append(definicion["concepto"])
    if "criterio_operativo" in definicion:
        partes.append("")
        partes.append("**Criterio operativo:** " + definicion["criterio_operativo"])
    # Campos extra específicos de una variable (ej. V25 tiene "salto_semantico")
    for clave, valor in definicion.items():
        if clave in ("concepto", "criterio_operativo"):
            continue
        etiqueta = clave.replace("_", " ").capitalize()
        partes.append("")
        partes.append(f"**{etiqueta}:** {valor}")
 
    return "\n".join(partes).strip()
 
 
def _aplanar_metodologia(metodologia: Union[str, dict]) -> str:
    """
    Convierte el campo `metodologia` a texto procedimental.
      - Si es string (legacy): lo devuelve tal cual.
      - Si es dict (v2): formatea los pasos numerados de forma legible.
    """
    if isinstance(metodologia, str):
        return metodologia
 
    if not isinstance(metodologia, dict):
        return str(metodologia)
 
    lineas = []
    for clave, valor in metodologia.items():
        # paso_1_identificacion → "Paso 1 — Identificación"
        partes = clave.split("_", 2)
        if len(partes) == 3 and partes[0] == "paso":
            nombre_paso = partes[2].replace("_", " ").capitalize()
            etiqueta = f"Paso {partes[1]} — {nombre_paso}"
        else:
            etiqueta = clave.replace("_", " ").capitalize()
        lineas.append(f"**{etiqueta}**")
        lineas.append(valor)
        lineas.append("")
 
    return "\n".join(lineas).strip()
 
 
def _aplanar_ejemplos_positivos(ejemplos: Union[str, list]) -> str:
    """
    Convierte `ejemplos` (legacy str) o `ejemplos_positivos` (v2 lista) a texto.
    """
    if isinstance(ejemplos, str):
        return ejemplos
 
    if not isinstance(ejemplos, list):
        return str(ejemplos)
 
    lineas = []
    for i, ej in enumerate(ejemplos, 1):
        if isinstance(ej, str):
            lineas.append(f"{i}. {ej}")
            continue
        if not isinstance(ej, dict):
            continue
        texto = ej.get("texto", "")
        razon = ej.get("razon", "") or ej.get("regla_inversion", "")
        etiqueta = ej.get("etiqueta")
        cabecera = f"{i}. «{texto}»"
        if etiqueta is not None:
            cabecera += f" → etiqueta {etiqueta}"
        lineas.append(cabecera)
        if razon:
            lineas.append(f"   Razón: {razon}")
    return "\n".join(lineas)
 
 
def _aplanar_ejemplos_negativos(ejemplos: list) -> str:
    """Convierte `ejemplos_negativos` (lista) a texto legible."""
    if not isinstance(ejemplos, list) or not ejemplos:
        return "(No se han documentado contraejemplos para esta variable.)"
 
    lineas = []
    for i, ej in enumerate(ejemplos, 1):
        if not isinstance(ej, dict):
            continue
        texto = ej.get("texto", "")
        razon = ej.get("razon_no_aplica", "")
        var_correcta = ej.get("variable_correcta", "")
        lineas.append(f"{i}. «{texto}»")
        if razon:
            lineas.append(f"   Por qué NO es {{nombre}}: {razon}")
        if var_correcta:
            lineas.append(f"   Si activa otra cosa, sería: {var_correcta}")
    return "\n".join(lineas)
 
 
def _aplanar_caso_limite(caso: dict) -> str:
    """Convierte `caso_limite_documentado` (dict) a texto."""
    if not isinstance(caso, dict) or not caso:
        return ""
    texto = caso.get("texto", "")
    decision = caso.get("decision", "")
    explicacion = caso.get("explicacion", "")
    if not texto:
        return ""
    partes = [f"Texto difícil: «{texto}»"]
    if decision:
        partes.append(f"Decisión: {decision}")
    if explicacion:
        partes.append(f"Explicación: {explicacion}")
    return "\n".join(partes)

# ============================================================================
# Lista de opciones y rango (lógica que ya tenías, conservada)
# ============================================================================
 
def _generar_lista_opciones(valores_posibles: list) -> str:
    """1 = No, 2 = Sí, ..."""
    return "\n".join(f"{i+1} = {val}" for i, val in enumerate(valores_posibles))
 
 
def _generar_rango_codigos(valores_posibles: list) -> str:
    """'1 o 2' o '1 al N'."""
    n = len(valores_posibles)
    if n == 2:
        return "1 o 2"
    return f"1 al {n}"
 

# ============================================================================
# Función principal: generar el prompt
# ============================================================================
 
def generar_prompt_dinamico(config: dict, texto: str, ruta_template: str) -> str:
    """
    Rellena el template .md con los datos de la variable.
    Compatible con JSON plano (legacy) y enriquecido (v2).
    """
    template = cargar_texto_template(ruta_template)
 
    valores = config["valores_posibles"]
    nombre = config["nombre"]
 
    # Aplanados
    definicion_str = _aplanar_definicion(config.get("definicion", ""))
    metodologia_str = _aplanar_metodologia(config.get("metodologia", ""))
 
    # Ejemplos positivos: aceptar nombres "ejemplos" (legacy) o "ejemplos_positivos" (v2)
    if "ejemplos_positivos" in config:
        ejemplos_pos_str = _aplanar_ejemplos_positivos(config["ejemplos_positivos"])
    else:
        ejemplos_pos_str = _aplanar_ejemplos_positivos(config.get("ejemplos", ""))
 
    # Ejemplos negativos (solo en v2)
    ejemplos_neg_str = _aplanar_ejemplos_negativos(config.get("ejemplos_negativos", []))
 
    # Caso límite (solo en v2)
    caso_limite_str = _aplanar_caso_limite(config.get("caso_limite_documentado", {}))
 
    # Opciones y rango (igual que en tu código)
    lista_opciones_str = _generar_lista_opciones(valores)
    rango_str = _generar_rango_codigos(valores)
 
    # Sustituir placeholders. Usamos un dict + format_map para que los
    # placeholders no presentes en el template no lancen error.
    contexto = {
        "nombre": nombre,
        "codigo": config.get("codigo", ""),
        "definicion": definicion_str,
        "metodologia": metodologia_str,
        # Compatibilidad con template viejo:
        "ejemplos": ejemplos_pos_str,
        # Placeholders nuevos disponibles para template enriquecido:
        "ejemplos_positivos": ejemplos_pos_str,
        "ejemplos_negativos": ejemplos_neg_str,
        "caso_limite": caso_limite_str,
        "texto_input": texto,
        "lista_opciones": lista_opciones_str,
        "rango_codigos": rango_str,
    }
 
    # Resolver el placeholder {nombre} dentro de los ejemplos negativos
    contexto["ejemplos_negativos"] = contexto["ejemplos_negativos"].replace(
        "{nombre}", nombre
    )
 
    # format_map permite que sobren claves en `contexto` sin lanzar KeyError
    # (algo que template.format() sí haría)
    return template.format_map(_SafeDict(contexto))
 
 
class _SafeDict(dict):
    """Dict que devuelve '' para claves ausentes en lugar de KeyError."""

    def __missing__(self, key):
        return "{" + key + "}"  # mantiene el placeholder si no hay dato


def parsear_respuesta_modelo(respuesta_raw: str) -> dict:
    """
    Parsea la respuesta cruda del modelo a dict.
    Maneja casos comunes en modelos pequeños:
      1. JSON correcto: {"codigo": 1, "explicacion": "...", "evidencias": [...]}
      2. JSON envuelto: {"respuesta": {"codigo": 1, ...}} → desenvuelve automáticamente
      3. Campos faltantes (típico cuando codigo=1, omiten "evidencias"): se rellenan con defaults seguros.
    """
    import json_repair
    data = json_repair.loads(respuesta_raw)

    if (
        isinstance(data, dict)
        and len(data) == 1
        and isinstance(next(iter(data.values())), dict)
    ):
        inner = next(iter(data.values()))
        if any(k in inner for k in ("codigo", "explicacion", "evidencias")):
            data = inner

    if not isinstance(data, dict):
        data = {}

    data.setdefault("codigo", 1)
    data.setdefault("explicacion", "")
    data.setdefault("evidencias", [])

    if not isinstance(data["evidencias"], list):
        data["evidencias"] = [str(data["evidencias"])] if data["evidencias"] else []

    return data
 

class BloqueAnalisisLenguajeSexista(BaseModel):
    """
    Modelo específico para 'lenguaje_sexista' que tiene 3 valores:
    1 = No
    2 = Sí
    3 = Sí; además se observa un salto semántico
    """
    codigo: int = Field(
        ..., 
        ge=1, 
        le=3, 
        description="Selección numérica: 1='No', 2='Sí', 3='Sí; además se observa un salto semántico'"
    )
    explicacion: str = Field(
        ..., 
        description="Cadena de pensamiento (Chain of Thought). Explica paso a paso por qué se ha seleccionado ese código."
    )
    evidencias: List[str] = Field(
        ..., 
        description="Lista exacta de frases, palabras o fragmentos extraídos del texto que justifican la decisión."
    )

class BloqueAnalisisBinario(BaseModel):
    """
    Modelo para variables con respuesta binaria:
    1 = No
    2 = Sí
    """
    codigo: int = Field(
        ..., 
        ge=1, 
        le=2, 
        description="Selección numérica: 1='No', 2='Sí'"
    )
    explicacion: str = Field(
        ..., 
        description="Cadena de pensamiento (Chain of Thought). Explica paso a paso por qué se ha seleccionado ese código, aplicando la metodología definida."
    )
    evidencias: List[str] = Field(
        ..., 
        description="Lista exacta de frases, palabras o fragmentos extraídos del texto que justifican la decisión."
    )


