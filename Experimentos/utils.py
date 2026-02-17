
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Any, List
import re
import ollama
from pathlib import Path

# =====================================================================================
# 0. Ollama
# =====================================================================================
def consultar_ollama(prompt: str, modelo: str = "gemma3:4b") -> str:
    """
    Función genérica para enviar cualquier prompt a Ollama.
    Devuelve la respuesta del modelo como texto limpio.
    """
    try:
        response = ollama.chat(model=modelo, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
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

def obtener_config_variable(datos_json: list, codigo_buscado: str) -> dict:
    """Filtra la lista de variables para encontrar la configuración específica."""
    for variable in datos_json:
        if variable["codigo"] == codigo_buscado:
            return variable
    raise ValueError(f"La variable con código '{codigo_buscado}' no existe en el archivo JSON.")

def cargar_texto_template(ruta: str) -> str:
    return Path(ruta).read_text(encoding="utf-8")

def generar_prompt_dinamico(config: Dict, texto: str, ruta_template: str) -> str:
    """
    Rellena el template .md con los datos de la variable y calcula las opciones.
    """
    template = cargar_texto_template(ruta_template)
    
    # 1. Generar la lista de opciones (1 = X, 2 = Y, ...)
    # Esto toma ["No", "Sí", "Salto..."] y crea el string formateado
    valores = config['valores_posibles']
    lista_opciones_str = "\n".join([f"{i+1} = {val}" for i, val in enumerate(valores)])
    
    # 2. Definir el rango textual para las instrucciones
    if len(valores) == 2:
        rango_str = "1 o 2"
    else:
        rango_str = f"1 al {len(valores)}" # Ej: "1 al 3"

    # 3. Formatear el template
    prompt_final = template.format(
        nombre=config['nombre'],
        definicion=config['definicion'],
        metodologia=config['metodologia'],
        ejemplos=config['ejemplos'],
        texto_input=texto, # Contexto amplio para Ollama
        lista_opciones=lista_opciones_str, # <--- INYECCIÓN DINÁMICA
        rango_codigos=rango_str          # <--- INYECCIÓN DINÁMICA
    )
    
    return prompt_final

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


