
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import re

# =====================================================================================
# 7a. Nombre Propio Titular
# =====================================================================================
class NombresDetectados(BaseModel):
    # Lista de cadenas con los nombres extraídos
    nombres: List[str] = Field(default_factory=list, description="Lista de nombres propios detectados")
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
    # Variable 1: ¿Tiene fotos? (1=No, 2=Sí)
    tiene_fotos_codigo: int = Field(..., ge=1, le=2, description="1=No, 2=Sí")
    
    # Variable 2: Número exacto de fotos
    cantidad: int = Field(..., ge=0, description="Número total de fotografías detectadas")