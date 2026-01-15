from pydantic import BaseModel, Field  # <--- CAMBIO AQUÍ
from typing import Optional

class AnalisisContenidoGeneral(BaseModel):
    # Variables de Texto Libre
    nombre_propio_titular: str = Field(..., description="Escribe el nombre propio que aparece en el titular. Si no hay, pon 'No aplica'.")
    
    # Variables Categóricas (Debe devolver el ID como string, ej: '1', '12')
    tema_id: str = Field(..., description="ID numérico del TEMA según el libro de códigos.")
    genero_periodista_id: str = Field(..., description="ID numérico del GENERO_PERIODISTA.")
    cita_titular_id: str = Field(..., description="ID numérico de CITA_TITULAR (0 o 1).")
    genero_nombre_propio_titular_id: str = Field(..., description="ID numérico de GENERO_NOMBRE_PROPIO_TITULAR.")
    personas_mencionadas_id: str = Field(..., description="ID numérico de PERSONAS_MENCIONADAS (1 o 2).")
    genero_personas_mencionadas_id: str = Field(..., description="ID numérico de GENERO_PERSONAS_MENCIONADAS.")
    
    # Campo Meta-Cognitivo (El agente explica por qué)
    razonamiento: str = Field(..., description="Explica brevemente por qué elegiste el TEMA y el GÉNERO DEL PERIODISTA.")

# Estado del Grafo
from typing import TypedDict, Optional

class AgentState(TypedDict):
    id_noticia: str
    texto_noticia: str
    resultado: Optional[AnalisisContenidoGeneral]
    errores_validacion: Optional[str]
    feedback_humano: Optional[str]
    intentos: int