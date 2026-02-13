from pydantic import BaseModel, Field
from typing import Optional, TypedDict

# BLOQUE 1: CONTENIDO_GENERAL
class AnalisisContenidoGeneral(BaseModel):
    # Variables Categóricas - CONTENIDO_GENERAL (SOLO códigos numéricos como string)
    tema_id: str = Field(..., description="Temática principal de la noticia. Códigos posibles: '1'=Científica/Investigación, '2'=Comunicación, '3'=De farándula o espectáculo, '4'=Deportiva, '5'=Economía (incluido: consumo; compras; viajes…), '6'=Educación/cultura, '7'=Empleo/Trabajo, '8'=Empresa, '9'=Judicial, '10'=Medioambiente, '11'=Policial, '12'=Política, '13'=Salud, '14'=Social, '15'=Tecnología, '16'=Transporte, '17'=Otros. Devolver SOLO código numérico. NO texto descriptivo.")
    genero_periodista_id: str = Field(..., description="Género del periodista/autor de la noticia. Códigos posibles: '1'=Masculino, '2'=Femenino, '3'=Mixto, '4'=Ns/Nc, '5'=Agencia/otros medios, '6'=Redacción, '7'=Corporativo. Devolver SOLO código numérico. NO 'Masculino' o 'Femenino'.")
    cita_titular_id: str = Field(..., description="Indica si el titular contiene una cita textual directa. Códigos posibles: '0'=No, '1'=Sí. Devolver SOLO código numérico. NO 'Sí' o 'No'.")
    
    # ORDEN LÓGICO: Primero la variable "matriz", luego el género
    # 1. Primero determinar si hay nombre propio en el titular
    nombre_propio_titular: str = Field(..., description="Nombre propio (nombre de persona) que aparece en el titular. Ejemplos: 'María García', 'Pedro Sánchez', 'Luther King'. Si NO hay nombre propio de persona, escribir 'No aplica'. IMPORTANTE: NO escribas el titular completo, solo el NOMBRE de la persona. Si el titular es '¿Dónde están los Luther King de hoy?', el nombre propio es 'Luther King' (o 'Martin Luther King'), NO el titular completo.")
    # 2. Luego, SOLO SI hay nombre propio (nombre_propio_titular != 'No aplica'), determinar el género
    genero_nombre_propio_titular_id: str = Field(..., description="Género de la persona cuyo nombre propio aparece en el titular. Códigos posibles: '1'=No hay, '2'=Sí, hombre, '3'=Sí, mujer, '4'=Sí, mujer y hombre. IMPORTANTE: Solo determina el género SI nombre_propio_titular != 'No aplica'. Si nombre_propio_titular = 'No aplica', entonces este campo DEBE ser '1' (No hay). Devolver SOLO código numérico.")
    
    # ORDEN LÓGICO: Primero la variable "matriz", luego el género
    # 1. Primero determinar si hay personas mencionadas
    personas_mencionadas_id: str = Field(..., description="Indica si se mencionan personas en la noticia. Códigos posibles: '1'=No, '2'=Sí. IMPORTANTE: Determina PRIMERO si hay personas mencionadas o no. Devolver SOLO código numérico. NO 'Sí' o 'No'.")
    # 2. Luego, SOLO SI hay personas mencionadas (personas_mencionadas_id = '2'), determinar el género
    genero_personas_mencionadas_id: str = Field(..., description="Género de las personas mencionadas en la noticia. Códigos posibles: '1'=No hay, '2'=Sí, hombre, '3'=Sí, mujer, '4'=Sí, mujer y hombre. IMPORTANTE: Solo determina el género SI personas_mencionadas_id = '2' (Sí). Si personas_mencionadas_id = '1' (No), entonces este campo DEBE ser '1' (No hay). Devolver SOLO código numérico.")
    
    # Variables IA
    menciona_ia_id: str = Field(..., description="Indica si la noticia menciona o hace referencia a inteligencia artificial. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    ia_tema_central_id: str = Field(..., description="Indica si la inteligencia artificial es el tema central de la noticia. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    significado_ia_id: str = Field(..., description="Indica si la noticia explica el significado o concepto de inteligencia artificial. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    
    # Variables de Igualdad y Diversidad (usando LENGUAJE_VARS: 1=No, 2=Sí)
    referencias_politicas_igualdad_id: str = Field(..., description="Indica si la noticia hace referencia a políticas públicas o medidas relacionadas con la igualdad de género. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    denuncia_desigualdad_genero_id: str = Field(..., description="Indica si la noticia denuncia situaciones de desigualdad por razón de género. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    presencia_mujeres_racializadas_id: str = Field(..., description="Indica si la noticia menciona o incluye a mujeres racializadas (mujeres de diferentes orígenes étnicos o raciales). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    presencia_mujeres_discapacidad_id: str = Field(..., description="Indica si la noticia menciona o incluye a mujeres con discapacidad. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    presencia_diversidad_generacional_id: str = Field(..., description="Indica si la noticia refleja diversidad generacional (diferentes grupos de edad). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")

# BLOQUE 2: LENGUAJE
class AnalisisLenguaje(BaseModel):
    # Variables de LENGUAJE (SOLO códigos numéricos, NO texto)
    lenguaje_sexista_id: str = Field(..., description="Indica si el texto utiliza lenguaje sexista o discriminatorio por razón de género. Códigos posibles: '1'=No, '2'=Sí, '3'=Sí, además se observa un salto semántico. Devolver SOLO código numérico. NO texto.")
    masculino_generico_id: str = Field(..., description="Indica si se utiliza el masculino genérico para referirse a grupos mixtos o a toda la población (ej: 'los españoles' para referirse a todas las personas). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO 'Sí' o 'No'.")
    hombre_humanidad_id: str = Field(..., description="Indica si se utiliza la palabra 'hombre' para referirse al conjunto de la humanidad o a todas las personas (ej: 'el hombre ha llegado a la luna'). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    uso_dual_zorra_id: str = Field(..., description="Indica si se utiliza el término 'zorra' o similar con doble sentido (insulto hacia mujeres vs. uso positivo hacia hombres). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    no_uso_cargos_mujeres_id: str = Field(..., description="Indica si se omite el uso de cargos o títulos profesionales cuando se refiere a mujeres (ej: 'María García' en lugar de 'la doctora María García'). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    sexismo_social_id: str = Field(..., description="Indica si el texto refleja o reproduce estereotipos y roles de género tradicionales (sexismo social). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    androcentrismo_id: str = Field(..., description="Indica si el texto presenta una perspectiva centrada en lo masculino como norma universal, invisibilizando o minimizando lo femenino. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    mujeres_sin_nombre_id: str = Field(..., description="Indica si se mencionan mujeres sin proporcionar su nombre completo o solo con apellido/genérico (ej: 'la esposa de', 'la madre'). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    asimetria_mujer_hombre_id: str = Field(..., description="Indica si hay asimetría en el tratamiento de mujeres y hombres en el texto (diferente nivel de detalle, importancia, o contexto). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    infantilizacion_mujeres_id: str = Field(..., description="Indica si se utiliza lenguaje que infantiliza o trata a las mujeres como menores (ej: diminutivos innecesarios, tono condescendiente). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    denominacion_sexualizada_id: str = Field(..., description="Indica si se utiliza denominación que sexualiza o enfatiza aspectos físicos/sexuales de las mujeres de forma innecesaria. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    denominacion_redundante_id: str = Field(..., description="Indica si se utiliza denominación redundante que enfatiza el género de forma innecesaria (ej: 'mujer científica' cuando 'científica' es suficiente). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    denominacion_dependiente_id: str = Field(..., description="Indica si se denomina a las mujeres en relación con un hombre (ej: 'esposa de', 'hija de', 'hermana de'). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    criterios_excepcion_id: str = Field(..., description="Indica si se aplican criterios de excepción o noticiabilidad diferentes para mujeres que para hombres (ej: destacar logros de mujeres como excepcionales). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    comparacion_mujeres_hombres_id: str = Field(..., description="Indica si se realiza una comparación explícita entre mujeres y hombres en el texto. Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")

# BLOQUE 3: FUENTES
class AnalisisFuentes(BaseModel):
    # Variables de Fuentes
    tiene_fuente_id: str = Field(..., description="Indica si la noticia incluye declaraciones o citas de fuentes (personas, instituciones, etc.). Códigos posibles: '1'=No, '2'=Sí. Devolver SOLO código numérico. NO texto.")
    numero_declaraciones: str = Field(..., description="Número total de declaraciones o citas de fuentes que aparecen en la noticia. Devolver número entero como string (ej: '0', '1', '3').")

# Estado del Grafo
class AgentState(TypedDict):
    id_noticia: str
    titular: str
    texto_noticia: str
    autor: str
    resultado_contenido: Optional[AnalisisContenidoGeneral]
    resultado_lenguaje: Optional[AnalisisLenguaje]
    resultado_fuentes: Optional[AnalisisFuentes]
    intentos: int
