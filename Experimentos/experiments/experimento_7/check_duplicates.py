import pandas as pd

# Cargar tu archivo de resultados
ruta_resultados = "../../results/7-Experimento-7_03_2026_resultados_modelo_2024_scrape.csv"
df_resultados = pd.read_csv(ruta_resultados)

# Filtrar las filas que tienen el 'IdNoticia' repetido
duplicados = df_resultados[df_resultados.duplicated(subset=['IdNoticia'], keep=False)]

print(f"Total de filas en el archivo: {len(df_resultados)}")
print(f"Total de registros duplicados: {len(duplicados)}")

if not duplicados.empty:
    print("\nEstos son los IDs que están repetidos y cuántas veces aparecen:")
    print(duplicados['IdNoticia'].value_counts().head(10)) # Muestra los 10 más repetidos
else:
    print("\n¡Todo perfecto! No hay noticias duplicadas.")