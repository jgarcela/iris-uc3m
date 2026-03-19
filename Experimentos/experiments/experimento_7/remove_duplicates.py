import pandas as pd

# La ruta exacta de tu archivo con los duplicados
archivo = "../../results/7-Experimento-7_03_2026_resultados_modelo_2024_scrape.csv"

# 1. Leemos el CSV
df = pd.read_csv(archivo)

# 2. Eliminamos duplicados basándonos en tu ID (nos quedamos con el primero que se guardó)
df_limpio = df.drop_duplicates(subset=['IdNoticia'], keep='first')

# 3. Sobrescribimos el archivo original con los datos limpios
df_limpio.to_csv(archivo, index=False, encoding='utf-8')

print(f"¡Listo! Pasamos de {len(df)} a {len(df_limpio)} filas. Duplicados eliminados.")