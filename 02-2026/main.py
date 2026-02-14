# 1. Filtramos el DataFrame para obtener solo las filas del año 2024
df_2024 = data[data['año'] == 2024]

# 2. Tomamos 1000 muestras aleatorias
# Nota: Si el total es menor a 1000, fallará a menos que uses replace=True
muestras_2024 = df_2024.sample(n=1000, random_state=42)

# 3. Guardamos solo la columna 'IdNoticia'
lista_ids = muestras_2024['IdNoticia'].tolist()

print(f"Lista de IDs: {lista_ids}")