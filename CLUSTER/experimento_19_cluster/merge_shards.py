"""Junta los CSV por shard del experimento_19 en un único resultado final."""
import glob
import os
import sys

import pandas as pd

output_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
patron = os.path.join(output_dir, "19-Experimento-19_03_2026_resultados_modelo_2024_scrape_shard*de*.csv")

ficheros = sorted(glob.glob(patron))
if not ficheros:
    sys.exit(f"No se encontraron shards en {patron}")

print(f"Uniendo {len(ficheros)} shards:")
for f in ficheros:
    print(f"  - {f}")

df = pd.concat([pd.read_csv(f) for f in ficheros], ignore_index=True)
if "IdNoticia" in df.columns:
    df = df.drop_duplicates(subset="IdNoticia")

salida = os.path.join(output_dir, "19-Experimento-19_03_2026_resultados_modelo_2024_scrape_FULL.csv")
df.to_csv(salida, index=False, encoding="utf-8")
print(f"\nOK: {len(df)} filas -> {salida}")
