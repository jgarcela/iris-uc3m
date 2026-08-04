"""Junta los CSV por shard de una corrida del cluster en un único CSV.

Uso:
    python merge_shards.py --input-dir results_b1_completo --output results_b1_completo/FULL.csv
    python merge_shards.py results_b1_completo            # salida por defecto: <dir>/FULL.csv
"""
import argparse
import glob
import os
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("input_dir", nargs="?", default=None)
ap.add_argument("--input-dir", dest="input_dir_opt", default=None)
ap.add_argument("--output", default=None)
ap.add_argument("--pattern", default="*shard*de*.csv",
                help="Glob de los CSV por shard (por defecto *shard*de*.csv)")
args = ap.parse_args()

input_dir = args.input_dir_opt or args.input_dir
if not input_dir:
    sys.exit("Falta el directorio de shards (posicional o --input-dir).")

patron = os.path.join(input_dir, args.pattern)
ficheros = sorted(glob.glob(patron))
if not ficheros:
    sys.exit(f"No se encontraron shards en {patron}")

print(f"Uniendo {len(ficheros)} shards:")
for f in ficheros:
    print(f"  - {f}")

df = pd.concat([pd.read_csv(f) for f in ficheros], ignore_index=True)
if "IdNoticia" in df.columns:
    df = df.drop_duplicates(subset="IdNoticia")

salida = args.output or os.path.join(input_dir, "FULL.csv")
df.to_csv(salida, index=False, encoding="utf-8")
print(f"\nOK: {len(df)} filas -> {salida}")
