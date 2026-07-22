"""
Recuperación (RAG ligero) sobre las guías de lenguaje reales de
Experimentos/methodology/ (las listadas en methodology_manifest.json).

Sin dependencias externas: índice TF-IDF en memoria sobre trozos de las guías,
troceados por encabezados Markdown. Permite que el agente consulte pasajes
literales de las guías expertas en runtime (tool `consultar_guia`), con cita
de fichero + sección. La tesis NO se indexa (no está en el manifest `guides`).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

METHODOLOGY_DIR = Path(__file__).resolve().parent.parent.parent / "methodology"
MANIFEST = METHODOLOGY_DIR / "methodology_manifest.json"

MAX_CHUNK_CHARS = 1500

_STOPWORDS = {
    "para", "como", "pero", "sus", "con", "una", "uno", "los", "las", "del",
    "que", "por", "mas", "muy", "sin", "sobre", "entre", "cuando", "donde",
    "este", "esta", "esto", "esos", "esas", "ese", "esa", "hay", "son", "ser",
    "puede", "cada", "toda", "todo", "todos", "todas", "segun", "tambien",
}


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _tokenizar(texto: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", _normalizar(texto))
            if len(t) >= 3 and t not in _STOPWORDS]


def _trocear(texto: str, fuente: str) -> list[dict]:
    """Trocea un .md por encabezados; agrupa hasta MAX_CHUNK_CHARS."""
    chunks: list[dict] = []
    titulo = "(inicio)"
    buffer: list[str] = []

    def _flush():
        cuerpo = "\n".join(buffer).strip()
        if cuerpo:
            chunks.append({"fuente": fuente, "titulo": titulo, "texto": cuerpo})

    for linea in texto.splitlines():
        if linea.startswith("#"):
            _flush()
            buffer = []
            titulo = linea.lstrip("#").strip() or titulo
        else:
            buffer.append(linea)
            if sum(len(x) for x in buffer) > MAX_CHUNK_CHARS:
                _flush()
                buffer = []
    _flush()
    return chunks


@lru_cache(maxsize=1)
def _indice() -> tuple:
    """Construye (chunks, tokens_por_chunk, idf) una sola vez."""
    guides = json.loads(MANIFEST.read_text(encoding="utf-8")).get("guides", [])
    chunks: list[dict] = []
    for nombre in guides:
        path = METHODOLOGY_DIR / nombre
        if path.is_file():
            chunks.extend(_trocear(path.read_text(encoding="utf-8"), nombre))
    tokens = [_tokenizar(c["texto"]) for c in chunks]
    df: dict[str, int] = {}
    for toks in tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = len(chunks) or 1
    idf = {t: math.log(1 + n / freq) for t, freq in df.items()}
    return chunks, tokens, idf


def buscar(query: str, k: int = 3) -> list[dict]:
    chunks, tokens, idf = _indice()
    q = _tokenizar(query)
    if not q or not chunks:
        return []
    puntuados = []
    for i, toks in enumerate(tokens):
        if not toks:
            continue
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = sum(tf.get(t, 0) * idf.get(t, 0.0) for t in q) / math.sqrt(len(toks))
        if score > 0:
            puntuados.append((score, i))
    puntuados.sort(reverse=True)
    return [{**chunks[i], "score": round(s, 3)} for s, i in puntuados[:k]]


def consultar_guia(query: str, k: int = 3) -> str:
    """Devuelve pasajes de las guías expertas relevantes a `query`, con cita."""
    res = buscar(query, k=k)
    if not res:
        return "(sin coincidencias en las guías para esa consulta)"
    bloques = []
    for r in res:
        cuerpo = r["texto"]
        if len(cuerpo) > MAX_CHUNK_CHARS:
            cuerpo = cuerpo[:MAX_CHUNK_CHARS] + "…"
        bloques.append(f"[{r['fuente']} › {r['titulo']}]\n{cuerpo}")
    return "\n\n".join(bloques)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "masculino generico inclusivo"
    print(consultar_guia(q))
