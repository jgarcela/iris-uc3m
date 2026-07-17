"""
Objetivo
---
Descargar y parsear un artículo con newspaper3k y devuelver sus metadatos e imágenes.


Uso
---
Desde terminal (Python con ``newspaper3k`` instalado; conviene un venv):

    python newspaper_fetch.py "https://elpais.com/opinion/2024-01-02/estupidez-artificial.html"

Si no pasas URL, se usa por defecto ese artículo de ejemplo (columna de opinión).

Desde otro módulo en el mismo directorio:

    from newspaper_fetch import process_article

    url = "https://elpais.com/opinion/2024-01-02/estupidez-artificial.html"
    r = process_article(url)
    if r.error:
        print(r.error)
    else:
        print(r.title)           # título
        print(r.text[:500])     # cuerpo (truncado aquí)
        print(r.top_image)      # URL de la imagen principal, si la detecta
        print(r.images)         # lista de URLs de imágenes en el artículo
"""

from dataclasses import dataclass
from typing import List, Optional

from newspaper import Article, Config


user_agent = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_config = Config()
_config.browser_user_agent = user_agent
_config.request_timeout = 10


@dataclass
class ArticleResult:
    url: str
    title: str
    text: str
    authors: List[str]
    publish_date: Optional[str]
    top_image: Optional[str]
    images: List[str]
    error: Optional[str] = None


def process_article(url: str, *, config: Optional[Config] = None) -> ArticleResult:
    """
    Descarga y parsea la URL con newspaper. Devuelve texto y listas de URLs de imágenes
    (top_image + images del artículo).
    """
    cfg = config if config is not None else _config
    articulo = Article(url, config=cfg)
    try:
        articulo.download()
        articulo.parse()
    except Exception as e:
        return ArticleResult(
            url=url,
            title="",
            text="",
            authors=[],
            publish_date=None,
            top_image=None,
            images=[],
            error=str(e),
        )

    pub = articulo.publish_date
    pub_str = pub.isoformat() if pub is not None else None

    # newspaper3k: imágenes del cuerpo del artículo; top_image suele ser la principal
    raw_images = list(articulo.images) if articulo.images else []
    top = articulo.top_image or None

    return ArticleResult(
        url=url,
        title=articulo.title or "",
        text=articulo.text or "",
        authors=list(articulo.authors) if articulo.authors else [],
        publish_date=pub_str,
        top_image=top,
        images=raw_images,
        error=None,
    )


if __name__ == "__main__":
    import sys

    # Ejemplo de artículo: https://elpais.com/opinion/2024-01-02/estupidez-artificial.html
    _default_example = (
        "https://elpais.com/opinion/2024-01-02/estupidez-artificial.html"
    )
    test_url = sys.argv[1] if len(sys.argv) > 1 else _default_example
    r = process_article(test_url)
    if r.error:
        print("Error:", r.error)
    else:
        print("Title:", r.title)
        print("Top image:", r.top_image)
        print("Images (%d):" % len(r.images))
        for img in r.images:
            print(" ", img)
