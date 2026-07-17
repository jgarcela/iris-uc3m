# Skill 3 — Highlighting y localización

Para **cada hallazgo** relevante (sesgo, incumplimiento de guía, etc.):

1. Identifica el **fragmento literal** del artículo (`excerpt`) copiado **exactamente** del texto UTF-8 proporcionado.
2. Calcula **`start`** y **`end`** como **índices de carácter Python** (posiciones sobre `str`, codepoints Unicode; igual que `len(texto)` y `texto[start:end]` debe reproducir `excerpt` si no hay espacios raros).
3. Si no puedes localizar con precisión, omite el hallazgo o marca `notas` explicando la incertidumbre; no inventes offsets.

El array **`findings`** del JSON unificado debe incluir, por cada ítem: `finding_id` (entero estable, 1..n), `variable_codigo`, `start`, `end`, `excerpt`.
