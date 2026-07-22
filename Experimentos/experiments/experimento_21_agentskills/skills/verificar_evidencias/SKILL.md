---
name: verificar_evidencias
description: Comprueba que los fragmentos citados como evidencia son literales del texto antes de cerrar el veredicto. Cárgala/úsala si vas a asignar codigo>1.
---

# Skill: verificación de evidencias

Refuerza la **trazabilidad** del veredicto (requisito HITL del proyecto IRIS). Úsala
siempre que vayas a asignar un `codigo > 1` (es decir, cuando afirmes que SÍ hay un fenómeno).

## Cuándo
- Antes de emitir `FINAL` con `codigo > 1`.
- No es necesaria si `codigo == 1` (sin fenómeno → `evidencias = []`).

## Cómo
1. Reúne los fragmentos exactos del texto que sostienen tu decisión.
2. Emite la acción: `VERIFICAR: ["fragmento 1", "fragmento 2"]`.
3. El sistema responde qué fragmentos son literales (`válidas`) y cuáles no (`inválidas`).
4. Conserva sólo las `válidas` en el campo `evidencias` del `FINAL`.
   Si una evidencia clave resulta `inválida`, revísala: cópiala **literal** del texto
   (mismas palabras, mismos signos) o reconsidera el veredicto.

## Regla
- `evidencias` deben ser spans **literales** del texto original, no paráfrasis.
- Un `codigo > 1` sin ninguna evidencia válida es sospechoso: reconsidera antes de cerrar.
