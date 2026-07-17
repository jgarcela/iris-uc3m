# Extract structured JSON

Use this skill when the user asks for machine-readable fields from text.

## Steps

1. List the fields the user requested (or infer: title, entities, sentiment).
2. Copy values verbatim from the text when possible.
3. Return **only** a single JSON object, no markdown fences.

## Output format

Valid JSON object with string values unless numbers are clearly numeric.
