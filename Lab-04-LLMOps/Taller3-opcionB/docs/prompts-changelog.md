cat > docs/prompts-changelog.md <<'EOF'
# Registro de versiones de prompts y guardrails

Una fila por cambio. No se reescribe una versión ya medida: si algo cambia, es una versión nueva.
La versión activa se declara con `SOLVER_PROMPT_VERSION` y queda en el atributo `prompt_version` de los spans.

| Versión | Fecha | Cambio | Archivos |
|---|---|---|---|
| v1 | 2026-10-10 | Prompts del Taller 03 sin cambios: línea base del Taller 04. | `solver/agentes/*.py` |
| v2 | 2026-10-14 | Freno de presupuesto con tope de salida (no cambia ningún prompt): la llamada no puede costar más de lo que queda, para no gastar la reserva del redactor. Se activa con `SOLVER_FRENO_SALIDA=1`. | `solver/cliente_llm.py`, `solver/config.py` |
EOF