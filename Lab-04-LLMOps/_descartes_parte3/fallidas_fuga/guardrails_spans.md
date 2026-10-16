# Spans guardrail de la sesión grupo-07-3-guardrails

68 observations en la sesión; 28 son guardrails.

| guardrail | spans | decisiones | latencia mediana (ms) | p95 (ms) |
|---|---|---|---|---|
| freno_repeticion | 2 | permitir×2 | 1.00 | 1.00 |
| guarda_sandbox | 2 | permitir×2 | 3.00 | 4.00 |
| inyeccion_en_pdf | 2 | redactar×1, permitir×1 | 4.00 | 6.00 |
| pii_en_entrada | 2 | permitir×1, redactar×1 | 2.00 | 3.00 |
| presupuesto_tokens | 12 | permitir×12 | 1.00 | 1.00 |
| salida_pii_secretos | 2 | permitir×2 | 0.50 | 1.00 |
| validar_entrada | 6 | permitir×6 | 1.00 | 2.00 |

## Datos personales en los spans

- «Mon…» aparece en 4 observation(s)  ← FUGA
- «CED…» aparece en 0 observation(s) ✓
