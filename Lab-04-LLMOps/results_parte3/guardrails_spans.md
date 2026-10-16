# Spans guardrail de la sesión grupo-07-3-guardrails

66 observations en la sesión; 28 son guardrails.

| guardrail | spans | decisiones | latencia mediana (ms) | p95 (ms) |
|---|---|---|---|---|
| freno_repeticion | 2 | permitir×2 | 0.50 | 1.00 |
| guarda_sandbox | 5 | bloquear×2, permitir×3 | 1.00 | 1.00 |
| inyeccion_en_pdf | 2 | redactar×1, permitir×1 | 3.00 | 4.00 |
| pii_en_entrada | 2 | permitir×1, redactar×1 | 2.00 | 2.00 |
| presupuesto_tokens | 12 | permitir×12 | 0.00 | 1.00 |
| salida_pii_secretos | 3 | redactar×1, permitir×2 | 1.00 | 1.00 |
| validar_entrada | 2 | permitir×2 | 1.00 | 1.00 |

## Datos personales en los spans

- «Mon…» aparece en 0 observation(s) ✓
- «170…» aparece en 0 observation(s) ✓
