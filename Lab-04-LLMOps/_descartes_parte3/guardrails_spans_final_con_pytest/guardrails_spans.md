# Spans guardrail de la sesión grupo-07-3-guardrails

1033 observations en la sesión; 394 son guardrails.

| guardrail | spans | decisiones | latencia mediana (ms) | p95 (ms) |
|---|---|---|---|---|
| freno_repeticion | 53 | bloquear×2, permitir×51 | 1.00 | 1.00 |
| guarda_sandbox | 54 | permitir×51, bloquear×2, requiere_confirmacion×1 | 1.00 | 1.00 |
| inyeccion_en_pdf | 22 | permitir×21, redactar×1 | 1.50 | 2.00 |
| pii_en_entrada | 22 | permitir×21, redactar×1 | 1.00 | 2.00 |
| presupuesto_tokens | 182 | permitir×180, bloquear×2 | 0.00 | 1.00 |
| salida_pii_secretos | 37 | permitir×36, redactar×1 | 1.00 | 1.00 |
| validar_entrada | 24 | advertir×22, permitir×2 | 1.00 | 1.00 |

## Datos personales en los spans

- «Mon…» aparece en 0 observation(s) ✓
