5 corridas · 1 spans con error

### corridas

| traza | duracion_s | tokens_entrada | tokens_salida | llamadas_llm | spans_con_error |
|---|---|---|---|---|---|
| 8925f721ddd1c7a4f02a0e9591781878 | 468.31 | 55,141 | 142,925 | 13 | 0 |
| 8c6a290dddf12975afec555d6b260378 | 159.28 | 30,110 | 51,129 | 12 | 0 |
| 97abc90797e9adc2fc86c98a4b085263 | 1,272.07 | 107,663 | 413,020 | 28 | 1 |
| a60f37b747502cd133ad102ae2c4dd88 | 147.12 | 34,424 | 50,565 | 12 | 0 |
| feef2fd3ffd71c162db6b86412af757a | 674.47 | 78,831 | 212,547 | 20 | 0 |

### resumen_de_corridas

| corridas | tokens_entrada | tokens_salida | latencia_p50_s | latencia_p95_s | latencia_max_s |
|---|---|---|---|---|---|
| 5 | 306,169 | 870,186 | 468.31 | 1,152.55 | 1,272.07 |

### tiempo_por_tipo_de_span

| span | spans | total_s | promedio_s | errores |
|---|---|---|---|---|
| llm | 85 | 3,035.65 | 35.71 | 0 |
| corrida | 5 | 2,721.26 | 544.25 | 1 |
| agent | 111 | 2,721.02 | 24.51 | 0 |
| herramienta | 24 | 20.07 | 0.84 | 0 |
| guardrail | 130 | 0.12 | 0.00 | 0 |

### tokens_y_tiempo_por_agente

| agente | activaciones | tiempo_pared_s | llamadas_llm | tokens_entrada | tokens_salida | suma_llm_s | pct_del_tiempo_de_corridas | tokens_totales | concurrencia_llm |
|---|---|---|---|---|---|---|---|---|---|
| programador | 22 | 1,977.58 | 26 | 107618 | 595710 | 1,977.48 | 72.67 | 703328 | 1.00 |
| critico | 21 | 414.12 | 20 | 127976 | 128777 | 413.90 | 15.22 | 256753 | 1.00 |
| indexador | 5 | 135.66 | 28 | 11954 | 95804 | 479.71 | 4.99 | 107758 | 3.54 |
| redactor | 11 | 86.90 | 5 | 45320 | 25736 | 84.33 | 3.19 | 71056 | 0.97 |
| planificador | 6 | 80.27 | 6 | 13301 | 24159 | 80.24 | 2.95 | 37460 | 1.00 |
| ejecutor | 21 | 19.51 | 0 | 0 | 0 | 0 | 0.72 | 0 | 0 |
| investigador | 20 | 6.26 | 0 | 0 | 0 | 0 | 0.23 | 0 | 0 |
| lector | 5 | 0.72 | 0 | 0 | 0 | 0 | 0.03 | 0 | 0 |

### donde_se_va_el_tiempo_hojas

| span | nombre | spans | total_s | promedio_s | pct_de_la_suma_hoja |
|---|---|---|---|---|---|
| llm | programador | 26 | 1,977.48 | 76.06 | 64.71 |
| llm | indexador | 28 | 479.71 | 17.13 | 15.70 |
| llm | critico | 20 | 413.90 | 20.70 | 13.54 |
| llm | redactor | 5 | 84.33 | 16.87 | 2.76 |
| llm | planificador | 6 | 80.24 | 13.37 | 2.63 |
| herramienta | sandbox_python | 19 | 19.36 | 1.02 | 0.63 |
| herramienta | leer_pdf | 5 | 0.71 | 0.14 | 0.02 |
| guardrail | guarda_sandbox | 19 | 0.07 | 0.00 | 0.00 |
| guardrail | presupuesto_tokens | 87 | 0.04 | 0.00 | 0.00 |
| guardrail | freno_repeticion | 19 | 0.01 | 0.00 | 0.00 |
| guardrail | validar_entrada | 5 | 0.00 | 0.00 | 0.00 |

Agente que más TIEMPO consume (pared): programador con 1977.6 s (73 % del tiempo de las corridas).
Agente que más TOKENS gasta: programador con 703,328 (107,618 de entrada, 595,710 de salida).
Cuello de botella entre spans hoja (mayor tiempo TOTAL): llm:programador con 1977.5 s (65 % de la suma de hojas).
AVISO: la suma de spans hoja (3056 s) supera el tiempo de las corridas (2721 s): hay llamadas en paralelo. Para el reparto del tiempo real usa la tabla por agente (tiempo de pared).
