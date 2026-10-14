# Juez del solver — variante `mejora`

Juez: `gemma3:27b` (Ollama de la H200, `temperature=0`) · 2026-10-14 12:00 · 52 comprobaciones juzgadas (sin entregable, no juzgadas: ninguna)

## Calibración 1 — juez contra el verificador automático

- comprobaciones con las dos calificaciones: **47** (de tipo `juez`, sin verificador: 5)
- acuerdo: **0.957** · kappa de Cohen: **0.000**

| | juez: aprueba | juez: rechaza |
|---|---|---|
| **verificador: aprueba** | 45 | 2 (falso negativo del juez) |
| **verificador: rechaza** | 0 (falso positivo del juez) | 0 |

### Por tipo de comprobación

| tipo | n | verificador aprueba | juez aprueba | acuerdo | kappa |
|---|---|---|---|---|---|
| cifra | 7 | 1.00 | 1.00 | 1.00 | — |
| cifra_presente | 19 | 1.00 | 0.89 | 0.89 | 0.00 |
| contiene | 3 | 1.00 | 1.00 | 1.00 | — |
| no_aparece | 2 | 1.00 | 1.00 | 1.00 | — |
| paginas_max | 1 | 1.00 | 1.00 | 1.00 | — |
| palabras_max | 4 | 1.00 | 1.00 | 1.00 | — |
| procedencia | 5 | 1.00 | 1.00 | 1.00 | — |
| secciones | 6 | 1.00 | 1.00 | 1.00 | — |

### Por tarea

| tarea | n | verificador aprueba | juez aprueba | acuerdo |
|---|---|---|---|---|
| A | 7 | 7/7 | 7/7 | 1.00 |
| B | 6 | 6/6 | 6/6 | 1.00 |
| C | 7 | 7/7 | 7/7 | 1.00 |
| I1 | 6 | 6/6 | 6/6 | 1.00 |
| P1 | 5 | 5/5 | 3/5 | 0.60 |
| S1 | 7 | 7/7 | 7/7 | 1.00 |
| S2 | 9 | 9/9 | 9/9 | 1.00 |

### Comprobaciones de tipo `juez` (solo las califica el juez)

| id | tarea | juez | razón |
|---|---|---|---|
| A10 | A | aprueba | La discusión utiliza las cifras de la curva de aprendizaje para determinar el tamaño de entrenamiento óptimo para cada modelo, y explica el comportamiento en té |
| B10 | B | aprueba | La Parte 4 explica correctamente el comportamiento de la distribución con T→0 y T→∞, y utiliza las entropías calculadas como evidencia para respaldar sus afirma |
| C09 | C | aprueba | El análisis por consulta explica claramente que q5 falla debido a la falta de términos compartidos entre la consulta y el documento relevante (d09), lo que perj |
| S213 | S2 | aprueba | La respuesta 18 está pendiente, pero el texto indica que no se puede concluir que el modelo B es mejor, y se mencionan limitaciones del diseño. |
| I108 | I1 | aprueba | El reporte calcula e interpreta correctamente las métricas solicitadas y no se limita a la frase de aprobación. |

### Desacuerdos a revisar uno por uno

A veces el que se equivoca es el verificador (p. ej. un porcentaje escrito como fracción, un nombre escrito distinto). Para cada uno: ¿quién tiene razón? Si es el verificador, corrige el golden y dilo.

| id | tipo | verificador | juez | detalle del verificador | razón del juez |
|---|---|---|---|---|---|
| P104 | cifra_presente | aprueba | falla | verdad=1293.7083 ±0.006; hallada | La media calculada es incorrecta. La media correcta es (1250.50 + 980.25 + 1430.75 + 1120.00 + 1675.50 + 1305.25) / 6 = 1293.71. El entregab |
| P105 | cifra_presente | aprueba | falla | verdad=1277.8750 ±0.006; hallada | La mediana calculada es incorrecta. Ordenando los valores: 980.25, 1120.00, 1250.50, 1305.25, 1430.75, 1675.50. La mediana es (1250.50 + 130 |

Formato de las respuestas del juez: completo 52.

_Calibración 2 pendiente: falta `--etiquetas etiquetas.csv`._
