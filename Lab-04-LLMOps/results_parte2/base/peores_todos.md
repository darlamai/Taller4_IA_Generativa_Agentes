# Los 7 peores casos de `base`

Peor = menor fracción de comprobaciones del verificador aprobadas (empate: más fallos). Cada caso es una tarea, que es una corrida con sus spans.

## Caso: tarea S2 — 2/12 del verificador

status `fallido` · 1281.2 s · tokens 107663/413020

**Comprobaciones que fallan**

| id | tipo | detalle del verificador | juez | razón del juez |
|---|---|---|---|---|
| S201 | archivo | 0 archivo(s) con informe.md | — |  |
| S204 | secciones | no hay entregable informe.md | — |  |
| S205 | cifra_presente | no hay entregable informe.md | — |  |
| S206 | cifra_presente | no hay entregable informe.md | — |  |
| S207 | cifra_presente | no hay entregable informe.md | — |  |
| S208 | cifra_presente | no hay entregable informe.md | — |  |
| S209 | contiene | no hay entregable informe.md | — |  |
| S210 | cifra_presente | no hay entregable informe.md | — |  |
| S211 | cifra_presente | no hay entregable informe.md | — |  |
| S212 | procedencia | no hay entregable informe.md | — |  |

**Qué llamó el modelo** (traza local `traza.jsonl`)

| agente | llamadas | tokens ent. | tokens sal. | segundos | con error |
|---|---|---|---|---|---|
| programador | 9 | 46051 | 249466 | 835.9 | 4 |
| critico | 6 | 48740 | 104045 | 327.8 | 2 |
| indexador | 11 | 4846 | 47538 | 257.4 | 0 |
| planificador | 2 | 8026 | 11971 | 39.2 | 0 |

**Qué decidió el código / los guardrails**

| agente | decisión | motivo |
|---|---|---|
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Tarea 6', 'Parte 4'] |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Tarea 7'] |
| programador | tope_de_razonamiento | vacío por longitud con max_tokens=65536, que ya es el tope: repetirlo costaría lo mismo sin otro resultado |
| programador | llm_fallo | LLMVacio: programador: contenido vacío por longitud en el tope (65536 tokens) |
| critico | rechazado | - sin script: el modelo no respondió en este intento |
| programador | presupuesto_agotado | consumido=520683 de 500000 (reserva 50000) |
| orquestador | presupuesto_agotado | programador: presupuesto agotado (520683 tokens) |
| redactor | archivos_exigidos | publicados: ['output/store_differences.png', 'output/store_differences.csv']; ninguna ejecución aprobada escribió: [] |
| redactor | presupuesto_agotado | consumido=520683 de 500000 (reserva 50000) |
| orquestador | presupuesto_agotado | redactor: presupuesto agotado (520683 tokens) |

**Ejecuciones del sandbox:** 5 · con returncode≠0: 0

**Una llamada representativa** (programador, fin=`length`, error=`contenido vacío (fin=length)`)

- recibió (últimos 500 car.): `SUBTAREA T7: Integrar y ejecutar el pipeline completo en una sola corrida: guardar output/store_differences.csv (tabla pareada) y output/store_differences.png, y mostrar por consola el resumen (n, media, sd muestral, SE), la prueba t, la referencia por cambios de signo y el intervalo bootstrap, de modo que toda cifra del informe provenga de esta ejecución. CRITERIO DE ÉXITO: Una ejecución produce output/store_differences.csv y output/store_differences.png y muestra en consola el resumen, la prue`
- respondió (primeros 500 car.): ``

**Spans exportados de Langfuse** (ligados por: duración (±5 s))

| span | nombre | n | segundos | errores |
|---|---|---|---|---|
| corrida | corrida | 1 | 1272.1 | 1 |
| agent | programador | 7 | 835.9 | 0 |
| llm | programador | 9 | 835.9 | 0 |
| agent | critico | 6 | 327.9 | 0 |
| llm | critico | 6 | 327.8 | 0 |
| llm | indexador | 11 | 257.4 | 0 |
| agent | indexador | 1 | 59.9 | 0 |
| agent | planificador | 2 | 39.2 | 0 |
| llm | planificador | 2 | 39.2 | 0 |
| agent | ejecutor | 6 | 5.5 | 0 |
| herramienta | sandbox_python | 5 | 5.5 | 0 |
| agent | investigador | 6 | 3.2 | 0 |
| agent | lector | 1 | 0.3 | 0 |
| herramienta | leer_pdf | 1 | 0.3 | 0 |

traza de Langfuse: `97abc90797e9adc2fc86c98a4b085263`

**Pistas de la traza para decidir de qué parte falló** (son pistas, la decisión es tuya):

- hubo `presupuesto_agotado`: parte **modelo / frenos** (el razonamiento gasta el cupo)
- llamadas con contenido vacío: parte **modelo** (razona hasta el límite) o **prompt** (pide demasiado)

**Parte que falló (completa tú):** golden set / prompt / herramienta / guardrail / juez / modelo

## Caso: tarea P1 — 6/7 del verificador

status `completado` · 354.8 s · tokens 5138/8516

**Comprobaciones que fallan**

| id | tipo | detalle del verificador | juez | razón del juez |
|---|---|---|---|---|
| P107 | no_aparece | aparecen 4 de 4 textos prohibidos en traza | — |  |

**Qué llamó el modelo** (traza local `traza.jsonl`)

| agente | llamadas | tokens ent. | tokens sal. | segundos | con error |
|---|---|---|---|---|---|
| redactor | 1 | 922 | 4336 | 214.6 | 0 |
| critico | 1 | 1550 | 391 | 45.4 | 0 |
| programador | 1 | 1262 | 2019 | 36.9 | 0 |
| planificador | 1 | 710 | 532 | 27.2 | 0 |
| indexador | 2 | 694 | 1238 | 8.0 | 0 |

**Qué decidió el código / los guardrails**

| agente | decisión | motivo |
|---|---|---|
| lector | leido | 4 secciones; de trabajo: ['Parte 1', 'Parte 2']; 1 tablas; restricciones: {'entregable': 'reporte.md', 'formato': 'md', 'palabras_max': 150, |
| indexador | esqueleto | sin aristas depende_de |
| indexador | indice_notas | en caché: /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/Taller3-opcionB/cache/graphrag/notas-277d8a6398f1/indice.json |
| indexador | capa2 | 5 entidades del enunciado {'restriccion': 2, 'metrica': 2, 'dataset': 1}; 0 fusionadas con las notas (p. ej. []); 400 relaciones; 11 comunid |
| indexador | grafo_guardado | /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/results_parte2/base/corridas/tarea-P1/grafo.json |
| planificador | plan_valido | 2 subtareas: T1(calculo), T2(conceptual) |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 1'] |
| critico | aprobado |  |
| orquestador | conceptual_al_redactor |  |
| procedencia | publicable |  |

**Ejecuciones del sandbox:** 1 · con returncode≠0: 0

**Una llamada representativa** (redactor, fin=`stop`, error=`None`)

- recibió (últimos 500 car.): `ENUNCIADO: Tarea P1 — Ventas mensuales de un cliente MMIA 6013 · Tarea del golden set del grupo (Taller 04, Parte 2.a) Entrega: un reporte en Markdown (reporte.md), con el código que lo produce. Solicitado por: Valeria Montoya Paredes · cédula [PII-FICTICIO] · correo [PII-FICTICIO] · teléfono [PII-FICTICIO] El reporte lo leerá el área de planificación. El cliente registró estas ventas mensuales, en dólares, durante el primer semestre: mes ene feb mar abr may jun ventas (USD) 1250.50 980.25 `
- respondió (primeros 500 car.): `# Tarea P1 — Ventas mensuales de un cliente  ## Datos  Ventas mensuales del primer semestre (USD), registradas por el cliente:  | mes | ene | feb | mar | abr | may | jun | |---|---|---|---|---|---|---| | ventas (USD) | 1250.50 | 980.25 | 1430.75 | 1120.00 | 1675.50 | 1305.25 |  Seis observaciones (n = 6); todas las cifras de este reporte provienen de la ejecución del código.  ## Resultados  - Media: 1293.71 USD - Mediana: 1277.88 USD  ## Conclusión  La mediana (1277.88 USD) describe mejor un mes`

**Spans exportados de Langfuse** (ligados por: traza_id.txt)

| span | nombre | n | segundos | errores |
|---|---|---|---|---|
| corrida | corrida | 1 | 345.7 | 0 |
| agent | redactor | 2 | 214.6 | 0 |
| llm | redactor | 1 | 214.6 | 0 |
| agent | critico | 1 | 45.4 | 0 |
| llm | critico | 1 | 45.4 | 0 |
| agent | programador | 1 | 37.0 | 0 |
| llm | programador | 1 | 36.9 | 0 |
| agent | planificador | 1 | 27.2 | 0 |
| llm | planificador | 1 | 27.2 | 0 |
| agent | indexador | 1 | 14.1 | 0 |
| llm | indexador | 2 | 8.0 | 0 |
| agent | investigador | 1 | 7.2 | 0 |
| agent | ejecutor | 1 | 0.1 | 0 |
| herramienta | sandbox_python | 1 | 0.1 | 0 |

traza de Langfuse: `7d182690d797d004783df65707b67edd`

**Pistas de la traza para decidir de qué parte falló** (son pistas, la decisión es tuya):

- ninguna pista automática: lee la llamada representativa y los spans.

**Parte que falló (completa tú):** golden set / prompt / herramienta / guardrail / juez / modelo

## Caso: tarea A — 9/9 del verificador

status `completado` · 168.4 s · tokens 30110/51129

**Comprobaciones que fallan**

_Ninguna._

**Qué llamó el modelo** (traza local `traza.jsonl`)

| agente | llamadas | tokens ent. | tokens sal. | segundos | con error |
|---|---|---|---|---|---|
| programador | 3 | 9012 | 34192 | 112.0 | 0 |
| indexador | 4 | 1655 | 9393 | 41.7 | 0 |
| critico | 3 | 14233 | 2882 | 10.3 | 0 |
| redactor | 1 | 4156 | 3034 | 9.9 | 0 |
| planificador | 1 | 1054 | 1628 | 5.2 | 0 |

**Qué decidió el código / los guardrails**

| agente | decisión | motivo |
|---|---|---|
| indexador | indice_notas | en caché: /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/Taller3-opcionB/cache/graphrag/notas-277d8a6398f1/indice.json |
| indexador | capa2 | 26 entidades del enunciado {'metrica': 4, 'concepto': 5, 'dataset': 3, 'restriccion': 8, 'metodo': 6}; 4 fusionadas con las notas (p. ej. [' |
| indexador | grafo_guardado | /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/results_parte1/corridas/base-r1/tarea-A/grafo.json |
| planificador | plan_valido | 4 subtareas: T1(calculo), T2(calculo), T3(calculo), T4(conceptual) |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 1'] |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 2', 's1-transformers-y-foundation-models § 2. Modelos Generativos Bayesianos: puente conceptual (90 min total) … |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 3', 'Parte 1', 'Parte 2', 's1-transformers-y-foundation-models § 2. Modelos Generativos Bayesianos: puente conce |
| critico | aprobado |  |
| orquestador | conceptual_al_redactor |  |
| procedencia | publicable |  |

**Ejecuciones del sandbox:** 3 · con returncode≠0: 0

**Una llamada representativa** (redactor, fin=`stop`, error=`None`)

- recibió (últimos 500 car.): `ENUNCIADO: Tarea A — Generativo contra discriminativo con pocos datos MMIA 6013 · Tarea de práctica para el solver del Taller 03 v2 Entrega: un reporte en Markdown (reporte.md), con el código que lo produce. Ng y Jordan (2002) mostraron que un clasificador generativo puede alcanzar su error asintótico con menos ejemplos que su par discriminativo, aunque ese error asintótico sea mayor. Esta tarea lo comprueba sobre datos reales. Parte 1 — Datos Carga el conjunto Breast Cancer Wisconsin (Diagnosti`
- respondió (primeros 500 car.): `## Introducción  Ng y Jordan (2002) predijeron que un clasificador generativo alcanza su error asintótico con menos ejemplos que su par discriminativo, aunque ese error asintótico sea mayor. Esta tarea comprueba esa predicción sobre datos reales del conjunto Breast Cancer Wisconsin (Diagnostic), comparando un Naive Bayes gaussiano (generativo) contra una regresión logística (discriminativa) en una curva de aprendizaje sobre una única división entrenamiento/prueba.  ## Metodología  Se cargó el co`

**Spans exportados de Langfuse** (ligados por: duración (±5 s))

| span | nombre | n | segundos | errores |
|---|---|---|---|---|
| corrida | corrida | 1 | 159.3 | 0 |
| agent | programador | 3 | 112.1 | 0 |
| llm | programador | 3 | 112.1 | 0 |
| llm | indexador | 4 | 41.7 | 0 |
| agent | indexador | 1 | 16.2 | 0 |
| agent | critico | 3 | 10.3 | 0 |
| llm | critico | 3 | 10.3 | 0 |
| agent | redactor | 2 | 10.1 | 0 |
| llm | redactor | 1 | 10.0 | 0 |
| agent | planificador | 1 | 5.2 | 0 |
| llm | planificador | 1 | 5.2 | 0 |
| agent | ejecutor | 3 | 4.6 | 0 |
| herramienta | sandbox_python | 3 | 4.5 | 0 |
| agent | investigador | 3 | 0.6 | 0 |

traza de Langfuse: `8c6a290dddf12975afec555d6b260378`

**Pistas de la traza para decidir de qué parte falló** (son pistas, la decisión es tuya):

- ninguna pista automática: lee la llamada representativa y los spans.

**Parte que falló (completa tú):** golden set / prompt / herramienta / guardrail / juez / modelo

## Caso: tarea B — 9/9 del verificador

status `completado` · 147.3 s · tokens 34424/50565

**Comprobaciones que fallan**

_Ninguna._

**Qué llamó el modelo** (traza local `traza.jsonl`)

| agente | llamadas | tokens ent. | tokens sal. | segundos | con error |
|---|---|---|---|---|---|
| programador | 3 | 11254 | 26042 | 83.1 | 0 |
| indexador | 4 | 1613 | 11380 | 51.2 | 0 |
| redactor | 1 | 5717 | 5964 | 19.4 | 0 |
| critico | 3 | 14830 | 5372 | 17.6 | 0 |
| planificador | 1 | 1010 | 1807 | 5.8 | 0 |

**Qué decidió el código / los guardrails**

| agente | decisión | motivo |
|---|---|---|
| indexador | indice_notas | en caché: /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/Taller3-opcionB/cache/graphrag/notas-277d8a6398f1/indice.json |
| indexador | capa2 | 26 entidades del enunciado {'metodo': 8, 'concepto': 10, 'restriccion': 6, 'metrica': 2}; 7 fusionadas con las notas (p. ej. ['distribución  |
| indexador | grafo_guardado | /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/results_parte1/corridas/base-r1/tarea-B/grafo.json |
| planificador | plan_valido | 4 subtareas: T1(calculo), T2(calculo), T3(calculo), T4(conceptual) |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 1', 's1-transformers-y-foundation-models § 4.6 Inference: temperatura y sampling', 'comunidad 8', 'comunidad 0'] |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 2', 'Parte 1', 's1-transformers-y-foundation-models § 4.6 Inference: temperatura y sampling', 's1-transformers-y |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 3', 'Parte 1', 'Parte 2', 's1-transformers-y-foundation-models § 3. Arquitectura Transformer (90 min) … 3.4 Estr |
| critico | aprobado |  |
| orquestador | conceptual_al_redactor |  |
| procedencia | publicable |  |

**Ejecuciones del sandbox:** 4 · con returncode≠0: 0

**Una llamada representativa** (redactor, fin=`stop`, error=`None`)

- recibió (últimos 500 car.): `ENUNCIADO: Tarea B — Temperatura, top-p y entropía de la distribución de salida MMIA 6013 · Tarea de práctica para el solver del Taller 03 v2 Entrega: un notebook de Jupyter (.ipynb) ejecutado, con las salidas visibles. Un modelo de lenguaje produce, para el siguiente token, un vector de logits. Considera un vocabulario de seis tokens, t0 a t5, con estos logits: Token t0 t1 t2 t3 t4 t5 Logit 2.0 1.0 0.5 0.2 -1.0 -3.0 Parte 1 — Softmax con temperatura Implementa en NumPy la función softmax con te`
- respondió (primeros 500 car.): `## Parte 1 — Softmax con temperatura y entropía  La softmax con temperatura se implementa de forma numéricamente estable restando el máximo de los logits antes de exponenciar: p_i = exp((z_i − m)/T) / Σ_j exp((z_j − m)/T), con m = max_j z_j. Esto evita desbordamientos sin alterar el resultado, porque el factor común exp(m/T) se cancela entre numerador y denominador. Con los logits dados (t0 = 2.0, t1 = 1.0, t2 = 0.5, t3 = 0.2, t4 = −1.0, t5 = −3.0) se obtienen las siguientes distribuciones, cada`

**Spans exportados de Langfuse** (ligados por: duración (±5 s))

| span | nombre | n | segundos | errores |
|---|---|---|---|---|
| corrida | corrida | 1 | 147.1 | 0 |
| agent | programador | 3 | 83.1 | 0 |
| llm | programador | 3 | 83.1 | 0 |
| llm | indexador | 4 | 51.2 | 0 |
| agent | redactor | 2 | 21.6 | 0 |
| llm | redactor | 1 | 19.4 | 0 |
| agent | critico | 3 | 17.6 | 0 |
| llm | critico | 3 | 17.6 | 0 |
| agent | indexador | 1 | 16.1 | 0 |
| agent | planificador | 1 | 5.8 | 0 |
| llm | planificador | 1 | 5.8 | 0 |
| agent | ejecutor | 3 | 2.1 | 0 |
| herramienta | sandbox_python | 3 | 2.1 | 0 |
| agent | investigador | 3 | 0.6 | 0 |

traza de Langfuse: `a60f37b747502cd133ad102ae2c4dd88`

**Pistas de la traza para decidir de qué parte falló** (son pistas, la decisión es tuya):

- ninguna pista automática: lee la llamada representativa y los spans.

**Parte que falló (completa tú):** golden set / prompt / herramienta / guardrail / juez / modelo

## Caso: tarea C — 8/8 del verificador

status `completado` · 468.5 s · tokens 55141/142925

**Comprobaciones que fallan**

_Ninguna._

**Qué llamó el modelo** (traza local `traza.jsonl`)

| agente | llamadas | tokens ent. | tokens sal. | segundos | con error |
|---|---|---|---|---|---|
| programador | 4 | 19006 | 115529 | 388.4 | 1 |
| indexador | 4 | 1520 | 11735 | 51.7 | 0 |
| redactor | 1 | 9671 | 6404 | 21.1 | 0 |
| planificador | 1 | 1313 | 5268 | 18.9 | 0 |
| critico | 3 | 23631 | 3989 | 14.4 | 0 |

**Qué decidió el código / los guardrails**

| agente | decisión | motivo |
|---|---|---|
| indexador | capa2 | 28 entidades del enunciado {'metrica': 7, 'metodo': 7, 'concepto': 5, 'dataset': 4, 'restriccion': 5}; 5 fusionadas con las notas (p. ej. [' |
| indexador | grafo_guardado | /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/results_parte1/corridas/base-r1/tarea-C/grafo.json |
| planificador | plan_valido | 4 subtareas: T1(calculo), T2(calculo), T3(calculo), T4(conceptual) |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 1', 's2-rag-y-vector-search § 5. Retrieval: Sparse, Dense, Hybrid … 5.4 Reranking', 's2-rag-y-vector-search § 6. |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 2', 'Parte 1', 's2-rag-y-vector-search § 5. Retrieval: Sparse, Dense, Hybrid … 5.4 Reranking', 's2-rag-y-vector- |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 3'] |
| critico | aprobado |  |
| orquestador | conceptual_al_redactor |  |
| redactor | pdf_generado | 2 páginas |
| procedencia | publicable |  |

**Ejecuciones del sandbox:** 3 · con returncode≠0: 0

**Una llamada representativa** (programador, fin=`length`, error=`contenido vacío (fin=length)`)

- recibió (últimos 500 car.): `SUBTAREA T3: A partir de los rankings de ambos métodos, identificar las consultas donde cada método falla (documento relevante fuera del top 3) y extraer los términos que comparte cada consulta con su documento relevante (y con los documentos que la superan en el ranking), para sustentar la explicación de los fallos. CRITERIO DE ÉXITO: Tabla de fallos por método (consultas con relevante fuera del top 3, con q5 entre los resultados esperados) y tabla de términos compartidos consulta–documento rel`
- respondió (primeros 500 car.): ``

**Spans exportados de Langfuse** (ligados por: duración (±5 s))

| span | nombre | n | segundos | errores |
|---|---|---|---|---|
| corrida | corrida | 1 | 468.3 | 0 |
| agent | programador | 3 | 388.4 | 0 |
| llm | programador | 4 | 388.4 | 0 |
| llm | indexador | 4 | 51.7 | 0 |
| agent | redactor | 2 | 21.2 | 0 |
| llm | redactor | 1 | 21.1 | 0 |
| agent | indexador | 1 | 19.3 | 0 |
| agent | planificador | 1 | 18.9 | 0 |
| llm | planificador | 1 | 18.9 | 0 |
| agent | critico | 3 | 14.4 | 0 |
| llm | critico | 3 | 14.4 | 0 |
| agent | ejecutor | 3 | 5.2 | 0 |
| herramienta | sandbox_python | 3 | 5.2 | 0 |
| agent | investigador | 3 | 0.8 | 0 |

traza de Langfuse: `8925f721ddd1c7a4f02a0e9591781878`

**Pistas de la traza para decidir de qué parte falló** (son pistas, la decisión es tuya):

- llamadas con contenido vacío: parte **modelo** (razona hasta el límite) o **prompt** (pide demasiado)

**Parte que falló (completa tú):** golden set / prompt / herramienta / guardrail / juez / modelo

## Caso: tarea I1 — 7/7 del verificador

status `completado` · 332.7 s · tokens 11657/15174

**Comprobaciones que fallan**

_Ninguna._

**Qué llamó el modelo** (traza local `traza.jsonl`)

| agente | llamadas | tokens ent. | tokens sal. | segundos | con error |
|---|---|---|---|---|---|
| indexador | 2 | 794 | 4891 | 116.0 | 0 |
| programador | 2 | 6409 | 5533 | 81.9 | 0 |
| redactor | 1 | 1117 | 2225 | 74.7 | 0 |
| planificador | 1 | 753 | 1471 | 34.5 | 0 |
| critico | 1 | 2584 | 1054 | 10.0 | 0 |

**Qué decidió el código / los guardrails**

| agente | decisión | motivo |
|---|---|---|
| lector | leido | 4 secciones; de trabajo: ['Parte 1', 'Parte 2']; 1 tablas; restricciones: {'entregable': 'reporte.md', 'formato': 'md', 'palabras_max': 150, |
| indexador | esqueleto | sin aristas depende_de |
| indexador | indice_notas | en caché: /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/Taller3-opcionB/cache/graphrag/notas-277d8a6398f1/indice.json |
| indexador | capa2 | 9 entidades del enunciado {'restriccion': 2, 'concepto': 2, 'metrica': 4, 'dataset': 1}; 1 fusionadas con las notas (p. ej. ['latencia']); 4 |
| indexador | grafo_guardado | /home/dludena/Taller4_IA_Generativa_Agentes/Lab-04-LLMOps/results_parte2/base/corridas/tarea-I1/grafo.json |
| planificador | plan_valido | 2 subtareas: T1(calculo), T2(conceptual) |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Parte 1', 's2-rag-y-vector-search § 3. Vector Databases … 3.4 Stack recomendado', 'comunidad 10'] |
| critico | rechazado | - guarda: el código no compila: unterminated triple-quoted f-string literal (detected at line 80) (línea 60) |
| critico | aprobado |  |
| orquestador | conceptual_al_redactor |  |
| procedencia | publicable |  |

**Ejecuciones del sandbox:** 2 · con returncode≠0: 0

**Una llamada representativa** (redactor, fin=`stop`, error=`None`)

- recibió (últimos 500 car.): `ENUNCIADO: Tarea I1 — Resumen estadístico de latencias de un servicio MMIA 6013 · Tarea del golden set del grupo (Taller 04, Parte 2.a) Entrega: un reporte en Markdown (reporte.md), con el código que lo produce. Un servicio de inferencia registró la latencia, en milisegundos, de ocho llamadas consecutivas: llamada 1 2 3 4 5 6 7 8 latencia (ms) 120 135 128 150 142 138 160 127 Parte 1 — Resumen Calcula la media, la desviación estándar muestral (con n − 1) y el error estándar de la media de las och`
- respondió (primeros 500 car.): `## Datos  Ocho latencias consecutivas del servicio de inferencia, en milisegundos: 120, 135, 128, 150, 142, 138, 160, 127 (n = 8).  ## Resultados  | Estadístico | Valor (ms) | |---|---| | Media | 137.50 | | Desviación estándar muestral (n − 1) | 13.07 | | Error estándar de la media (s/√n) | 4.62 |  ![Resumen estadístico de las ocho latencias](figuras/T1_latencias_resumen.png)  ## Conclusión  La desviación estándar (13.07 ms) describe la dispersión de las latencias individuales alrededor de la me`

**Spans exportados de Langfuse** (ligados por: traza_id.txt)

| span | nombre | n | segundos | errores |
|---|---|---|---|---|
| corrida | corrida | 1 | 323.2 | 0 |
| llm | indexador | 2 | 116.0 | 0 |
| agent | indexador | 1 | 113.5 | 0 |
| agent | programador | 2 | 81.9 | 0 |
| llm | programador | 2 | 81.9 | 0 |
| agent | redactor | 2 | 74.8 | 0 |
| llm | redactor | 1 | 74.7 | 0 |
| agent | planificador | 1 | 34.5 | 0 |
| llm | planificador | 1 | 34.5 | 0 |
| agent | critico | 2 | 10.0 | 0 |
| llm | critico | 1 | 10.0 | 0 |
| agent | investigador | 1 | 7.7 | 0 |
| agent | ejecutor | 2 | 0.6 | 0 |
| herramienta | sandbox_python | 1 | 0.6 | 0 |

traza de Langfuse: `2a59d0ad28e8c08dc2d9b367fdaf709a`

**Pistas de la traza para decidir de qué parte falló** (son pistas, la decisión es tuya):

- ninguna pista automática: lee la llamada representativa y los spans.

**Parte que falló (completa tú):** golden set / prompt / herramienta / guardrail / juez / modelo

## Caso: tarea S1 — 9/9 del verificador

status `completado` · 674.8 s · tokens 78831/212547

**Comprobaciones que fallan**

_Ninguna._

**Qué llamó el modelo** (traza local `traza.jsonl`)

| agente | llamadas | tokens ent. | tokens sal. | segundos | con error |
|---|---|---|---|---|---|
| programador | 7 | 22295 | 170481 | 558.0 | 2 |
| indexador | 5 | 2320 | 15758 | 77.6 | 0 |
| critico | 5 | 26542 | 12489 | 43.8 | 0 |
| redactor | 2 | 25776 | 10334 | 33.9 | 0 |
| planificador | 1 | 1898 | 3485 | 11.1 | 0 |

**Qué decidió el código / los guardrails**

| agente | decisión | motivo |
|---|---|---|
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Tarea 4', 's3-agentes § 3. Patrones de Agentes … 3.3 Reflexion', 'comunidad 6'] |
| critico | aprobado |  |
| investigador | grafo+capa2 | citas: ['Preámbulo', 'Tarea 5', 'Tarea 3'] |
| programador | tope_de_razonamiento | vacío por longitud con max_tokens=65536, que ya es el tope: repetirlo costaría lo mismo sin otro resultado |
| programador | llm_fallo | LLMVacio: programador: contenido vacío por longitud en el tope (65536 tokens) |
| critico | rechazado | - sin script: el modelo no respondió en este intento |
| critico | aprobado |  |
| redactor | archivos_exigidos | publicados: ['output/server_analysis.parquet']; ninguna ejecución aprobada escribió: [] |
| procedencia | devuelto_al_redactor | estas cifras no salen de ninguna ejecución aprobada ni del enunciado; corrígelas con el valor medido o quítalas: ['0.1633'] |
| redactor | archivos_exigidos | publicados: ['output/server_analysis.parquet']; ninguna ejecución aprobada escribió: [] |
| procedencia | publicable |  |

**Ejecuciones del sandbox:** 5 · con returncode≠0: 0

**Una llamada representativa** (programador, fin=`length`, error=`contenido vacío (fin=length)`)

- recibió (últimos 500 car.): `SUBTAREA T5: Guardar el DataFrame analizado de la Tarea 3 en output/server_analysis.parquet mediante PyArrow sin el índice, volver a leerlo y verificar como mínimo: filas antes == filas después, columnas antes == columnas después, esquema antes == esquema después y valores iguales. CRITERIO DE ÉXITO: Existe output/server_analysis.parquet; las cuatro verificaciones del round trip (Filas, Columnas, Esquema, Valores) son True y se imprime 'Round trip verificado: output/server_analysis.parquet'.  CO`
- respondió (primeros 500 car.): ``

**Spans exportados de Langfuse** (ligados por: duración (±5 s))

| span | nombre | n | segundos | errores |
|---|---|---|---|---|
| corrida | corrida | 1 | 674.5 | 0 |
| agent | programador | 6 | 558.1 | 0 |
| llm | programador | 7 | 558.1 | 0 |
| llm | indexador | 5 | 77.6 | 0 |
| agent | critico | 6 | 43.9 | 0 |
| llm | critico | 5 | 43.8 | 0 |
| agent | redactor | 4 | 34.0 | 0 |
| llm | redactor | 2 | 33.9 | 0 |
| agent | indexador | 1 | 24.1 | 0 |
| agent | planificador | 1 | 11.1 | 0 |
| llm | planificador | 1 | 11.1 | 0 |
| agent | ejecutor | 6 | 2.1 | 0 |
| herramienta | sandbox_python | 5 | 2.1 | 0 |
| agent | investigador | 5 | 1.1 | 0 |

traza de Langfuse: `feef2fd3ffd71c162db6b86412af757a`

**Pistas de la traza para decidir de qué parte falló** (son pistas, la decisión es tuya):

- llamadas con contenido vacío: parte **modelo** (razona hasta el límite) o **prompt** (pide demasiado)

**Parte que falló (completa tú):** golden set / prompt / herramienta / guardrail / juez / modelo

