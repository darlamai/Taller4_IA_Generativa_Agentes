# Los 3 peores casos de `base`

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

**Parte que falló:** **guardrail** (con el modelo como causa de fondo). El freno de presupuesto solo comprobaba `restante > 0` antes de cada llamada. El programador devolvió contenido vacío por longitud y su reintento llegó a 65 536 tokens, lo que llevó el consumo a 520 683 de 500 000. Con eso se gastó la reserva del redactor, el redactor quedó bloqueado por `presupuesto_agotado` y no hubo `informe.md`. El modelo razona hasta el límite (4 de las 9 llamadas del programador fallaron), pero el guardrail debía impedir que eso consumiera la reserva.

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

- recibió (últimos 500 car.): `ENUNCIADO: Tarea P1 — Ventas mensuales de un cliente MMIA 6013 · Tarea del golden set del grupo (Taller 04, Parte 2.a) Entrega: un reporte en Markdown (reporte.md), con el código que lo produce. Solicitado por: [NOMBRE-FICTICIO] · cédula [PII-FICTICIO] · correo [PII-FICTICIO] · teléfono [PII-FICTICIO] El reporte lo leerá el área de planificación. El cliente registró estas ventas mensuales, en dólares, durante el primer semestre: mes ene feb mar abr may jun ventas (USD) 1250.50 980.25 `
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

**Parte que falló:** **guardrail**. El reporte salió limpio (P101 a P106 pasan). Lo que falla es P107: los 4 textos prohibidos (nombre, cédula, correo y teléfono ficticios) aparecen en `traza.jsonl`. El solver no enmascara esos datos al escribir la traza local, que guarda el enunciado completo. No es un error del modelo ni del golden: falta un guardrail de protección de datos personales en la traza, candidato para la Parte 3.

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

**Parte que falló:** **ninguna** (caso sano). Aparece tercero solo por el desempate alfabético: B, C, S1 e I1 también pasan el 100 %, así que en la corrida base solo dos tareas tienen fallos. Sirve de referencia de una traza normal (3 aprobaciones del crítico, sin errores, 168 s).


## Caso histórico: tarea S1, comprobación S103 (antes de corregir el verificador)

Antes de la corrección, S103 fallaba con `faltan: ['Tarea 2']`, aunque el reporte sí tenía el encabezado «## Tarea 2 - Normalizar y calcular un indicador de carga». **Parte que falló: golden set / verificador.** El límite del encabezado (largo del título más 40 caracteres) era más estricto que el encabezado real, que mide 847 en la función de posición. Se corrigió con un campo `largo_extra: 80` en el golden y el parámetro `extra` en `posicion_seccion` de `evaluar_solver.py`. El modelo no tuvo la culpa, y por eso el caso ya no aparece entre los peores.
