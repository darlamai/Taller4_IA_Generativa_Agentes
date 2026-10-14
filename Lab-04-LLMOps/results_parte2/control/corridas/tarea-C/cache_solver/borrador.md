## Objetivo

Comparar la recuperación léxica (TF-IDF + similitud coseno) con la semántica latente (LSA mediante TruncatedSVD a 4 dimensiones) sobre un corpus de 10 documentos y 6 consultas con un documento relevante cada una, evaluando Hit@1, Hit@3 y MRR. Se identifican las consultas que cada método falla, se explica el fallo mediante el solapamiento léxico entre consulta y documento relevante, y se discute qué aportaría un recuperador denso en el caso límite (q5) y a qué costo.

## Método

**Corpus y juicios:** 10 documentos (d01–d10) y 6 consultas (q1–q6), cada una con su relevante: q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02.

**Línea base léxica:** TfidfVectorizer de scikit-learn con parámetros por defecto (vocabulario de 105 términos) y similitud coseno consulta–documento; se ordenan los 10 documentos por consulta.

**Semántica latente:** la misma matriz TF-IDF se proyecta con TruncatedSVD(n_components=4, random_state=0), ajustado solo con los 10 documentos; las consultas se proyectan con el mismo vectorizador y svd.transform, y se rankea por coseno en el espacio latente. Las 4 componentes explican una varianza acumulada de 0.3953.

**Evaluación:** Hit@1, Hit@3 y MRR sobre las 6 consultas; fallo = relevante fuera del top 3. Solapamiento léxico: intersección de tokens únicos con el analizador por defecto (minúsculas, tokens de 2+ caracteres, sin stop words ni stemming).

## Resultados

**Tabla 1 — Métricas agregadas (Parte 2):**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (léxico) | 0.8333 | 0.8333 | 0.8611 |
| LSA (4 dimensiones) | 0.6667 | 0.8333 | 0.75 |

![Comparación de métricas TF-IDF frente a LSA](figuras/T2_comparacion_metricas_tfidf_lsa.png)

**Tabla 2 — Puesto del relevante por consulta (Parte 3):**

| Consulta | Relevante | Rank TF-IDF | Rank LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

TF-IDF falla solo q5 (rank 6); LSA también falla solo q5 (rank 6), pero además degrada q4 al puesto 3 y pierde Hit@1: d05 queda con similitud 0.8984 frente a d01 (0.9220) y d10 (0.9206).

En q5, la consulta comparte con d09 un único término, «la» (df 6: aparece en 6 de los 10 documentos, poco discriminativo), y 7 de sus tokens están fuera del vocabulario: determinista, elegir, hace, más, palabra, sea, siguiente. El único término de contenido presente en el vocabulario, «modelo», aparece en d03, que encabeza el ranking TF-IDF con 0.3107 (d06 0.2009, d04 0.1896), mientras d09 alcanza solo 0.0790. En LSA, d09 llega a 0.4999 frente a d04 con 0.9427. En contraste, q1 comparte 8 términos con d04 (cobertura 0.8889) y ambos métodos la resuelven en el puesto 1.

![Términos compartidos entre cada consulta y su documento relevante](figuras/T3_T3_terminos_compartidos.png)

## Discusión

El fallo de q5 es de cobertura léxica, no de ranking: d09 habla de «temperatura», «logits», «softmax» y «probabilidad», términos que no aparecen en la consulta, cuyo contenido («determinista», «siguiente palabra») no está en ningún documento. Sin solapamiento, TF-IDF solo puede apoyarse en palabras functionales («la», df 6) y en «modelo», que arrastra a d03 al primer puesto. LSA no remedia el problema: con 4 componentes que capturan 0.3953 de la varianza, solo recombina co-ocurrencias presentes en el corpus, y al no existir señal léxica para q5–d09, el relevante sigue en el puesto 6; además, la compresión difumina distancias finas, como muestra q4, donde d05 cae al puesto 3 por detrás de d01 y d10.

**Recuperador denso (Parte 4):** un codificador entrenado en grandes corpus aprendería que «más determinista al elegir la siguiente palabra» y «valores bajos de temperatura concentran la probabilidad» expresan la misma relación semántica (determinismo ↔ baja temperatura), colocando q5 cerca de d09 sin necesidad de solapamiento de tokens; resolvería así la única consulta donde ambos métodos de esta tarea fallan (rank 6 en los dos). El costo frente a TF-IDF es sustancial: requiere un modelo preentrenado, cómputo de codificación tanto en la indexación como en cada consulta (mayor latencia), memoria para los pesos del modelo y los vectores densos, menor interpretabilidad del ranking y riesgo de degradación fuera del dominio de entrenamiento; TF-IDF no necesita entrenamiento, es barato y explicable término a término, como evidencian las 8 coincidencias léxicas de q1.
