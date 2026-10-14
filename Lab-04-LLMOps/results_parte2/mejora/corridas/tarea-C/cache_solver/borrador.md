## Objetivo

Comparar la recuperación léxica (TF-IDF + similitud coseno) contra la semántica latente (LSA mediante TruncatedSVD a 4 dimensiones) sobre un corpus de 10 documentos y 6 consultas con un documento relevante cada una, midiendo Hit@1, Hit@3 y MRR. Se identifican las consultas donde cada método falla, se explican los fallos mediante los términos compartidos entre consulta y documento relevante, y se discute qué aportaría un recuperador denso frente a ambos.

## Método

Corpus: d01–d10; consultas q1–q6 con juicios q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02.

1. **TF-IDF (línea base):** `TfidfVectorizer` con parámetros por defecto, ajustado solo sobre los 10 documentos (vocabulario de 105 términos); similitud coseno consulta-documento y ordenamiento de los 10 documentos por consulta.
2. **LSA:** `TruncatedSVD(n_components=4, random_state=0)` ajustado sobre la misma matriz TF-IDF; las consultas se proyectan con `transform` (sin reajuste) y se recalcula la similitud coseno en el espacio latente de 4 dimensiones.
3. **Evaluación:** Hit@1, Hit@3 y MRR sobre las 6 consultas. Fallo: relevante fuera del top 3. Análisis de términos compartidos: intersección de vocabulario y pesos TF-IDF entre cada consulta y su relevante.

## Resultados

**Tabla comparativa (Parte 2):**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (línea base) | 0.8333 | 0.8333 | 0.8611 |
| LSA (TruncatedSVD, 4D) | 0.6667 | 0.8333 | 0.7500 |

LSA reduce Hit@1 en 0.1667 y MRR en 0.1111, y deja Hit@3 sin cambio. La varianza explicada por los 4 componentes es 0.0342, 0.1266, 0.1160 y 0.1186 (total 0.3953).

**Fallos por consulta (Parte 3):**

| Consulta | Relevante | Rank TF-IDF | Rank LSA | Términos compartidos (n) |
|---|---|---|---|---|
| q1 | d04 | 1 | 1 | 8 (frecuencia, léxica, ranking, términos, función, pondera, de, la) |
| q2 | d03 | 1 | 1 | 3 (al, fragmentos, prompt) |
| q3 | d08 | 1 | 1 | 5 (bajo, entrena, matrices, rango, de) |
| q4 | d05 | 1 | 3 | 4 (compara, orientación, vectores, la) |
| q5 | d09 | 6 | 6 | 1 (la) |
| q6 | d02 | 1 | 1 | 5 (autoatención, capas, feed, forward, con) |

Ambos métodos fallan solo en q5. En las cinco consultas resueltas, consulta y relevante comparten entre 3 y 8 términos, varios discriminativos (df 1, idf 2.7047: frecuencia, fragmentos, matrices, orientación, autoatención), y la similitud TF-IDF del relevante va de 0.4569 (q4) a 0.6891 (q1). En q4, LSA baja del puesto 1 al 3: en el espacio latente d01 (0.9220) y d10 (0.9206) superan a d05 (0.8984).

![Métricas Hit@1, Hit@3 y MRR de TF-IDF frente a LSA](figuras/T2_metricas_tfidf_vs_lsa.png)

![Términos compartidos entre q5 y su documento relevante d09](figuras/T3_T3_q5_terminos.png)

## Discusión

El léxico basta cuando la consulta reutiliza las palabras del documento: en q1–q4 y q6 los términos compartidos incluyen términos únicos del corpus (df 1, idf 2.7047) que dominan el coseno y colocan al relevante en el puesto 1 en ambos métodos.

**q5 falla en los dos métodos** porque consulta y d09 comparten un único término, "la" (df 6, idf 1.4520; peso 0.2440 en la consulta y 0.3238 en el documento), prácticamente no discriminativo. La similitud TF-IDF de d09 es 0.0790, muy por debajo de la de d03 (0.3107), que encabeza el ranking por coincidencias superficiales. d09 expresa la idea con otro vocabulario (temperatura, logits, softmax, probabilidad), sin solape léxico con "determinista" o "palabra".

**LSA no lo corrige:** d09 queda en el puesto 6 con similitud latente 0.4999. Con solo 10 documentos, los 4 componentes capturan 0.3953 de la varianza y solo pueden agrupar términos que coocurren en el corpus; como "determinista" no coocurre con "temperatura" en ningún documento, no existe puente latente que los conecte. La proyección además introduce ruido: en q4 el relevante cae al puesto 3 aun con similitud latente alta (0.8984).

**Recuperador denso (Parte 4):** un modelo de embeddings entrenado resolvería q5 mejor porque su preentrenamiento sobre grandes corpus aprende relaciones de sinonimia y paráfrasis —que "más determinista" corresponde a "temperatura baja concentra la probabilidad"— y colocaría a q5 y d09 cerca aunque no compartan ningún término de contenido, que es justo el caso que TF-IDF (similitud 0.0790) y LSA (0.4999, mismo puesto 6) no resuelven. El costo frente a TF-IDF es considerable: requiere datos y cómputo de entrenamiento (o un modelo preentrenado), inferencia con una red neuronal para codificar cada consulta y documento, índices de vectores densos con mayor memoria y mayor latencia; TF-IDF se ajusta sin supervisión sobre la colección, es barato e interpretable, y aquí alcanza Hit@3 0.8333, pero es ciego a la sinonimia.
