## Introducción

Ng y Jordan (2002) predijeron que un clasificador generativo alcanza su error asintótico con menos ejemplos que su par discriminativo, aunque ese error asintótico sea mayor. Esta tarea comprueba esa predicción sobre datos reales del conjunto Breast Cancer Wisconsin (Diagnostic), comparando un Naive Bayes gaussiano (generativo) contra una regresión logística (discriminativa) en una curva de aprendizaje sobre una única división entrenamiento/prueba.

## Metodología

Se cargó el conjunto con `sklearn.datasets.load_breast_cancer` (569 ejemplos, 30 atributos; 212 casos malignos y 357 benignos). Se creó una división 70 %/30 % estratificada por clase con `random_state=42`, que fue la única usada en toda la tarea: 398 ejemplos de entrenamiento (148 malignos, 250 benignos) y 171 de prueba (64 malignos, 107 benignos); las proporciones resultantes fueron 0.6995 y 0.3005. El conjunto de prueba no se usó para ajustar nada.

Se entrenaron dos modelos: un GaussianNB y una LogisticRegression (solver lbfgs, `random_state=42`). Para la regresión logística los atributos se estandarizaron con `StandardScaler` ajustado solo con el entrenamiento (verificado: la media del escalador coincide exactamente con la del entrenamiento y difiere de la de todos los datos, con diferencia máxima 4.1049). Para la curva de aprendizaje se reentrenaron ambos modelos con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del entrenamiento (19, 39, 99, 199 y 398 ejemplos; `random_state=42`), evaluando siempre sobre el mismo conjunto de prueba de 171 ejemplos y transformándolo con el mismo escalador. Se midió exactitud (accuracy) y F1 macro.

## Resultados

![Distribución de clases del conjunto Breast Cancer Wisconsin](figuras/T1_distribucion_clases.png)

Con el entrenamiento completo (398 ejemplos), las métricas sobre el conjunto de prueba fueron:

| Modelo | Accuracy (prueba) | F1 macro (prueba) |
|---|---|---|
| Naive Bayes gaussiano | 0.9357 | 0.9307 |
| Regresión logística | 0.9883 | 0.9875 |

![Tabla de métricas de la Parte 2](figuras/T2_tabla_parte2_metricas.png)

La curva de aprendizaje, con la exactitud sobre el conjunto de prueba fijo, fue:

| % del entrenamiento | n ejemplos | Accuracy NB | Accuracy RL |
|---|---|---|---|
| 5 | 19 | 0.9415 | 0.9357 |
| 10 | 39 | 0.9474 | 0.9357 |
| 25 | 99 | 0.9357 | 0.9474 |
| 50 | 199 | 0.9240 | 0.9766 |
| 100 | 398 | 0.9357 | 0.9883 |

![Curva de aprendizaje: exactitud contra número de ejemplos de entrenamiento, una curva por modelo](figuras/T3_curva_aprendizaje_parte3.png)

El Naive Bayes gaussiano tuvo mayor exactitud con 19 y 39 ejemplos (0.9415 y 0.9474 frente a 0.9357 de la regresión logística); la regresión logística superó al Naive Bayes a partir de 99 ejemplos (0.9474 frente a 0.9357) y amplió la ventaja hasta 0.9883 frente a 0.9357 con los 398 ejemplos.

## Discusión

Los datos reproducen el cruce que predicen Ng y Jordan. Con muy pocos ejemplos (19 y 39, es decir, 5 % y 10 % del entrenamiento) el modelo generativo gana: 0.9415 contra 0.9357 y 0.9474 contra 0.9357. El cruce ocurre entre 39 y 99 ejemplos: con 99 (25 %) la regresión logística ya adelanta al Naive Bayes (0.9474 contra 0.9357) y con 199 y 398 ejemplos la ventaja discriminativa crece con claridad (0.9766 contra 0.9240 y 0.9883 contra 0.9357).

La forma de las curvas ilustra el papel del sesgo y la varianza. El Naive Bayes gaussiano impone un sesgo fuerte —asume independencia condicional de los 30 atributos dada la clase, algo falso en este conjunto, donde atributos como radio, perímetro y área están correlacionados— pero a cambio tiene varianza baja: su exactitud se mueve en una banda estrecha (0.9240–0.9474) desde 19 ejemplos y se estabiliza pronto cerca de su error asintótico, que además es peor (0.9357 con 398 ejemplos). La regresión logística no impone esa suposición de independencia: tiene menor sesgo asintótico y de hecho alcanza la mejor exactitud de la tarea (0.9883), pero sus parámetros se estiman con mayor varianza, por lo que con 19 y 39 ejemplos rinde igual (0.9357) y solo aprovecha los datos adicionales gradualmente: 0.9474 con 99, 0.9766 con 199 y 0.9883 con 398, una mejora monótona.

En términos de Ng y Jordan: el generativo converge más rápido (su curva es casi plana desde el inicio) hacia un error asintótico mayor; el discriminativo converge más lentamente hacia un error asintótico menor. La recomendación práctica que se desprende de estas cifras es usar el modelo generativo cuando el presupuesto de etiquetas es muy escaso (menos de unos 99 ejemplos aquí) y el discriminativo cuando hay datos suficientes.

## Conclusiones

Sobre la división fija 70/30 (398/171 ejemplos) del conjunto Breast Cancer Wisconsin, la regresión logística estandarizada superó al Naive Bayes gaussiano con el entrenamiento completo (accuracy 0.9883 contra 0.9357; F1 macro 0.9875 contra 0.9307). La curva de aprendizaje mostró el cruce predicho por Ng y Jordan: el generativo fue mejor con 19 y 39 ejemplos, el discriminativo desde 99 en adelante. El comportamiento es consistente con la teoría: el sesgo fuerte del Naive Bayes le da baja varianza y convergencia rápida a un error asintótico mayor, mientras que la regresión logística, con menor sesgo, necesita más datos para desplegar su ventaja asintótica.
