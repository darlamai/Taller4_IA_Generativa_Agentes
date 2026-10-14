## Datos

Latencias (ms) registradas en ocho llamadas consecutivas a un servicio de inferencia:

| Llamada | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Latencia (ms) | 120 | 135 | 128 | 150 | 142 | 138 | 160 | 127 |

n = 8.

## Resultados

Código que produce las cifras:

```python
import numpy as np
x = np.array([120,135,128,150,142,138,160,127.])
print(round(x.mean(),2), round(x.std(ddof=1),2), round(x.std(ddof=1)/np.sqrt(x.size),2))
```

| Estadístico | Valor (ms) |
|---|---|
| Media | 137.50 |
| Desviación estándar muestral (n − 1) | 13.07 |
| Error estándar de la media | 4.62 |

## Conclusión

La desviación estándar (13.07 ms) mide la dispersión de las latencias individuales respecto a la media (137.50 ms). El error estándar (4.62 ms) mide la precisión de esa media como estimación del promedio del servicio, es decir, cuánto variaría entre muestras de ocho llamadas.
