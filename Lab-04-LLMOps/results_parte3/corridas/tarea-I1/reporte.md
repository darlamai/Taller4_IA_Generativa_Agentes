## Datos

Ocho latencias consecutivas del servicio de inferencia, en milisegundos: 120, 135, 128, 150, 142, 138, 160, 127 (n = 8).

```python
import numpy as np
x = np.array([120, 135, 128, 150, 142, 138, 160, 127], dtype=float)
media = x.mean()           # media
s = x.std(ddof=1)          # desviación estándar muestral (n - 1)
se = s / np.sqrt(x.size)   # error estándar de la media
```

## Resultados

| Estadístico | Valor (ms) |
| --- | --- |
| Media | 137.50 |
| Desviación estándar muestral (n − 1) | 13.07 |
| Error estándar de la media | 4.62 |

## Conclusión

La desviación estándar muestral (13.07 ms) describe la dispersión de las latencias individuales alrededor de la media de 137.50 ms. El error estándar de la media (4.62 ms) describe la precisión de esa media como estimación y se calcula dividiendo la desviación estándar entre la raíz cuadrada del tamaño muestral (n = 8).
