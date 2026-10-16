## Datos

Ocho llamadas consecutivas de un servicio de inferencia; latencia en ms: 120, 135, 128, 150, 142, 138, 160, 127 (n = 8).

```python
import numpy as np
x = np.array([120, 135, 128, 150, 142, 138, 160, 127])
print(x.mean(), x.std(ddof=1), x.std(ddof=1)/np.sqrt(8))
```

## Resultados

| Estadístico | Valor (ms) |
|---|---|
| Media | 137.50 |
| Desviación estándar muestral (n − 1) | 13.07 |
| Error estándar de la media | 4.62 |

## Conclusión

La desviación estándar (13.07 ms) mide la dispersión de las latencias individuales alrededor de la media (137.50 ms), es decir, cuánto varía una llamada típica. El error estándar (4.62 ms) mide la precisión de esa media como estimación, es decir, cuánto fluctuaría entre muestras de ocho llamadas; es menor porque promediar reduce la variabilidad aleatoria.
