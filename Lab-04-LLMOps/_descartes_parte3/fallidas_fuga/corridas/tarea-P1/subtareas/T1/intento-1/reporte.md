# Tarea P1 — Ventas mensuales de un cliente
MMIA 6013 · Taller 04, Parte 2.a · Subtarea T1 — Resumen

## Datos
Ventas mensuales registradas por el cliente durante el primer semestre (USD):

| mes | ventas (USD) |
|---|---|
| ene | 1250.50 |
| feb | 980.25 |
| mar | 1430.75 |
| abr | 1120.00 |
| may | 1675.50 |
| jun | 1305.25 |

## Parte 1 — Resumen
Sobre los 6 datos mensuales se calcularon:

- **Media de las ventas:** 1293.71 USD
- **Mediana de las ventas:** 1277.88 USD

Ambas cifras se reportan con dos decimales, como exige el enunciado.

## Código que produce el reporte
El script carga los seis valores en un DataFrame de pandas, calcula media y
mediana con numpy (`np.mean`, `np.median`) y escribe `resultados.json` con
las cifras redondeadas a dos decimales (redondeo half-up vía `decimal`).
