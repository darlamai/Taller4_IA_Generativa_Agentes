# Antes / después de la mejora

Mismo golden set, mismo evaluador y mismo modelo. Comprobaciones del verificador comparadas: **61**.

| | aprobadas | de 61 |
|---|---|---|
| base | 50 | 0.820 |
| mejora | 60 | 0.984 |
| diferencia | +10 | +0.164 |

Cambiaron **10** comprobaciones: 10 mejoraron, 0 empeoraron.

**Ruido del modelo:** entre la corrida base y la de control (misma versión, mismo golden) cambiaron **10 de 61** comprobaciones (16.4%) sin tocar nada: eso es azar del muestreo.

## Qué cambió, caso por caso

| tarea | id | tipo | base | mejora | cambio | ¿cambia también sin mejora? | ¿era el objetivo? |
|---|---|---|---|---|---|---|---|
| S2 | S201 | archivo | 0 | 1 | mejoró | sí | sí |
| S2 | S204 | secciones | 0 | 1 | mejoró | sí | sí |
| S2 | S205 | cifra_presente | 0 | 1 | mejoró | sí | sí |
| S2 | S206 | cifra_presente | 0 | 1 | mejoró | sí | sí |
| S2 | S207 | cifra_presente | 0 | 1 | mejoró | sí | sí |
| S2 | S208 | cifra_presente | 0 | 1 | mejoró | sí | sí |
| S2 | S209 | contiene | 0 | 1 | mejoró | sí | sí |
| S2 | S210 | cifra_presente | 0 | 1 | mejoró | sí | sí |
| S2 | S211 | cifra_presente | 0 | 1 | mejoró | sí | sí |
| S2 | S212 | procedencia | 0 | 1 | mejoró | sí | sí |

## Separar lo buscado de lo que cambió solo por azar

- **Lo que la mejora buscaba** (comprobaciones objetivo que cambiaron y que NO cambian solas en el control): ninguna
- **Cambios que también ocurren sin mejora** (comprobaciones inestables, atribuibles al azar): ['S201', 'S204', 'S205', 'S206', 'S207', 'S208', 'S209', 'S210', 'S211', 'S212']
- **Cambios fuera del objetivo y estables en el control** (efecto colateral, regresión o azar no visto): ninguno

Con una sola corrida por versión y este número de comprobaciones, una diferencia pequeña no se distingue del azar: dilo así en el informe.

## Costo y tiempo por tarea

| tarea | status base | status mejora | tokens base | tokens mejora | s base | s mejora |
|---|---|---|---|---|---|---|
| A | completado | completado | 81239 | 72534 | 168.4 | 133.6 |
| B | completado | completado | 84989 | 66857 | 147.3 | 96.6 |
| C | completado | completado | 198066 | 314447 | 468.5 | 688.2 |
| I1 | completado | completado | 26831 | 27367 | 332.7 | 43.9 |
| P1 | completado | completado | 13654 | 12582 | 354.8 | 17.4 |
| S1 | completado | completado | 291378 | 389226 | 674.8 | 840.7 |
| S2 | fallido | parcial | 520683 | 478971 | 1281.2 | 919.5 |

## Lo que dice el juez (solo comprobaciones juzgadas en las dos versiones)

| tarea | id | tipo | juez base | juez mejora |
|---|---|---|---|---|
| — | — | — | sin cambios | — |
