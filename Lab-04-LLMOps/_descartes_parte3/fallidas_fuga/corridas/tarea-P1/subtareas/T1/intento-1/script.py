"""
Subtarea T1 — Ventas mensuales de un cliente (MMIA 6013 · Taller 04, Parte 2.a)
Carga las ventas mensuales del semestre, calcula la media y la mediana y las
reporta redondeadas a dos decimales en resultados.json (contrato) y reporte.md.
"""

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ datos
MESES = ["ene", "feb", "mar", "abr", "may", "jun"]
VENTAS_USD = [1250.50, 980.25, 1430.75, 1120.00, 1675.50, 1305.25]

df = pd.DataFrame({"mes": MESES, "ventas_usd": VENTAS_USD})

# ---------------------------------------------------------- estadísticas
valores = df["ventas_usd"].to_numpy(dtype=float)
media = float(np.mean(valores))
mediana = float(np.median(valores))


def redondear_2(x: float) -> float:
    """Redondeo convencional (half-up) a exactamente dos decimales."""
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


media_2 = redondear_2(media)
mediana_2 = redondear_2(mediana)

# ------------------------------------------------------- resultados.json
resultados = {
    "subtarea": "T1_media_y_mediana_ventas",
    "ventas_mensuales_usd": VENTAS_USD,
    "meses": MESES,
    "n_observaciones": int(len(valores)),
    "media_ventas_usd": media_2,
    "mediana_ventas_usd": mediana_2,
    "media_ventas_usd_sin_redondear": media,
    "mediana_ventas_usd_sin_redondear": mediana,
    "tabla_ventas": df.to_dict(orient="records"),
}

Path("resultados.json").write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ----------------------------------------------------------- reporte.md
lineas_tabla = "\n".join(
    f"| {m} | {v:.2f} |" for m, v in zip(MESES, VENTAS_USD)
)
reporte = f"""# Tarea P1 — Ventas mensuales de un cliente
MMIA 6013 · Taller 04, Parte 2.a · Subtarea T1 — Resumen

## Datos
Ventas mensuales registradas por el cliente durante el primer semestre (USD):

| mes | ventas (USD) |
|---|---|
{lineas_tabla}

## Parte 1 — Resumen
Sobre los {len(valores)} datos mensuales se calcularon:

- **Media de las ventas:** {media_2:.2f} USD
- **Mediana de las ventas:** {mediana_2:.2f} USD

Ambas cifras se reportan con dos decimales, como exige el enunciado.

## Código que produce el reporte
El script carga los seis valores en un DataFrame de pandas, calcula media y
mediana con numpy (`np.mean`, `np.median`) y escribe `resultados.json` con
las cifras redondeadas a dos decimales (redondeo half-up vía `decimal`).
"""
Path("reporte.md").write_text(reporte, encoding="utf-8")

# --------------------------------------------------------------- resumen
print("Subtarea T1 completada.")
print(f"Datos (USD): {VENTAS_USD}")
print(f"n observaciones: {len(valores)}")
print(f"Media   : {media_2:.2f} USD (sin redondear: {media})")
print(f"Mediana : {mediana_2:.2f} USD (sin redondear: {mediana})")
print("Archivos escritos: resultados.json, reporte.md")
