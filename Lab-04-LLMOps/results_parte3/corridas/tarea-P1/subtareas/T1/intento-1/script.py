"""
Subtarea T1 — Ventas mensuales del semestre (ene-jun)
Calcula la media y la mediana de las ventas mensuales (USD),
redondeadas a dos decimales, y escribe resultados.json.
"""

import json
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Carga de los datos del enunciado (Parte 1, tabla de ventas)
# ------------------------------------------------------------------
meses = ["ene", "feb", "mar", "abr", "may", "jun"]
ventas = [1250.50, 980.25, 1430.75, 1120.00, 1675.50, 1305.25]

df = pd.DataFrame({"mes": meses, "ventas_usd": ventas})
valores = df["ventas_usd"].to_numpy(dtype=float)

# ------------------------------------------------------------------
# 2. Cálculo de media y mediana (sin redondear)
# ------------------------------------------------------------------
media_raw = float(np.mean(valores))
mediana_raw = float(np.median(valores))

# ------------------------------------------------------------------
# 3. Redondeo a dos decimales (half-up, criterio financiero)
# ------------------------------------------------------------------
def redondear_2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

media_2 = redondear_2(media_raw)
mediana_2 = redondear_2(mediana_raw)

# ------------------------------------------------------------------
# 4. Contrato de salida: resultados.json
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1_resumen_ventas_semestre",
    "unidad": "USD",
    "meses": meses,
    "ventas_usd": ventas,
    "n_observaciones": len(ventas),
    "media_usd": media_2,
    "mediana_usd": mediana_2,
    "media_usd_2dec": f"{media_2:.2f}",
    "mediana_usd_2dec": f"{mediana_2:.2f}",
    "media_usd_sin_redondear": media_raw,
    "mediana_usd_sin_redondear": mediana_raw,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 5. Resumen por consola (criterio de éxito: dos decimales exactos)
# ------------------------------------------------------------------
print("Ventas mensuales (USD):")
for m, v in zip(meses, ventas):
    print(f"  {m}: {v:.2f}")
print("-" * 40)
print(f"Media  de las ventas: {media_2:.2f} USD")
print(f"Mediana de las ventas: {mediana_2:.2f} USD")
print("Resultados escritos en resultados.json")
