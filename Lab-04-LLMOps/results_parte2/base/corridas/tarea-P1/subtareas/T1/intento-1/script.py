import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

# ---------------------------------------------------------------
# SUBTAREA T1: media y mediana de las ventas mensuales (ene-jun)
# Datos tomados textualmente del enunciado (Tarea P1, Parte 1)
# ---------------------------------------------------------------

meses = ["ene", "feb", "mar", "abr", "may", "jun"]
ventas_usd = [1250.50, 980.25, 1430.75, 1120.00, 1675.50, 1305.25]

# --- Cálculo de las dos cifras pedidas ---
media = float(np.mean(ventas_usd))
mediana = float(np.median(ventas_usd))


def redondear_2_dec(x: float) -> float:
    """Redondeo estándar (half-up) a exactamente dos decimales."""
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


media_2 = redondear_2_dec(media)
mediana_2 = redondear_2_dec(mediana)

# --- Contrato de salida: resultados.json ---
resultados = {
    "subtarea": "T1_resumen_ventas_mensuales",
    "meses": meses,
    "ventas_usd": ventas_usd,
    "n_observaciones": len(ventas_usd),
    "media_ventas_usd": media_2,
    "mediana_ventas_usd": mediana_2,
    "media_ventas_usd_2dec": f"{media_2:.2f}",
    "mediana_ventas_usd_2dec": f"{mediana_2:.2f}",
}

ruta_salida = Path("resultados.json")
ruta_salida.write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# --- Resumen breve en consola ---
print("Ventas mensuales (USD):", ventas_usd)
print(f"Suma total: {sum(ventas_usd):.2f} USD en {len(ventas_usd)} meses")
print(f"Media de las ventas:   {media_2:.2f} USD")
print(f"Mediana de las ventas: {mediana_2:.2f} USD")
print(f"Resultados escritos en: {ruta_salida.resolve()}")
