# T1: Cargar ventas mensuales enero-junio y calcular media y mediana (2 decimales)
from pathlib import Path
import json
import numpy as np

# --- Datos del enunciado (Parte 1, tabla página 1) ---
meses = ["ene", "feb", "mar", "abr", "may", "jun"]
ventas = [1250.50, 980.25, 1430.75, 1120.00, 1675.50, 1305.25]  # USD

ventas_arr = np.array(ventas, dtype=float)

# --- Cálculo de las dos cifras pedidas ---
media = float(np.mean(ventas_arr))
mediana = float(np.median(ventas_arr))

# Redondeo a dos decimales (formato exigido por el enunciado)
media_2d = round(media, 2)
mediana_2d = round(mediana, 2)

# --- Contrato de salida: resultados.json ---
resultados = {
    "media_usd": media_2d,
    "mediana_usd": mediana_2d,
    "media_usd_sin_redondear": media,
    "mediana_usd_sin_redondear": mediana,
    "ventas_usd": ventas,
    "meses": meses,
    "n_meses": len(ventas),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --- Resumen breve en consola ---
print("T1 - Resumen de ventas mensuales (enero a junio, USD)")
print(f"Ventas registradas: {dict(zip(meses, ventas))}")
print(f"Media:   {media_2d:.2f} USD")
print(f"Mediana: {mediana_2d:.2f} USD")
print("Resultados escritos en resultados.json")
