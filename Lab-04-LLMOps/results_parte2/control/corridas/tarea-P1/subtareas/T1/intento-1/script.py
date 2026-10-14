# T1 — Ventas mensuales enero-junio: media y mediana con dos decimales
from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP

# Datos del enunciado (Parte 1 — Resumen)
meses = ["ene", "feb", "mar", "abr", "may", "jun"]
ventas = [1250.50, 980.25, 1430.75, 1120.00, 1675.50, 1305.25]

# Cálculo exacto con Decimal (evita artefactos de punto flotante) y redondeo half-up
ventas_dec = [Decimal(str(v)) for v in ventas]
n = len(ventas_dec)

media = sum(ventas_dec) / Decimal(n)

ventas_ord = sorted(ventas_dec)
if n % 2 == 0:
    mediana = (ventas_ord[n // 2 - 1] + ventas_ord[n // 2]) / Decimal(2)
else:
    mediana = ventas_ord[n // 2]

dos_dec = Decimal("0.01")
media_2d = media.quantize(dos_dec, rounding=ROUND_HALF_UP)
mediana_2d = mediana.quantize(dos_dec, rounding=ROUND_HALF_UP)

# Contrato de salida: resultados.json con las dos cifras pedidas
resultados = {
    "media_ventas_usd": float(media_2d),
    "mediana_ventas_usd": float(mediana_2d),
    "media_ventas_usd_2dec": f"{media_2d:.2f}",
    "mediana_ventas_usd_2dec": f"{mediana_2d:.2f}",
    "ventas_usd": ventas,
    "meses": meses,
}

out_path = Path("resultados.json")
out_path.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

# Resumen breve
print("=== T1: Resumen de ventas mensuales (USD) ===")
for m, v in zip(meses, ventas):
    print(f"  {m}: {v:.2f}")
print(f"Media   : {media_2d:.2f} USD")
print(f"Mediana : {mediana_2d:.2f} USD")
print(f"Resultados escritos en: {out_path.resolve()}")
