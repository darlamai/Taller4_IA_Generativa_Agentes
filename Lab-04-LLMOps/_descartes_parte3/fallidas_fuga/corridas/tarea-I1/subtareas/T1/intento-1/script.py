# T1 — Resumen estadístico de latencias de un servicio (MMIA 6013 · Taller 04, Parte 2.a)
# Calcula media, desviación estándar muestral (n-1) y error estándar de la media (s/√n)
# sobre las ocho latencias registradas, y escribe los resultados en resultados.json.

import json
import numpy as np

# --- Datos del enunciado (ocho llamadas consecutivas, latencia en ms) ---
latencias = np.array([120, 135, 128, 150, 142, 138, 160, 127], dtype=float)

n = latencias.size

# --- Cálculos ---
media = float(np.mean(latencias))                      # media aritmética
s_muestral = float(np.std(latencias, ddof=1))          # desviación estándar muestral (n-1)
sem = s_muestral / float(np.sqrt(n))                   # error estándar de la media: s/√n

# --- Cifras reportadas con dos decimales (contrato de la subtarea) ---
media_2dec = round(media, 2)
s_2dec = round(s_muestral, 2)
sem_2dec = round(sem, 2)

resultados = {
    "subtarea": "T1_resumen_estadistico_latencias",
    "n": int(n),
    "datos_latencia_ms": [float(x) for x in latencias],
    "media_ms": media_2dec,
    "desviacion_estandar_muestral_ms": s_2dec,
    "error_estandar_media_ms": sem_2dec,
    # Representación textual con exactamente dos decimales (p. ej. "137.50")
    "media_ms_2dec": f"{media:.2f}",
    "desviacion_estandar_muestral_ms_2dec": f"{s_muestral:.2f}",
    "error_estandar_media_ms_2dec": f"{sem:.2f}",
    # Valores sin redondear, para trazabilidad
    "media_ms_sin_redondear": media,
    "desviacion_estandar_muestral_ms_sin_redondear": s_muestral,
    "error_estandar_media_ms_sin_redondear": sem,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --- Resumen en consola ---
print("T1 — Resumen estadístico de latencias (ms)")
print(f"Datos (n={n}): {[int(x) for x in latencias]}")
print(f"Media: {media:.2f} ms")
print(f"Desviación estándar muestral (n-1): {s_muestral:.2f} ms")
print(f"Error estándar de la media (s/√n): {sem:.2f} ms")
print("Resultados escritos en resultados.json")
