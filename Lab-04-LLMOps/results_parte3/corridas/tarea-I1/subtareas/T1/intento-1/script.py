# T1 — Resumen estadístico de latencias de un servicio (MMIA 6013, Taller 04, Parte 2.a)
# Calcula media, desviación estándar muestral (n-1) y error estándar de la media
# para las ocho latencias registradas, redondeadas a dos decimales.

import json
import numpy as np

# Datos del enunciado (Parte 1): ocho llamadas consecutivas
latencias = np.array([120, 135, 128, 150, 142, 138, 160, 127], dtype=float)

n = latencias.size

# Media
media = float(np.mean(latencias))

# Desviación estándar muestral (denominador n - 1 => ddof=1)
sd_muestral = float(np.std(latencias, ddof=1))

# Error estándar de la media: s / sqrt(n)
sem = sd_muestral / np.sqrt(n)

# Cifras redondeadas a dos decimales (formato exigido por el enunciado)
media_2d = round(media, 2)
sd_2d = round(sd_muestral, 2)
sem_2d = round(float(sem), 2)

resultados = {
    "subtarea": "T1_resumen_estadistico_latencias",
    "n": int(n),
    "latencias_ms": latencias.tolist(),
    "media_ms": media_2d,
    "desviacion_estandar_muestral_ms": sd_2d,
    "error_estandar_media_ms": sem_2d,
    "valores_sin_redondear": {
        "media_ms": media,
        "desviacion_estandar_muestral_ms": sd_muestral,
        "error_estandar_media_ms": float(sem),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("=== T1: Resumen estadístico de latencias (ms) ===")
print(f"Latencias: {latencias.tolist()}")
print(f"n = {n}")
print(f"Media: {media_2d:.2f} ms")
print(f"Desviación estándar muestral (n-1): {sd_2d:.2f} ms")
print(f"Error estándar de la media: {sem_2d:.2f} ms")
print("Resultados escritos en resultados.json")
