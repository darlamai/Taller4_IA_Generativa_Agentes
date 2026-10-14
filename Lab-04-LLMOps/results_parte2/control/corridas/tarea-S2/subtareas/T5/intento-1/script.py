# -*- coding: utf-8 -*-
"""
T5 — Referencia nula exacta por cambios de signo (enumeración completa 2^16).

Enumera las 2^16 = 65 536 configuraciones de signos del vector de diferencias
pareadas (difference = error_a − error_b), calcula la media nula de cada
configuración, cuenta cuántas tienen magnitud al menos tan extrema como la
media observada y reporta: total de configuraciones, conteo de extremas y
p-value exacto = conteo_extremas / 65 536.

Salidas:
  - resultados.json                  (contrato de la subtarea)
  - output/store_differences.csv     (tabla pareada con diferencias)
  - output/store_differences.png     (figura: diferencias + referencia nula)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) Carga y preparación del vector de diferencias
# ----------------------------------------------------------------------------
ruta_csv = Path("data/model_errors.csv")
df = pd.read_csv(ruta_csv)

columnas_esperadas = ["store_id", "error_a", "error_b"]
faltantes = [c for c in columnas_esperadas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas en {ruta_csv}: {faltantes}")
if df[columnas_esperadas].isna().any().any():
    raise ValueError("Hay valores nulos en las columnas requeridas")

diff = (df["error_a"].astype(float) - df["error_b"].astype(float)).to_numpy()
n = int(diff.size)
if n != 16:
    raise ValueError(f"Se esperaban 16 diferencias pareadas, hay {n}")

# Verificación opcional (no fatal) contra la subtarea previa T2
coincide_t2 = None
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    try:
        t2 = json.loads(ruta_t2.read_text(encoding="utf-8"))
        filas = t2.get("tabla_pareada", [])
        if len(filas) == n:
            d_t2 = np.array([float(f["difference"]) for f in filas])
            ids_t2 = [str(f["store_id"]) for f in filas]
            coincide_t2 = bool(
                ids_t2 == [str(s) for s in df["store_id"]]
                and np.allclose(d_t2, diff, atol=1e-9, rtol=0.0)
            )
    except Exception:
        coincide_t2 = None

# ----------------------------------------------------------------------------
# 2) Enumeración completa de las 2^16 configuraciones de signos
#    Comparación EXACTA en enteros (diferencias escaladas a milésimas),
#    pues las diferencias son múltiplos exactos de 0.1.
# ----------------------------------------------------------------------------
ESCALA = 1000
total_cfg = 2 ** n
indices = np.arange(total_cfg, dtype=np.int64)
bits = (indices[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1
signos = (2 * bits - 1).astype(np.int64)            # (65536, 16) con ±1

d_esc = np.rint(diff * ESCALA).astype(np.int64)     # diferencias exactas escaladas
if not np.allclose(d_esc / ESCALA, diff, atol=1e-9, rtol=0.0):
    raise ValueError("Las diferencias no son representables en la escala exacta elegida")

sumas_nulas = signos @ d_esc                        # suma firmada exacta por configuración
suma_obs = int(d_esc.sum())                         # suma observada (6700 milésimas = 6.7)
media_obs = suma_obs / (ESCALA * n)                 # 0.41875

extremas_mask = np.abs(sumas_nulas) >= abs(suma_obs)
conteo_extremas = int(extremas_mask.sum())
p_exacto = conteo_extremas / total_cfg

conteo_pos = int((sumas_nulas >= abs(suma_obs)).sum())
conteo_neg = int((sumas_nulas <= -abs(suma_obs)).sum())
conteo_igual = int((np.abs(sumas_nulas) == abs(suma_obs)).sum())

medias_nulas = sumas_nulas / (ESCALA * n)
media_nula_min = float(medias_nulas.min())
media_nula_max = float(medias_nulas.max())

# Comprobaciones internas de la enumeración
assert total_cfg == 65536
assert int(signos[total_cfg - 1].sum()) == n            # configuración todos +1 presente
assert int(sumas_nulas[total_cfg - 1]) == suma_obs      # reproduce el estadístico observado
assert conteo_pos == conteo_neg                          # simetría exacta de la referencia
assert conteo_extremas == conteo_pos + conteo_neg
assert conteo_extremas >= 2                              # ± configuración observada contadas

# ----------------------------------------------------------------------------
# 3) Verificación independiente A: enumeración en float64 con tolerancia
# ----------------------------------------------------------------------------
medias_nulas_float = (signos * diff[None, :]).mean(axis=1)
conteo_float = int(
    (np.abs(medias_nulas_float) >= abs(float(diff.mean())) - 1e-9).sum()
)
assert conteo_float == conteo_extremas

# ----------------------------------------------------------------------------
# 4) Verificación independiente B: programación dinámica exacta de sumas con signo
# ----------------------------------------------------------------------------
offset = int(np.abs(d_esc).sum())
tam = 2 * offset + 1
dp = np.zeros(tam, dtype=np.int64)
dp[offset] = 1
for a in d_esc:
    u = int(abs(a))
    ndp = np.zeros(tam, dtype=np.int64)
    ndp[u:] += dp[: tam - u]        # ndp[i] += dp[i - u]
    ndp[: tam - u] += dp[u:]        # ndp[i] += dp[i + u]
    dp = ndp
assert int(dp.sum()) == total_cfg
idx = np.arange(tam)
conteo_dp = int(dp[np.abs(idx - offset) >= abs(suma_obs)].sum())
assert conteo_dp == conteo_extremas

# ----------------------------------------------------------------------------
# 5) Referencia comparativa: prueba t de una muestra (bilateral)
# ----------------------------------------------------------------------------
t_res = stats.ttest_1samp(diff, 0.0)
t_stat = float(t_res.statistic)
p_t = float(t_res.pvalue)

# ----------------------------------------------------------------------------
# 6) Archivos exigidos: output/store_differences.csv y output/store_differences.png
# ----------------------------------------------------------------------------
Path("output").mkdir(parents=True, exist_ok=True)

tabla = df[columnas_esperadas].copy()
tabla["difference"] = diff
tabla["favors"] = np.where(tabla["difference"] > 0, "B",
                    np.where(tabla["difference"] < 0, "A", "tie"))
tabla.to_csv("output/store_differences.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
colores = np.where(diff > 0, "#2166ac", "#b2182b")
axes[0].bar(range(n), diff, color=colores)
axes[0].axhline(0.0, color="black", lw=0.8)
axes[0].axhline(media_obs, color="green", ls="--", lw=1.2,
                label=f"media observada = {media_obs:.5f}")
axes[0].set_xticks(range(n))
axes[0].set_xticklabels([str(s) for s in df["store_id"]], rotation=90)
axes[0].set_ylabel("difference = error_a − error_b")
axes[0].set_title("Diferencias pareadas por tienda")
axes[0].legend(loc="lower right", fontsize=8)

bins = np.linspace(-1.25, 1.25, 101)
axes[1].hist(medias_nulas[~extremas_mask], bins=bins, color="#9ecae1",
             label="medias nulas no extremas")
axes[1].hist(medias_nulas[extremas_mask], bins=bins, color="#fb6a4a",
             label=f"medias nulas extremas ({conteo_extremas})")
axes[1].axvline(media_obs, color="green", ls="--", lw=1.4, label="|media observada|")
axes[1].axvline(-media_obs, color="green", ls=":", lw=1.4)
axes[1].set_xlabel("media nula (cambios de signo)")
axes[1].set_ylabel("frecuencia (configuraciones)")
axes[1].set_title(f"Referencia nula exacta: 2^{n} = {total_cfg:,} configuraciones\n"
                  f"p exacto = {conteo_extremas}/{total_cfg} = {p_exacto:.6f}")
axes[1].legend(loc="upper left", fontsize=8)
fig.suptitle("T5 · Comparación de referencias nulas: prueba exacta por cambios de signo",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("output/store_differences.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T5_referencia_nula_exacta_por_cambios_de_signo",
    "fuente_datos": "data/model_errors.csv",
    "n_tiendas": n,
    "vector_diferencias": [float(x) for x in diff],
    "media_diferencia_observada": media_obs,
    "suma_diferencias_observada": suma_obs / ESCALA,
    "total_configuraciones": total_cfg,
    "conteo_configuraciones_extremas": conteo_extremas,
    "p_value_exacto": p_exacto,
    "prueba_exacta_cambios_de_signo": {
        "metodo": (
            "enumeración completa de las 2^16 configuraciones de signos del vector de "
            "diferencias; la media nula de cada configuración se compara en aritmética "
            "exacta (enteros escalados) con el criterio |media nula| >= |media observada|"
        ),
        "total_configuraciones": total_cfg,
        "total_configuraciones_es_2_elevado_16": total_cfg == 65536,
        "conteo_configuraciones_extremas": conteo_extremas,
        "p_value_exacto": p_exacto,
        "p_value_fraccion": [conteo_extremas, total_cfg],
        "conteo_extremas_lado_positivo": conteo_pos,
        "conteo_extremas_lado_negativo": conteo_neg,
        "conteo_medias_nulas_con_magnitud_igual_a_la_observada": conteo_igual,
        "media_nula_minima": media_nula_min,
        "media_nula_maxima": media_nula_max,
        "percentil_2_5_medias_nulas": float(np.percentile(medias_nulas, 2.5)),
        "percentil_97_5_medias_nulas": float(np.percentile(medias_nulas, 97.5)),
    },
    "verificaciones": {
        "enumeracion_vs_dinamica_exacta_coinciden": bool(conteo_dp == conteo_extremas),
        "enumeracion_vs_float_con_tolerancia_coinciden": bool(conteo_float == conteo_extremas),
        "simetria_colas_positiva_negativa": bool(conteo_pos == conteo_neg),
        "configuracion_observada_incluida_en_extremas": True,
        "diferencias_coinciden_con_T2": coincide_t2,
    },
    "comparacion_con_prueba_t": {
        "t_statistic": t_stat,
        "p_value_t_bilateral": p_t,
        "diferencia_absoluta_p_values": abs(p_t - p_exacto),
        "nota": (
            "Ambos procedimientos evalúan la misma hipótesis nula (media de las "
            "diferencias = 0) pero usan referencias nulas distintas: la prueba t usa la "
            "distribución t de Student de la media estandarizada (supone diferencias iid "
            "aproximadamente normales), mientras que la prueba exacta por cambios de signo "
            "usa la distribución de la media bajo todas las asignaciones de signos "
            "equiprobables (solo requiere que, bajo H0, cada diferencia sea simétrica "
            "respecto de 0). Por eso no son intercambiables, aunque aquí produzcan "
            "valores numéricos cercanos."
        ),
    },
    "salidas": {
        "csv_tabla_diferencias": "output/store_differences.csv",
        "figura": "output/store_differences.png",
        "resultados_json": "resultados.json",
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 8) Resumen
# ----------------------------------------------------------------------------
print("=" * 66)
print("T5 · Referencia nula exacta por cambios de signo")
print("=" * 66)
print(f"Tiendas pareadas (n)                       : {n}")
print(f"Media observada de las diferencias         : {media_obs:.6f}")
print(f"Total de configuraciones enumeradas        : {total_cfg} (= 2^16)")
print(f"Configuraciones extremas (|media| >= |obs|): {conteo_extremas}")
print(f"  - lado positivo / lado negativo          : {conteo_pos} / {conteo_neg}")
print(f"  - con magnitud exactamente igual a |obs| : {conteo_igual}")
print(f"p-value exacto por cambios de signo        : {p_exacto:.10f}")
print(f"Referencia comparativa (prueba t bilateral): t = {t_stat:.4f}, p = {p_t:.6f}")
print("Verificaciones (DP exacta, float+tolerancia, simetría): OK")
print("Salidas: resultados.json, output/store_differences.csv, output/store_differences.png")
