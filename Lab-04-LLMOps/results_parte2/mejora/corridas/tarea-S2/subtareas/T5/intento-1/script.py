import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

TOL = 1e-12  # tolerancia numérica para comparar medias en punto flotante

RUTA_DATOS = Path("data/model_errors.csv")
RUTA_T4 = Path("entrada/T4/resultados.json")
RUTA_JSON = Path("resultados.json")
RUTA_CSV_DIFF = Path("output/store_differences.csv")
RUTA_PNG_DIFF = Path("output/store_differences.png")
RUTA_PNG_HIST = Path("t5_signflip_null_distribution.png")

# ----------------------------------------------------------------------
# 1) Vector de diferencias: difference = error_a - error_b (pareado por store_id)
# ----------------------------------------------------------------------
t4 = None
if RUTA_T4.exists():
    try:
        with open(RUTA_T4, "r", encoding="utf-8") as f:
            t4 = json.load(f)
    except Exception:
        t4 = None

df = None
if RUTA_DATOS.exists():
    df = pd.read_csv(RUTA_DATOS)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.sort_values("store_id").reset_index(drop=True)
    diffs = (df["error_a"].astype(float) - df["error_b"].astype(float)).to_numpy()
    store_ids = df["store_id"].astype(str).tolist()
    fuente = str(RUTA_DATOS)
else:
    # Respaldo: usar el vector de diferencias reportado por T4
    diffs = np.asarray(t4["vector_diferencias"], dtype=float)
    store_ids = [f"S{i:02d}" for i in range(1, len(diffs) + 1)]
    fuente = f"{RUTA_T4} (vector_diferencias)"

n = len(diffs)
observed_mean = float(np.mean(diffs))
observed_sum = float(np.sum(diffs))
abs_obs = abs(observed_mean)

# Verificación contra T4
media_t4 = float(t4["mean"]) if (t4 and "mean" in t4) else None
coincide_media_t4 = bool(media_t4 is not None and abs(media_t4 - observed_mean) <= 1e-9)
vector_t4 = t4.get("vector_diferencias") if t4 else None
coincide_vector_t4 = bool(
    vector_t4 is not None
    and np.allclose(np.asarray(vector_t4, dtype=float), diffs, atol=1e-9, rtol=0.0)
)

# ----------------------------------------------------------------------
# 2) Enumeración EXHAUSTIVA de las 2^16 = 65536 configuraciones de signos
#    bit j de idx = 0 -> signo +1 (original); bit j = 1 -> signo -1 (flip)
#    idx = 0 corresponde a la configuración observada (todos los signos +)
# ----------------------------------------------------------------------
total = 2 ** n
idx = np.arange(total, dtype=np.int64)
bits = (idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1
signs = (1.0 - 2.0 * bits).astype(np.float64)          # (65536, 16) con +1/-1
null_sums = signs @ diffs                              # suma de cada configuración
null_means = null_sums / n                             # media de cada configuración

# ----------------------------------------------------------------------
# 3) Conteo de medias nulas al menos tan extremas como la observada
# ----------------------------------------------------------------------
abs_null = np.abs(null_means)
extreme_mask = abs_null >= (abs_obs - TOL)
extreme_count = int(np.count_nonzero(extreme_mask))
p_value = extreme_count / total

strict_count = int(np.count_nonzero(abs_null > (abs_obs + TOL)))
equal_count = extreme_count - strict_count
upper_tail = int(np.count_nonzero(null_means >= (abs_obs - TOL)))
lower_tail = int(np.count_nonzero(null_means <= (-abs_obs + TOL)))

obs_config_ok = bool(abs(null_means[0] - observed_mean) <= TOL)
obs_in_extremes = bool(extreme_mask[0])

null_summary = {
    "min": float(null_means.min()),
    "max": float(null_means.max()),
    "media": float(null_means.mean()),
    "sd": float(null_means.std(ddof=1)),
    "cuantil_2_5": float(np.quantile(null_means, 0.025)),
    "cuantil_97_5": float(np.quantile(null_means, 0.975)),
    "max_abs": float(abs_null.max()),
}

# Prueba t de referencia (misma que T4, recalculada sobre los datos)
t_res = stats.ttest_1samp(diffs, 0.0)
t_stat = float(t_res.statistic)
t_p = float(t_res.pvalue)
gl = n - 1

# ----------------------------------------------------------------------
# 4) Archivos exigidos por el enunciado: output/store_differences.csv y .png
# ----------------------------------------------------------------------
RUTA_CSV_DIFF.parent.mkdir(parents=True, exist_ok=True)
RUTA_PNG_DIFF.parent.mkdir(parents=True, exist_ok=True)

if df is not None:
    out_df = pd.DataFrame({
        "store_id": df["store_id"].astype(str).values,
        "error_a": df["error_a"].astype(float).values,
        "error_b": df["error_b"].astype(float).values,
        "difference": diffs,
    })
else:
    out_df = pd.DataFrame({"store_id": store_ids, "difference": diffs})
out_df["favors"] = np.where(out_df["difference"] > 0, "B",
                            np.where(out_df["difference"] < 0, "A", "tie"))
out_df.to_csv(RUTA_CSV_DIFF, index=False)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#1f77b4" if d >= 0 else "#d62728" for d in out_df["difference"]]
ax.bar(out_df["store_id"], out_df["difference"], color=colors)
ax.axhline(0.0, color="black", lw=0.8)
ax.axhline(observed_mean, color="green", ls="--", lw=1.4,
           label=f"media observada = {observed_mean:.4f}")
ax.set_xlabel("Tienda")
ax.set_ylabel("Diferencia de error (error_a - error_b)")
ax.set_title("Diferencias de error por tienda (pareadas)")
ax.legend()
fig.tight_layout()
fig.savefig(RUTA_PNG_DIFF, dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5) Figura adicional: distribución exacta nula de las medias por cambios de signo
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(null_means, bins=80, color="#9ecae1", edgecolor="white")
ax.axvline(0.0, color="black", lw=0.8)
ax.axvline(observed_mean, color="green", ls="--", lw=1.6, label="media observada")
ax.axvline(-observed_mean, color="green", ls=":", lw=1.6, label="-media observada")
ax.set_xlabel("Media nula bajo cambios de signo")
ax.set_ylabel("Frecuencia (configuraciones)")
ax.set_title(f"Distribución exacta de medias nulas (2^{n} = {total} configuraciones)\n"
             f"extremas: {extreme_count}/{total}  ->  p = {p_value:.6f}")
ax.legend()
fig.tight_layout()
fig.savefig(RUTA_PNG_HIST, dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T5",
    "descripcion": ("Prueba exacta por cambios de signo: enumeración de las 2^16 = 65536 "
                    "configuraciones de signos del vector de diferencias, conteo de medias "
                    "nulas al menos tan extremas como la observada y p-value exacto"),
    "fuente_datos": fuente,
    "definicion_diferencia": "difference = error_a - error_b, emparejada por store_id",
    "metodo": ("enumeración exhaustiva y determinista de las 2^16 configuraciones de signos "
               "(sin muestreo ni aleatoriedad)"),
    "n": int(n),
    "media_observada": observed_mean,
    "suma_observada": observed_sum,
    "umbral_extremo_en_media": abs_obs,
    "umbral_extremo_en_suma": abs(observed_sum),
    "criterio_extremo": "|media_nula| >= |media_observada| (tolerancia numerica 1e-12)",
    "total_configuraciones": int(total),
    "conteo_extremas": extreme_count,
    "p_value": p_value,
    "p_value_como_fraccion": f"{extreme_count}/{total}",
    "p_value_formula": "p_value = conteo_extremas / total_configuraciones",
    "aclaracion_notacion": ("El enunciado pide enumerar las 2^16 = 65536 configuraciones de "
                            "signos; la notación '216' en fragmentos auxiliares corresponde a 2^16."),
    "desglose_conteo": {
        "estrictamente_mas_extremas": strict_count,
        "igualmente_extremas": equal_count,
        "nota": "la configuración observada (todos los signos +) está incluida en el conteo de extremas",
    },
    "colas": {
        "media_nula_mayor_igual_que_observada": upper_tail,
        "media_nula_menor_igual_que_menos_observada": lower_tail,
        "colas_suman_conteo_extremas": bool(upper_tail + lower_tail == extreme_count),
    },
    "distribucion_nula_medias": null_summary,
    "verificacion": {
        "configuracion_todo_positivo_iguala_media_observada": obs_config_ok,
        "configuracion_observada_cuenta_como_extrema": obs_in_extremes,
        "media_T4": media_t4,
        "coincide_con_media_T4": coincide_media_t4,
        "coincide_con_vector_T4": coincide_vector_t4,
    },
    "comparacion_con_prueba_t": {
        "estadistico_t": t_stat,
        "grados_libertad": int(gl),
        "p_value_t_bilateral": t_p,
        "p_value_cambios_de_signo": p_value,
        "diferencia_abs": abs(t_p - p_value),
        "nota": ("Ambas referencias nulas son bilaterales sobre la media de las diferencias y "
                 "producen valores numéricos cercanos, pero no son intercambiables: la prueba t "
                 "asume que las diferencias son una muestra aleatoria de una población "
                 "aproximadamente normal y usa la distribución t de Student del estadístico "
                 "estandarizado; la prueba exacta por cambios de signo solo asume que, bajo H0, "
                 "cada diferencia es simétrica alrededor de 0 (signos intercambiables), "
                 "condiciona en las magnitudes observadas |d_i| y usa la distribución exacta de "
                 "las 2^16 medias, sin suponer normalidad."),
    },
    "interpretacion_p_value": ("El p-value es la probabilidad, condicional a la hipótesis nula de "
                               "efecto nulo (cada diferencia simétrica alrededor de 0, signos "
                               "igualmente probables), de observar una media de las diferencias al "
                               "menos tan extrema en magnitud como la observada; se obtiene "
                               "contando configuraciones de signos, no ajustando parámetros."),
    "archivos_generados": [str(RUTA_CSV_DIFF), str(RUTA_PNG_DIFF), str(RUTA_PNG_HIST)],
}

with open(RUTA_JSON, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7) Resumen
# ----------------------------------------------------------------------
print("T5 - Prueba exacta por cambios de signo")
print(f"  n diferencias            : {n}")
print(f"  media observada          : {observed_mean:.6f}")
print(f"  total de configuraciones : {total}")
print(f"  configuraciones extremas : {extreme_count}")
print(f"  p-value exacto           : {p_value:.6f}  ({extreme_count}/{total})")
print(f"  comparación              : p_t(bilateral) = {t_p:.6f} vs p_signflip = {p_value:.6f}")
print(f"  coincide con media de T4 : {coincide_media_t4}")
print(f"  archivos generados       : {RUTA_JSON}, {RUTA_CSV_DIFF}, {RUTA_PNG_DIFF}, {RUTA_PNG_HIST}")
