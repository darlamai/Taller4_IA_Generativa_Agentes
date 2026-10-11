# -*- coding: utf-8 -*-
"""
T5 - Referencia nula exacta por cambios de signo.

Subtarea: enumerar las 2^16 = 65536 configuraciones de signos del vector de
diferencias pareadas (error_a - error_b, 16 tiendas), calcular la media nula de
cada configuración, contar cuántas medias nulas tienen magnitud al menos tan
extrema como la observada, y reportar:
  - total de configuraciones (65536),
  - conteo de configuraciones extremas,
  - p-value exacto por cambios de signo,
  - el p-value bilateral de la prueba t de una muestra (T4) para comparación.

Salidas:
  - resultados.json                     (contrato de la subtarea)
  - sign_flip_null_distribution.png     (histograma de la referencia nula)
  - output/store_differences.csv        (entregable exigido por el enunciado)
  - output/store_differences.png        (entregable exigido por el enunciado)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ----------------------------------------------------------------------------
# 0) Utilidades
# ----------------------------------------------------------------------------

def cargar_json_previo(ruta):
    p = Path(ruta)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 1) Datos: data/model_errors.csv -> vector de diferencias pareadas
# ----------------------------------------------------------------------------
csv_path = Path("data/model_errors.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo de datos: {csv_path}")

df = pd.read_csv(csv_path)
req = ["store_id", "error_a", "error_b"]
faltan = [c for c in req if c not in df.columns]
if faltan:
    raise ValueError(f"Faltan las columnas exigidas {faltan} en {csv_path}")
if df[req].isna().any().any():
    raise ValueError("Hay valores nulos en las columnas exigidas (store_id, error_a, error_b)")
if df["store_id"].duplicated().any():
    raise ValueError("Hay identificadores de tienda (store_id) duplicados")
for c in ["error_a", "error_b"]:
    v = pd.to_numeric(df[c], errors="coerce")
    if v.isna().any():
        raise ValueError(f"La columna {c} contiene valores no numéricos")
    vv = v.to_numpy(dtype=float)
    if not np.isfinite(vv).all():
        raise ValueError(f"La columna {c} contiene valores no finitos")
    if (vv < 0).any():
        raise ValueError(f"La columna {c} contiene valores negativos")
    df[c] = vv

d = (df["error_a"] - df["error_b"]).to_numpy(dtype=float)
n = int(d.size)
if n != 16:
    raise ValueError(f"Se esperaban 16 pares de tiendas; se encontraron {n}")

obs_mean = float(d.mean())
obs_abs = float(abs(obs_mean))
obs_sum = float(d.sum())

# ----------------------------------------------------------------------------
# 2) Enumeración exhaustiva de las 2^16 configuraciones de signos
# ----------------------------------------------------------------------------
n_conf = 2 ** n
if n_conf != 65536:
    raise ValueError(f"2^{n} = {n_conf} != 65536")

# Matriz de signos en {-1,+1}: el bit i de cada índice decide el signo de d_i.
idx = np.arange(n_conf, dtype=np.int64)
bits = np.arange(n, dtype=np.int64)
signs = (((idx[:, None] >> bits[None, :]) & 1).astype(np.int8) * 2) - 1  # (65536, 16)

# Aritmética entera EXACTA: las diferencias son múltiplos de 0.1.
escala = 10
d_int = np.round(d * escala).astype(np.int64)
if not np.allclose(d_int / float(escala), d, rtol=0.0, atol=1e-9):
    raise ValueError("Las diferencias no son múltiplos de 0.1; no se puede usar aritmética entera exacta")

null_sums_int = signs.astype(np.int64) @ d_int        # sumas con signo, exactas (enteros)
null_means = null_sums_int / float(escala * n)        # medias nulas en unidades originales

n_unicas = int(np.unique(signs, axis=0).shape[0])     # verificación: 65536 configuraciones únicas

# ----------------------------------------------------------------------------
# 3) Conteo de medias nulas al menos tan extremas como la observada
# ----------------------------------------------------------------------------
obs_sum_int = int(np.round(obs_sum * escala))
if obs_sum_int != int(d_int.sum()):
    raise ValueError("Inconsistencia entre la suma observada y la suma entera de diferencias")
abs_obs_int = abs(obs_sum_int)

extreme_mask = np.abs(null_sums_int) >= abs_obs_int   # criterio bilateral: |media nula| >= |media obs|
count_extreme = int(extreme_mask.sum())
count_strict = int((np.abs(null_sums_int) > abs_obs_int).sum())
count_equal = count_extreme - count_strict
p_exact = count_extreme / n_conf                      # denominador 2^16 => valor diádico exacto

# Verificación independiente en coma flotante con tolerancia numérica
count_extreme_float = int((np.abs(null_means) >= obs_abs - 1e-9).sum())
verif_float_ok = bool(count_extreme_float == count_extreme)

# La configuración observada (todos +) y su opuesta (todos -) deben estar contadas
config_observada_incluida = bool(
    np.any(null_sums_int == int(d_int.sum())) and np.any(null_sums_int == -int(d_int.sum()))
)

# Tabla exacta de la referencia nula (valores únicos y sus conteos)
valores_unicos, conteos_unicos = np.unique(null_means, return_counts=True)
if int(conteos_unicos.sum()) != n_conf:
    raise ValueError("La tabla de valores únicos no suma 65536 configuraciones")
referencia_unicos = [{"media": float(m), "conteo": int(c)}
                     for m, c in zip(valores_unicos, conteos_unicos)]

# ----------------------------------------------------------------------------
# 4) Prueba t de una muestra sobre las diferencias (bilateral) — comparable
# ----------------------------------------------------------------------------
t_res = stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
t_stat = float(t_res.statistic)
p_t = float(t_res.pvalue)
gl = n - 1
p_t_sf = float(2.0 * stats.t.sf(abs(t_stat), gl))     # verificación con la distribución t

# ----------------------------------------------------------------------------
# 5) Resumen de la distribución nula exacta
# ----------------------------------------------------------------------------
dist_summary = {
    "min": float(null_means.min()),
    "max": float(null_means.max()),
    "media": float(null_means.mean()),
    "sd": float(null_means.std(ddof=1)),
    "sd_teorica_bajo_signos": float(np.sqrt(np.sum(d ** 2)) / n),
    "simetrica_en_cero": bool(np.isclose(null_means.max(), -null_means.min(), rtol=0, atol=1e-12)),
    "percentiles": {
        "2.5": float(np.percentile(null_means, 2.5)),
        "50": float(np.percentile(null_means, 50)),
        "97.5": float(np.percentile(null_means, 97.5)),
    },
}

# ----------------------------------------------------------------------------
# 6) Verificación cruzada con subtareas previas (T2 y T4), si están disponibles
# ----------------------------------------------------------------------------
t2 = cargar_json_previo("entrada/T2/resultados.json")
verif_t2 = {"disponible": t2 is not None}
if t2 is not None and isinstance(t2.get("tabla_pareada"), list) and t2["tabla_pareada"]:
    diffs_t2 = np.array([r["difference"] for r in t2["tabla_pareada"]], dtype=float)
    verif_t2["n_T2"] = int(diffs_t2.size)
    verif_t2["n_coincide"] = bool(diffs_t2.size == n)
    verif_t2["diferencias_coinciden"] = bool(
        diffs_t2.size == n and np.allclose(np.sort(diffs_t2), np.sort(d), rtol=0, atol=1e-9)
    )

t4 = cargar_json_previo("entrada/T4/resultados.json")
verif_t4 = {"disponible": t4 is not None}
if t4 is not None:
    p4 = t4.get("p_value_two_sided")
    m4 = t4.get("mean")
    verif_t4["p_value_t_T4"] = p4
    verif_t4["p_value_coincide"] = bool(p4 is not None and np.isclose(p4, p_t, rtol=0, atol=1e-12))
    verif_t4["media_T4"] = m4
    verif_t4["media_coincide"] = bool(m4 is not None and np.isclose(m4, obs_mean, rtol=0, atol=1e-12))

# ----------------------------------------------------------------------------
# 7) Figura: histograma de la referencia nula exacta por cambios de signo
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.5))
counts_hist, edges, patches = ax.hist(null_means, bins=121, color="#9ecae1",
                                      edgecolor="white", linewidth=0.3)
ancho_bin = edges[1] - edges[0]
centros = edges[:-1] + ancho_bin / 2.0
for patch, centro in zip(patches, centros):
    if abs(centro) >= obs_abs:
        patch.set_facecolor("#de2d26")
        patch.set_alpha(0.95)
ax.axvline(obs_mean, color="black", lw=1.6, ls="--",
           label=f"media observada = {obs_mean:.6f}")
ax.axvline(-obs_mean, color="black", lw=1.6, ls=":")
ax.set_title(
    "Referencia nula exacta por cambios de signo (2^16 = 65536 medias)\n"
    f"extremas: {count_extreme} de 65536  ->  p exacto = {count_extreme}/65536 = {p_exact:.6f}"
    f"   |   p t bilateral = {p_t:.6f}"
)
ax.set_xlabel("media nula de las diferencias (error_a - error_b)")
ax.set_ylabel("número de configuraciones de signos")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("sign_flip_null_distribution.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8) Entregables exigidos por el enunciado: output/store_differences.csv y .png
# ----------------------------------------------------------------------------
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

tabla = df[req].copy()
tabla["difference"] = tabla["error_a"] - tabla["error_b"]
tabla["favors"] = np.where(tabla["difference"] > 0, "B",
                    np.where(tabla["difference"] < 0, "A", "empate"))
tabla.to_csv(out_dir / "store_differences.csv", index=False)

fig2, ax2 = plt.subplots(figsize=(9.5, 5.0))
colores = np.where(tabla["difference"].to_numpy() >= 0, "#31a354", "#e6550d")
ax2.bar(tabla["store_id"], tabla["difference"], color=colores, edgecolor="black", linewidth=0.4)
ax2.axhline(0.0, color="black", lw=0.8)
ax2.axhline(obs_mean, color="black", ls="--", lw=1.2, label=f"media = {obs_mean:.6f}")
ax2.set_title("Diferencias pareadas por tienda (error_a - error_b; verde favorece a B, naranja a A)")
ax2.set_xlabel("tienda")
ax2.set_ylabel("diferencia de error")
ax2.legend(loc="upper right")
fig2.tight_layout()
fig2.savefig(out_dir / "store_differences.png", dpi=120)
plt.close(fig2)

# ----------------------------------------------------------------------------
# 9) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T5: referencia nula exacta por cambios de signo (enumeración de 2^16 configuraciones)",
    "fuente_datos": "data/model_errors.csv",
    "definicion_diferencia": "difference = error_a - error_b (positiva favorece a B)",
    "n": n,
    "differences": d.tolist(),
    "observed_sum": obs_sum,
    "observed_mean": obs_mean,
    "observed_abs_mean": obs_abs,
    "metodo": (
        "enumeración exhaustiva y determinista de las 2^16 configuraciones de signos del vector de "
        "diferencias; para cada configuración se calcula la media nula y se cuenta cuántas tienen "
        "magnitud al menos tan extrema como la media observada"
    ),
    "criterio_extremidad": "|media_nula| >= |media_observada| (bilateral, 'al menos tan extrema')",
    "aritmetica": (
        "exacta en enteros: las diferencias son múltiplos de 0.1, se trabaja con d*10 y sumas con "
        "signo enteras (media nula = suma_entera/160); verificación independiente en coma flotante "
        "con tolerancia 1e-9"
    ),
    "escala_entera": float(escala * n),
    "suma_observada_en_enteros": obs_sum_int,
    "semilla": "no aplica: enumeración exhaustiva determinista, sin aleatoriedad",
    "total_configurations": int(n_conf),
    "total_configuraciones": int(n_conf),
    "total_configuraciones_esperado": 65536,
    "total_coincide_con_esperado": bool(n_conf == 65536),
    "configuraciones_unicas_enumeradas": n_unicas,
    "configuracion_observada_incluida": config_observada_incluida,
    "count_extreme": count_extreme,
    "conteo_de_extremas": count_extreme,
    "count_extreme_strictly_greater": count_strict,
    "count_extreme_equal": count_equal,
    "p_value_exact": p_exact,
    "p_value_exacto_cambios_de_signo": p_exact,
    "p_value_exact_fraccion": f"{count_extreme}/{n_conf}",
    "p_value_formula": "p = (# configuraciones con |media nula| >= |media observada|) / 65536",
    "incluye_configuracion_observada": True,
    "verificacion_coma_flotante": {
        "count_extreme_float": count_extreme_float,
        "coincide_con_entero": verif_float_ok,
    },
    "prueba_t_comparable": {
        "nombre": "prueba t de una muestra sobre el vector de diferencias (equivalente a la t pareada)",
        "funcion_scipy": "scipy.stats.ttest_1samp",
        "media_nula": 0.0,
        "alternative": "two-sided",
        "statistic": t_stat,
        "degrees_of_freedom": gl,
        "p_value_two_sided": p_t,
        "p_value_verificacion_sf": p_t_sf,
    },
    "t_statistic": t_stat,
    "degrees_of_freedom": gl,
    "p_value_t": p_t,
    "comparacion_p_values": {
        "p_value_t_bilateral": p_t,
        "p_value_exacto_signflip": p_exact,
        "diferencia_abs": float(abs(p_exact - p_t)),
        "razon_signflip_sobre_t": float(p_exact / p_t),
        "lectura": (
            "ambos p-values son cercanos y llevan a la misma decisión (no rechazar H0 al 5%), pero "
            "provienen de referencias nulas distintas: la t aproxima la nula con una distribución t "
            "de Student con 15 gl (supone diferencias normales i.i.d.), mientras que el cambio de "
            "signo construye la nula exacta por enumeración de las 65536 medias (supone que, bajo "
            "H0, cada diferencia es simétrica alrededor de 0)"
        ),
    },
    "distribucion_nula": dist_summary,
    "referencia_nula_unicos": referencia_unicos,
    "figura": "sign_flip_null_distribution.png",
    "verificacion_con_subtareas_previas": {"T2": verif_t2, "T4": verif_t4},
    "notas_para_informe": {
        "p_value_condicional_a_H0": (
            "el p-value es la probabilidad, calculada bajo la hipótesis nula, de observar un "
            "estadístico al menos tan extremo como el obtenido; NO es la probabilidad de que H0 "
            "sea cierta"
        ),
        "diseno": (
            "16 tiendas medidas con ambos modelos (diseño pareado); la unidad de análisis es la "
            "diferencia por tienda y la estadística es la media de las diferencias"
        ),
        "supuestos_prueba_t": (
            "bajo H0 (media de las diferencias = 0) el estadístico t sigue una distribución t de "
            "Student con n-1 = 15 gl; esto supone diferencias i.i.d. con distribución "
            "aproximadamente normal (o n grande)"
        ),
        "supuestos_cambios_de_signo": (
            "bajo H0 cada diferencia proviene de una distribución simétrica alrededor de 0, de "
            "modo que asignar signos +1/-1 a las diferencias observadas genera datos igualmente "
            "plausibles; no requiere normalidad, pero sí simetría de cada diferencia"
        ),
        "no_intercambiables": (
            "la t y los cambios de signo no son procedimientos intercambiables: usan referencias "
            "nulas distintas (t de Student continua vs. distribución exacta discreta de las 65536 "
            "medias) y supuestos distintos (normalidad vs. simetría); que sus p-values numéricos "
            "sean cercanos es una coincidencia empírica de estos datos, no una equivalencia de "
            "procedimientos"
        ),
    },
    "salidas": {
        "resultados_json": "resultados.json",
        "figura_nula": "sign_flip_null_distribution.png",
        "tabla_csv": "output/store_differences.csv",
        "figura_diferencias": "output/store_differences.png",
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 10) Resumen breve
# ----------------------------------------------------------------------------
print("=" * 72)
print("T5 - Referencia nula exacta por cambios de signo")
print(f"  n tiendas (pares)                  : {n}")
print(f"  media observada de las diferencias : {obs_mean:.10f}  (|media| = {obs_abs:.10f})")
print(f"  configuraciones enumeradas         : {n_conf} (únicas: {n_unicas})")
print(f"  configuraciones extremas           : {count_extreme} "
      f"(estrictamente mayores: {count_strict}, iguales: {count_equal})")
print(f"  p-value exacto (cambios de signo)  : {count_extreme}/{n_conf} = {p_exact:.10f}")
print(f"  p-value t bilateral (comparable)   : {p_t:.10f}  (t = {t_stat:.6f}, gl = {gl})")
print(f"  diferencia absoluta entre p-values : {abs(p_exact - p_t):.10f}")
print("  Archivos escritos: resultados.json, sign_flip_null_distribution.png,")
print("                     output/store_differences.csv, output/store_differences.png")
print("=" * 72)
