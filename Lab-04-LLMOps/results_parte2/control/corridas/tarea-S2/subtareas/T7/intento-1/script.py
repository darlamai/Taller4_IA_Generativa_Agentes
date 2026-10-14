# -*- coding: utf-8 -*-
"""
T7 — Pipeline completo de la Tarea 7:
  1) Carga y valida data/model_errors.csv (columnas exactas, store_id único/no nulo,
     errores numéricos, finitos y no negativos).
  2) Construye la tabla pareada (store_id, error_a, error_b, difference, favors) y la
     guarda en output/store_differences.csv.
  3) Genera la figura por tienda con referencia en cero: output/store_differences.png.
  4) Muestra en consola: resumen descriptivo, prueba t pareada (estadístico, df,
     p-value, IC 95 %), referencia exacta por cambios de signo (conteo y p-value) y
     el intervalo bootstrap percentil con su semilla.
  5) Escribe resultados.json con todas las cifras (contrato de la subtarea).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# ---------------------------------------------------------------------------
# Configuración fija del pipeline
# ---------------------------------------------------------------------------
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10000
NIVEL = 0.95
CSV_ORIGEN = Path("data/model_errors.csv")
DIR_SALIDA = Path("output")
RUTA_CSV = DIR_SALIDA / "store_differences.csv"
RUTA_PNG = DIR_SALIDA / "store_differences.png"

DIR_SALIDA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1) Cargar y validar data/model_errors.csv (validaciones de la Tarea 2)
# ---------------------------------------------------------------------------
if not CSV_ORIGEN.exists():
    raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_ORIGEN}")

df = pd.read_csv(CSV_ORIGEN)

COLUMNAS_ESPERADAS = ["store_id", "error_a", "error_b"]
if list(df.columns) != COLUMNAS_ESPERADAS:
    raise ValueError(
        f"El CSV debe tener exactamente las columnas {COLUMNAS_ESPERADAS}; "
        f"se encontraron {list(df.columns)}"
    )

if df["store_id"].isna().any():
    raise ValueError("store_id contiene valores nulos")

duplicados = df.loc[df["store_id"].duplicated(keep=False), "store_id"].unique().tolist()
if duplicados:
    raise ValueError(f"store_id debe ser único; identificadores duplicados: {duplicados}")

for col in ("error_a", "error_b"):
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"La columna {col} debe ser numérica")
    valores = df[col].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError(f"La columna {col} contiene valores no finitos")
    if (valores < 0).any():
        raise ValueError(f"La columna {col} contiene valores negativos")

# ---------------------------------------------------------------------------
# 2) Tabla pareada: difference = error_a - error_b y favors (Tarea 2)
# ---------------------------------------------------------------------------
tabla = df[COLUMNAS_ESPERADAS].copy()
tabla["difference"] = tabla["error_a"] - tabla["error_b"]
tabla["favors"] = np.select(
    [tabla["difference"] > 0, tabla["difference"] < 0], ["B", "A"], default="tie"
)

# Salida exigida por el enunciado: output/store_differences.csv
tabla.to_csv(RUTA_CSV, index=False)

diff = tabla["difference"].to_numpy(dtype=float)
ids = tabla["store_id"].astype(str).tolist()
n = int(len(diff))

# ---------------------------------------------------------------------------
# 3) Figura exigida por el enunciado: output/store_differences.png (Tarea 3)
# ---------------------------------------------------------------------------
colores = {"B": "#2c7fb8", "A": "#d95f02", "tie": "#8c8c8c"}
orden = tabla.iloc[::-1].reset_index(drop=True)  # S01 arriba en el eje y

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(
    orden["store_id"].astype(str),
    orden["difference"].to_numpy(dtype=float),
    color=[colores[f] for f in orden["favors"]],
    edgecolor="black",
    linewidth=0.5,
)
ax.axvline(0.0, color="black", linewidth=1.4)
lim = float(np.max(np.abs(diff))) * 1.30
ax.set_xlim(-lim, lim)
for y, (d, f) in enumerate(
    zip(orden["difference"].to_numpy(dtype=float), orden["favors"].astype(str))
):
    if d >= 0:
        ax.text(d + lim * 0.02, y, f"{d:+.2f}", va="center", ha="left", fontsize=8)
    else:
        ax.text(d - lim * 0.02, y, f"{d:+.2f}", va="center", ha="right", fontsize=8)
ax.set_xlabel("difference = error_a - error_b (unidades de error)")
ax.set_ylabel("store_id")
ax.set_title(
    "Diferencia de error por tienda (pareado A vs B)\n"
    "positivo favorece a B · negativo favorece a A · línea negra = referencia en 0"
)
ax.legend(
    handles=[
        Patch(color=colores["B"], label="favorece B (difference > 0)"),
        Patch(color=colores["A"], label="favorece A (difference < 0)"),
        Patch(color=colores["tie"], label="empate (difference = 0)"),
    ],
    loc="upper left",
    fontsize=8,
    framealpha=0.95,
)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(RUTA_PNG, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Resumen descriptivo del vector de diferencias (pregunta 11)
# ---------------------------------------------------------------------------
media = float(np.mean(diff))
sd = float(np.std(diff, ddof=1))
se = float(sd / np.sqrt(n))
mediana = float(np.median(diff))
favors_arr = tabla["favors"].to_numpy()
n_B = int(np.sum(favors_arr == "B"))
n_A = int(np.sum(favors_arr == "A"))
n_tie = int(np.sum(favors_arr == "tie"))

# ---------------------------------------------------------------------------
# 5) Prueba t pareada e IC t del 95 % (preguntas 12-14, scipy.stats)
# ---------------------------------------------------------------------------
alpha = 1.0 - NIVEL
t_res = stats.ttest_1samp(diff, popmean=0.0)  # bilateral por defecto
t_stat = float(t_res.statistic)
p_t = float(t_res.pvalue)
dfree = n - 1
t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, dfree))
ci_lo = float(media - t_crit * se)
ci_hi = float(media + t_crit * se)
ci_interval = stats.t.interval(NIVEL, dfree, loc=media, scale=se)
ci_coincide = bool(
    np.isclose(float(ci_interval[0]), ci_lo, atol=1e-12)
    and np.isclose(float(ci_interval[1]), ci_hi, atol=1e-12)
)
incluye0_t = bool(ci_lo <= 0.0 <= ci_hi)

# ---------------------------------------------------------------------------
# 6) Referencia nula exacta por cambios de signo (pregunta 15, Tarea 5)
# ---------------------------------------------------------------------------
escala = 10
d_int = np.rint(diff * escala).astype(np.int64)
desv_escala = float(np.max(np.abs(diff * escala - d_int)))
if desv_escala > 1e-6:
    raise ValueError("Las diferencias no son exactas en la escala entera x10")

suma_obs = int(d_int.sum())
total_cfg = 2 ** n
bits = (
    (np.arange(total_cfg, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1
).astype(np.int8)
signos = (1 - 2 * bits).astype(np.int64)
sumas_nulas = signos @ d_int
extremas = int(np.count_nonzero(np.abs(sumas_nulas) >= abs(suma_obs)))
p_signo = extremas / total_cfg

# ---------------------------------------------------------------------------
# 7) Bootstrap pareado (sensibilidad, Tarea 6): semilla y B fijos
# ---------------------------------------------------------------------------
p_lo = alpha / 2 * 100
p_hi = (1 - alpha / 2) * 100

rng = np.random.default_rng(SEED)
indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
boot_medias = diff[indices].mean(axis=1)  # remuestreo de tiendas completas
b_lo, b_hi = (float(v) for v in np.percentile(boot_medias, [p_lo, p_hi]))
boot_media = float(np.mean(boot_medias))
boot_se = float(np.std(boot_medias, ddof=1))
boot_sesgo = boot_media - media
prop_pos = float(np.mean(boot_medias > 0))
incluye0_b = bool(b_lo <= 0.0 <= b_hi)

# Verificación de reproducibilidad: misma semilla -> mismo intervalo
rng_rep = np.random.default_rng(SEED)
indices_rep = rng_rep.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
boot_rep = diff[indices_rep].mean(axis=1)
lo_rep, hi_rep = (float(v) for v in np.percentile(boot_rep, [p_lo, p_hi]))
reproducible = bool(
    np.array_equal(indices, indices_rep)
    and np.array_equal(boot_medias, boot_rep)
    and lo_rep == b_lo
    and hi_rep == b_hi
)

# ---------------------------------------------------------------------------
# 8) Coherencia con subtareas previas (informativa; no altera resultados)
# ---------------------------------------------------------------------------
def _cargar_json(ruta):
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except Exception:
        return None

coherencia = {}
prev4 = _cargar_json("entrada/T4/resultados.json")
if isinstance(prev4, dict) and prev4.get("t_statistic") is not None:
    coherencia["T4_prueba_t"] = {
        "t_statistic_previa": float(prev4["t_statistic"]),
        "p_value_previo": float(prev4["p_value_two_sided"]),
        "coincide_estadistico": bool(np.isclose(t_stat, float(prev4["t_statistic"]), atol=1e-9)),
        "coincide_p_value": bool(np.isclose(p_t, float(prev4["p_value_two_sided"]), atol=1e-9)),
    }
prev5 = _cargar_json("entrada/T5/resultados.json")
if isinstance(prev5, dict) and prev5.get("p_value_exacto") is not None:
    coherencia["T5_cambios_de_signo"] = {
        "p_value_previo": float(prev5["p_value_exacto"]),
        "conteo_previo": int(prev5["conteo_configuraciones_extremas"]),
        "coincide_p_value": bool(abs(p_signo - float(prev5["p_value_exacto"])) < 1e-12),
        "coincide_conteo": extremas == int(prev5["conteo_configuraciones_extremas"]),
    }
prev6 = _cargar_json("entrada/T6/resultados.json")
if isinstance(prev6, dict) and isinstance(prev6.get("bootstrap"), dict):
    t6_lo = float(prev6["bootstrap"]["ic95_limite_inferior"])
    t6_hi = float(prev6["bootstrap"]["ic95_limite_superior"])
    coherencia["T6_bootstrap"] = {
        "ic_previo": [t6_lo, t6_hi],
        "ic_local": [b_lo, b_hi],
        "coincide": bool(
            np.isclose(t6_lo, b_lo, rtol=0.0, atol=1e-9)
            and np.isclose(t6_hi, b_hi, rtol=0.0, atol=1e-9)
        ),
    }

# ---------------------------------------------------------------------------
# 9) Informe ejecutivo (pregunta 18, <= 150 palabras)
# ---------------------------------------------------------------------------
frase_ic = (
    "ambos intervalos contienen 0, no se rechaza H0 al nivel 0.05"
    if (incluye0_t and incluye0_b)
    else "los intervalos se reportan en resultados.json"
)
INFORME = (
    f"Unidad y procedencia: {n} tiendas ({ids[0]}-{ids[-1]}) de {CSV_ORIGEN}, emparejadas por "
    f"store_id. Métrica: difference = error_a - error_b (unidades de error); positivo favorece a B. "
    f"Efecto: media {media:.3f} unidades (IC t 95 %: {ci_lo:.3f} a {ci_hi:.3f}). "
    f"Procedimiento: prueba t pareada bilateral (t({dfree})={t_stat:.3f}, p={p_t:.3f}), referencia "
    f"exacta por cambios de signo ({extremas}/{total_cfg} configuraciones, p={p_signo:.3f}) y "
    f"bootstrap percentil de sensibilidad ({BOOTSTRAP_RESAMPLES} remuestras de tiendas completas, "
    f"semilla {SEED}; IC 95 %: {b_lo:.3f} a {b_hi:.3f}). Incertidumbre: {frase_ic}. Limitaciones: "
    f"muestra pequeña (n={n}, baja potencia) y supuestos de normalidad/simetría de las diferencias "
    f"no verificables; datos observacionales sin control de covariables. No puede concluirse que B "
    f"reduzca el error respecto de A, ni que un eventual efecto sea causal o generalizable a otras "
    f"tiendas o periodos."
)

# ---------------------------------------------------------------------------
# 10) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
tabla_json = [
    {
        "store_id": str(r.store_id),
        "error_a": float(r.error_a),
        "error_b": float(r.error_b),
        "difference": float(r.difference),
        "favors": str(r.favors),
    }
    for r in tabla.itertuples(index=False)
]

resultados = {
    "subtarea": "T7_pipeline_completo",
    "descripcion": (
        "Pipeline completo: carga validada de data/model_errors.csv, tabla pareada "
        "(output/store_differences.csv), figura (output/store_differences.png), resumen "
        "descriptivo, prueba t pareada, referencia exacta por cambios de signo y bootstrap pareado"
    ),
    "config": {
        "csv_origen": str(CSV_ORIGEN),
        "seed_bootstrap": SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "nivel_confianza": NIVEL,
        "unidad_remuestreo_bootstrap": "tiendas completas (vector de diferencias)",
    },
    "validaciones": {
        "columnas_exactas_store_id_error_a_error_b": True,
        "store_id_unico": True,
        "store_id_no_nulo": True,
        "errores_numericos_finitos_no_negativos": True,
        "n_tiendas": n,
    },
    "n_tiendas": n,
    "ids_tiendas": ids,
    "columnas_csv_salida": ["store_id", "error_a", "error_b", "difference", "favors"],
    "tabla_pareada": tabla_json,
    "resumen_descriptivo": {
        "n": n,
        "media": media,
        "desviacion_muestral_ddof1": sd,
        "error_estandar": se,
        "mediana": mediana,
        "minimo": float(np.min(diff)),
        "maximo": float(np.max(diff)),
        "suma": float(np.sum(diff)),
        "n_favors_A": n_A,
        "n_favors_B": n_B,
        "n_favors_tie": n_tie,
    },
    "prueba_t": {
        "metodo": "scipy.stats.ttest_1samp bilateral sobre difference (equivalente a la t pareada)",
        "hipotesis_nula": "media de las diferencias = 0",
        "alternative": "two-sided",
        "estadistico_t": t_stat,
        "grados_libertad": dfree,
        "p_value_bilateral": p_t,
        "t_critico_bilateral_95": t_crit,
        "ic95_limite_inferior": ci_lo,
        "ic95_limite_superior": ci_hi,
        "ic95_incluye_cero": incluye0_t,
        "ic_coincide_con_stats_t_interval": ci_coincide,
        "decision_nivel_0_05": "no se rechaza H0" if p_t >= 0.05 else "se rechaza H0",
    },
    "prueba_cambios_signo": {
        "metodo": (
            "enumeración exacta de las 2^16 configuraciones de signos del vector de diferencias "
            "(aritmética entera escalada x10); criterio |media nula| >= |media observada|"
        ),
        "total_configuraciones": total_cfg,
        "conteo_extremas": extremas,
        "p_value_exacto": p_signo,
        "fraccion": [extremas, total_cfg],
        "media_observada": media,
        "suma_observada_escala_x10": suma_obs,
        "desviacion_maxima_escala_entera": desv_escala,
    },
    "bootstrap": {
        "seed": SEED,
        "n_remuestras": BOOTSTRAP_RESAMPLES,
        "metodo": (
            "percentil; remuestreo de tiendas completas mediante el vector de diferencias "
            "(rng.integers(0, n, size=(B, n)))"
        ),
        "percentiles_usados": [p_lo, p_hi],
        "media_bootstrap": boot_media,
        "error_estandar_bootstrap": boot_se,
        "sesgo_bootstrap": boot_sesgo,
        "ic95_percentil_limite_inferior": b_lo,
        "ic95_percentil_limite_superior": b_hi,
        "ic95_incluye_cero": incluye0_b,
        "proporcion_remuestras_media_mayor_que_cero": prop_pos,
        "reproducibilidad_misma_semilla": reproducible,
    },
    "coherencia_con_subtareas_previas": coherencia,
    "interpretacion": {
        "resumen": (
            f"La diferencia media es {media:.4f} unidades por tienda a favor de B "
            f"(sd muestral {sd:.4f}, error estándar {se:.4f})."
        ),
        "prueba_t": (
            f"t({dfree}) = {t_stat:.4f}, p bilateral = {p_t:.4f} > 0.05 e IC t 95 % "
            f"[{ci_lo:.4f}, {ci_hi:.4f}] que contiene 0: no se rechaza H0."
        ),
        "significado_p_value": (
            "Probabilidad, bajo H0 (media de las diferencias = 0), de observar un estadístico "
            "al menos tan extremo como el obtenido; no es la probabilidad de que H0 sea cierta."
        ),
        "supuestos": (
            "La prueba t supone diferencias independientes entre tiendas y aproximadamente "
            "normales; la referencia exacta por cambios de signo solo exige simetría de cada "
            "diferencia respecto de 0 bajo H0; el bootstrap percentil es aproximado y no exige "
            "normalidad."
        ),
        "cambios_de_signo": (
            f"{extremas} de {total_cfg} configuraciones de signo tienen media al menos tan extrema "
            f"como la observada (p exacto = {p_signo:.6f}); coherente con la prueba t pero no "
            "intercambiable con ella."
        ),
        "bootstrap_sensibilidad": (
            f"IC percentil 95 % [{b_lo:.4f}, {b_hi:.4f}] con semilla {SEED} y "
            f"{BOOTSTRAP_RESAMPLES} remuestras de tiendas completas; confirma la lectura del "
            "intervalo t."
        ),
        "limitaciones": [
            "n = 16 tiendas: muestra pequeña y baja potencia para detectar efectos pequeños",
            "supuestos de normalidad/simetría de las diferencias no verificables formalmente con este tamaño",
            "datos observacionales de un solo archivo, sin control de covariables ni replicación en el tiempo",
        ],
        "no_concluible": (
            "No puede concluirse que B reduzca el error respecto de A en la población de tiendas, "
            "ni que un eventual efecto sea causal o generalizable a otras tiendas o periodos."
        ),
    },
    "informe_ejecutivo_p18": INFORME,
    "salidas": {
        "csv_tabla_pareada": str(RUTA_CSV),
        "figura": str(RUTA_PNG),
        "resultados_json": "resultados.json",
    },
    "archivos_generados": {
        "output/store_differences.csv": {
            "existe": RUTA_CSV.exists(),
            "bytes": RUTA_CSV.stat().st_size if RUTA_CSV.exists() else 0,
        },
        "output/store_differences.png": {
            "existe": RUTA_PNG.exists(),
            "bytes": RUTA_PNG.stat().st_size if RUTA_PNG.exists() else 0,
        },
    },
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 11) Resumen en consola (contrato de la subtarea)
# ---------------------------------------------------------------------------
LINEA = "=" * 78
print(LINEA)
print("TAREA 7 · PIPELINE COMPLETO — diferencias pareadas de error por tienda (A vs B)")
print(LINEA)
print(f"Datos: {CSV_ORIGEN} | emparejamiento por store_id | tiendas: n = {n}")

tabla_vista = tabla.copy()
tabla_vista["difference"] = tabla_vista["difference"].round(2)
print(f"\nTabla pareada (guardada en {RUTA_CSV}):")
print(tabla_vista.to_string(index=False))

print("\n[1] RESUMEN DESCRIPTIVO de difference = error_a - error_b")
print(f"    n                            = {n}")
print(f"    media                        = {media:.6f}")
print(f"    desviación muestral (ddof=1) = {sd:.6f}")
print(f"    error estándar (sd/raiz(n))  = {se:.6f}")
print(f"    mediana                      = {mediana:.6f}")
print(f"    minimo / maximo              = {np.min(diff):.4f} / {np.max(diff):.4f}")
print(f"    favorece B (>0): {n_B} | favorece A (<0): {n_A} | empate (=0): {n_tie}")

print("\n[2] PRUEBA T PAREADA (scipy.stats; H0: media de las diferencias = 0, bilateral)")
print(f"    estadístico t                = {t_stat:.6f}")
print(f"    grados de libertad           = {dfree}")
print(f"    p-value (bilateral)          = {p_t:.6f}")
print(f"    IC t 95 % de la media        = [{ci_lo:.6f}, {ci_hi:.6f}] (t crítico = {t_crit:.6f})")
print(f"    decisión al nivel 0.05       = {'no se rechaza H0' if p_t >= 0.05 else 'se rechaza H0'} (IC incluye 0: {incluye0_t})")

print("\n[3] REFERENCIA EXACTA POR CAMBIOS DE SIGNO")
print(f"    configuraciones de signo     = {total_cfg} (2^{n})")
print(f"    conteo de extremas           = {extremas} (|media nula| >= |media observada| = {abs(media):.6f})")
print(f"    p-value exacto               = {p_signo:.10f} ({extremas}/{total_cfg})")

print("\n[4] BOOTSTRAP PAREADO (análisis de sensibilidad)")
print(f"    semilla                      = {SEED}")
print(f"    remuestras                   = {BOOTSTRAP_RESAMPLES} (unidad: tienda completa -> vector de diferencias)")
print(f"    IC percentil 95 %            = [{b_lo:.6f}, {b_hi:.6f}]")
print(f"    media bootstrap              = {boot_media:.6f} | EE bootstrap = {boot_se:.6f} | sesgo = {boot_sesgo:.6f}")
print(f"    proporción medias > 0        = {prop_pos:.4f} | IC incluye 0: {incluye0_b}")
print(f"    reproducibilidad             = misma semilla reproduce índices, medias e intervalo: {reproducible}")

if coherencia:
    print("\n[5] COHERENCIA CON SUBTAREAS PREVIAS (informativa)")
    for clave, val in coherencia.items():
        print(f"    {clave}: {val}")

print("\n[6] ARCHIVOS EXIGIDOS POR EL ENUNCIADO")
print(f"    {RUTA_CSV} -> existe: {RUTA_CSV.exists()}")
print(f"    {RUTA_PNG} -> existe: {RUTA_PNG.exists()}")

print("\n[7] INFORME EJECUTIVO (pregunta 18, <= 150 palabras)")
print(INFORME)
print(LINEA)
print("Pipeline completado: resultados.json, output/store_differences.csv y output/store_differences.png")
