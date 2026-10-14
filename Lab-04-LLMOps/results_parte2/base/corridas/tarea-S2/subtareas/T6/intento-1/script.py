# ============================================================================
# T6 - Bootstrap pareado (IC percentil 95%, semilla 20260829, 10000 remuestras)
# ----------------------------------------------------------------------------
# - np.random.default_rng(SEED = 20260829), BOOTSTRAP_RESAMPLES = 10_000
# - Se remuestrean TIENDAS COMPLETAS mediante el vector de diferencias
#   (nunca error_a y error_b por separado).
# - Se reporta el intervalo percentil del 95 % como análisis de sensibilidad.
# - Se verifica reproducibilidad: dos ejecuciones con la misma semilla deben
#   producir exactamente el mismo intervalo.
# ============================================================================
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------- Parámetros del enunciado -------------------------
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
NIVEL_CONFIANZA = 0.95
ALFA = 1.0 - NIVEL_CONFIANZA

RUTA_CSV = Path("data/model_errors.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_SALIDA_CSV = Path("output/store_differences.csv")
RUTA_SALIDA_PNG = Path("output/store_differences.png")
RUTA_RESULTADOS = Path("resultados.json")

# ------------------- 1) Carga y validación ligera de datos ------------------
if not RUTA_CSV.exists():
    raise FileNotFoundError(f"No se encuentra el archivo de datos: {RUTA_CSV}")
df = pd.read_csv(RUTA_CSV)

columnas_exigidas = ["store_id", "error_a", "error_b"]
faltantes = [c for c in columnas_exigidas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas exigidas en {RUTA_CSV}: {faltantes}")
if df["store_id"].isna().any() or df["store_id"].duplicated().any():
    raise ValueError("El identificador store_id tiene valores nulos o duplicados")
for col in ("error_a", "error_b"):
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError(f"La columna {col} contiene valores no numéricos o no finitos")
    if (vals < 0).any():
        raise ValueError(f"La columna {col} contiene valores negativos")
    df[col] = vals

# ---------------- 2) Tabla pareada: difference = error_a - error_b ----------
df["difference"] = df["error_a"] - df["error_b"]
df["favors"] = np.where(df["difference"] > 0, "B",
                        np.where(df["difference"] < 0, "A", "tie"))

# Contraste con la subtarea T2 (si sus resultados están disponibles)
coincide_con_T2 = None
if RUTA_T2.exists():
    try:
        with open(RUTA_T2, "r", encoding="utf-8") as f:
            t2 = json.load(f)
        tabla_t2 = {fila["store_id"]: float(fila["difference"])
                    for fila in t2.get("tabla_pareada", [])}
        if tabla_t2:
            coincide_con_T2 = bool(
                len(tabla_t2) == len(df)
                and all(np.isclose(tabla_t2[sid], d, rtol=0, atol=1e-9)
                        for sid, d in zip(df["store_id"], df["difference"]))
            )
    except Exception:
        coincide_con_T2 = None

diffs = df["difference"].to_numpy(dtype=float)
error_a = df["error_a"].to_numpy(dtype=float)
error_b = df["error_b"].to_numpy(dtype=float)
n_tiendas = int(len(diffs))
media_observada = float(diffs.mean())

# ------------------------- 3) Bootstrap pareado -----------------------------
def bootstrap_pareado(diferencias, n_remuestras, semilla):
    """Remuestrea TIENDAS COMPLETAS: sortea índices de tiendas con reemplazo y
    los aplica al vector de diferencias (equivalente a remuestrear las filas
    (error_a, error_b) conjuntamente). Nunca remuestrea A y B por separado."""
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, len(diferencias),
                           size=(n_remuestras, len(diferencias)))
    medias = diferencias[indices].mean(axis=1)
    return medias, indices

# Ejecución 1 y Ejecución 2 con la MISMA semilla (verificación de reproducibilidad)
boot_1, idx_1 = bootstrap_pareado(diffs, BOOTSTRAP_RESAMPLES, SEED)
boot_2, idx_2 = bootstrap_pareado(diffs, BOOTSTRAP_RESAMPLES, SEED)

p_pct = [100.0 * ALFA / 2.0, 100.0 * (1.0 - ALFA / 2.0)]  # [2.5, 97.5]
ic_1 = np.percentile(boot_1, p_pct)
ic_2 = np.percentile(boot_2, p_pct)

intervalos_identicos = bool(np.array_equal(ic_1, ic_2))
medias_identicas = bool(np.array_equal(boot_1, boot_2))
indices_identicos = bool(np.array_equal(idx_1, idx_2))
max_diff_ejecuciones = float(np.max(np.abs(boot_1 - boot_2)))

ic_inf, ic_sup = float(ic_1[0]), float(ic_1[1])

# Coherencia: remuestrear filas completas (A y B juntas) con los MISMOS índices
# da exactamente las mismas medias que usar el vector de diferencias -> demuestra
# que "remuestrear tiendas completas" == "remuestrear el vector de diferencias".
medias_filas = (error_a[idx_1] - error_b[idx_1]).mean(axis=1)
coherencia_filas_vs_diferencias = bool(np.array_equal(medias_filas, boot_1))

# Contraste ILUSTRATIVO (NO válido, no es el resultado de la tarea): remuestrear
# error_a y error_b por separado rompe el emparejamiento.
rng_sep = np.random.default_rng(SEED)
idx_a = rng_sep.integers(0, n_tiendas, size=(BOOTSTRAP_RESAMPLES, n_tiendas))
idx_b = rng_sep.integers(0, n_tiendas, size=(BOOTSTRAP_RESAMPLES, n_tiendas))
medias_separado = (error_a[idx_a] - error_b[idx_b]).mean(axis=1)
ic_sep = np.percentile(medias_separado, p_pct)

# Sensibilidad informativa a la semilla (el IC OFICIAL es el de SEED)
boot_otra, _ = bootstrap_pareado(diffs, BOOTSTRAP_RESAMPLES, SEED + 1)
ic_otra = np.percentile(boot_otra, p_pct)

# ---------------- 4) Estadísticos resumen del bootstrap oficial -------------
boot_media = float(boot_1.mean())
boot_se = float(boot_1.std(ddof=1))
boot_sesgo = boot_media - media_observada
prop_mayor_cero = float((boot_1 > 0).mean())
prop_menor_cero = float((boot_1 < 0).mean())
ic_contiene_cero = bool(ic_inf <= 0.0 <= ic_sup)

# ------------------------- 5) Figura (PNG exigido) --------------------------
RUTA_SALIDA_PNG.parent.mkdir(parents=True, exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = ejes[0]
colores = np.where(diffs > 0, "#1f77b4", "#d62728")
ax.bar(df["store_id"], diffs, color=colores, edgecolor="black", linewidth=0.4)
ax.axhline(0.0, color="black", lw=0.8)
ax.axhline(media_observada, color="green", ls="--", lw=1.3,
           label=f"media = {media_observada:.4f}")
ax.set_xlabel("Tienda")
ax.set_ylabel("difference = error_a - error_b")
ax.set_title("Diferencias pareadas por tienda (unidad de remuestreo)")
ax.tick_params(axis="x", rotation=90)
ax.legend(handles=[Patch(color="#1f77b4", label="favorece a B (d>0)"),
                   Patch(color="#d62728", label="favorece a A (d<0)")],
          fontsize=8, loc="lower right")

ax = ejes[1]
ax.hist(boot_1, bins=60, color="#9ecae1", edgecolor="white")
ax.axvline(media_observada, color="green", ls="--", lw=1.6, label="media observada")
ax.axvline(ic_inf, color="red", lw=1.6)
ax.axvline(ic_sup, color="red", lw=1.6, label="IC percentil 95 %")
ax.axvline(0.0, color="black", ls=":", lw=1.2, label="0")
ax.set_xlabel("Media de la diferencia en cada remuestra bootstrap")
ax.set_ylabel("Frecuencia")
ax.set_title(f"Bootstrap pareado: {BOOTSTRAP_RESAMPLES} remuestras, semilla {SEED}")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(RUTA_SALIDA_PNG, dpi=120)
plt.close(fig)

# -------------------------- 6) CSV exigido ----------------------------------
RUTA_SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
df[["store_id", "error_a", "error_b", "difference", "favors"]].to_csv(
    RUTA_SALIDA_CSV, index=False)

# ------------------------- 7) resultados.json -------------------------------
tabla_salida = [
    {"store_id": str(sid), "error_a": float(a), "error_b": float(b),
     "difference": float(d), "favors": str(fv)}
    for sid, a, b, d, fv in zip(df["store_id"], df["error_a"], df["error_b"],
                                df["difference"], df["favors"])
]

explicacion_tiendas_completas = (
    "La unidad experimental es la tienda: cada una aporta un par (error_a, error_b) "
    "medido sobre el mismo caso, de modo que la cantidad que varía entre unidades es "
    "la diferencia d_i = error_a - error_b. Remuestrear tiendas completas (sortear "
    "índices de tiendas con reemplazo y aplicarlos al vector de diferencias) preserva "
    "la estructura de pares y la correlación dentro de cada tienda, que es lo que hace "
    "que el diseño pareado cancele la variabilidad entre tiendas. Si se remuestrearan "
    "error_a y error_b por separado se romperían los pares, se destruiría esa "
    "cancelación y se simularía un experimento de muestras independientes distinto del "
    "que se ejecutó, produciendo un intervalo más amplio y ajeno al estimando pareado."
)
problema_no_corregido = (
    "El bootstrap cuantifica la incertidumbre de muestreo del estadístico, pero no "
    "repara problemas del diseño ni de los datos: no corrige un emparejamiento erróneo "
    "o incompleto por store_id, ni sesgos sistemáticos de medición o selección, ni la "
    "falta de representatividad de las 16 tiendas respecto de la población de interés, "
    "ni dependencias entre tiendas (p. ej., tiendas medidas en periodos correlacionados). "
    "Además, con n=16 el remuestreo solo recicla los valores observados: los valores "
    "extremos o los pares mal medidos quedan integrados en la distribución bootstrap. "
    "Que el intervalo excluya (o incluya) cero no valida por sí solo el emparejamiento "
    "ni la calidad de los datos."
)

resultados = {
    "subtarea": "T6: bootstrap pareado (IC percentil 95 %, semilla 20260829, 10000 remuestras)",
    "semilla": SEED,
    "numero_de_remuestras": BOOTSTRAP_RESAMPLES,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "nivel_confianza": NIVEL_CONFIANZA,
    "percentiles_usados": p_pct,
    "metodo_percentil": "lineal (np.percentile, interpolación por defecto)",
    "estadistico_bootstrap": "media de difference = error_a - error_b",
    "unidad_de_remuestreo": "tienda completa (vector de diferencias; nunca error_a y error_b por separado)",
    "metodo": ("np.random.default_rng(20260829); se sortean índices de tiendas con "
               "reemplazo y se aplican al vector de diferencias"),
    "n_tiendas": n_tiendas,
    "diferencia_media_observada": media_observada,
    "diferencias_por_tienda": tabla_salida,
    "ic_percentil_95": {"limite_inferior": ic_inf, "limite_superior": ic_sup},
    "ic_percentil_95_inferior": ic_inf,
    "ic_percentil_95_superior": ic_sup,
    "ic_contiene_cero": ic_contiene_cero,
    "bootstrap_media": boot_media,
    "bootstrap_error_estandar": boot_se,
    "bootstrap_sesgo": boot_sesgo,
    "proporcion_bootstrap_mayor_que_cero": prop_mayor_cero,
    "proporcion_bootstrap_menor_que_cero": prop_menor_cero,
    "reproducibilidad_comprobada": intervalos_identicos,
    "reproducibilidad": {
        "ejecucion_1": {"semilla": SEED, "n_remuestras": BOOTSTRAP_RESAMPLES,
                        "ic_inferior": float(ic_1[0]), "ic_superior": float(ic_1[1])},
        "ejecucion_2": {"semilla": SEED, "n_remuestras": BOOTSTRAP_RESAMPLES,
                        "ic_inferior": float(ic_2[0]), "ic_superior": float(ic_2[1])},
        "intervalos_identicos": intervalos_identicos,
        "medias_bootstrap_identicas": medias_identicas,
        "indices_remuestreados_identicos": indices_identicos,
        "max_diferencia_absoluta_entre_ejecuciones": max_diff_ejecuciones,
    },
    "coherencia_remuestreo_filas_completas_vs_vector_diferencias": coherencia_filas_vs_diferencias,
    "contraste_ilustrativo_remuestreo_independiente_no_valido": {
        "descripcion": ("Solo ilustrativo: remuestrear error_a y error_b por separado "
                        "rompe el emparejamiento; NO es el resultado de la tarea."),
        "ic_percentil_95": {"limite_inferior": float(ic_sep[0]),
                            "limite_superior": float(ic_sep[1])},
    },
    "sensibilidad_informativa_otra_semilla": {
        "semilla": SEED + 1,
        "ic_percentil_95": {"limite_inferior": float(ic_otra[0]),
                            "limite_superior": float(ic_otra[1])},
        "nota": "Informativo; el IC oficial corresponde a la semilla 20260829.",
    },
    "explicacion_remuestrear_tiendas_completas": explicacion_tiendas_completas,
    "problema_no_corregido_por_bootstrap": problema_no_corregido,
    "coincide_con_T2": coincide_con_T2,
    "salidas": {
        "resultados_json": "resultados.json",
        "tabla_csv": "output/store_differences.csv",
        "figura_png": "output/store_differences.png",
    },
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------ 8) Resumen ----------------------------------
print("=" * 72)
print("T6 - Bootstrap pareado (remuestreo de tiendas completas)")
print("=" * 72)
print(f"Datos: {RUTA_CSV} | tiendas: {n_tiendas} | coincide con T2: {coincide_con_T2}")
print(f"Semilla: {SEED} | Remuestras: {BOOTSTRAP_RESAMPLES} | Nivel: {NIVEL_CONFIANZA:.0%}")
print(f"Media observada de difference: {media_observada:.6f}")
print(f"IC percentil 95 %: [{ic_inf:.6f}, {ic_sup:.6f}] | contiene cero: {ic_contiene_cero}")
print(f"Bootstrap: media={boot_media:.6f} | EE={boot_se:.6f} | sesgo={boot_sesgo:.6f}")
print(f"P(boot media > 0) = {prop_mayor_cero:.4f} | P(boot media < 0) = {prop_menor_cero:.4f}")
print(f"Reproducibilidad (misma semilla, 2 ejecuciones): intervalos idénticos = "
      f"{intervalos_identicos} | max |Δ| = {max_diff_ejecuciones}")
print(f"Coherencia filas completas vs vector de diferencias: {coherencia_filas_vs_diferencias}")
print(f"Contraste (no válido, A y B por separado): IC = [{ic_sep[0]:.6f}, {ic_sep[1]:.6f}]")
print(f"Salidas: {RUTA_RESULTADOS}, {RUTA_SALIDA_CSV}, {RUTA_SALIDA_PNG}")
