# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado.

Remuestrea TIENDAS COMPLETAS mediante el vector de diferencias
(difference = error_a - error_b, una diferencia por tienda), usando
np.random.default_rng(SEED = 20260829) y BOOTSTRAP_RESAMPLES = 10_000.
Reporta el intervalo percentil del 95 % (análisis de sensibilidad aproximado)
y comprueba que dos ejecuciones con la misma semilla producen el mismo
intervalo. Los errores de A y de B NUNCA se remuestrean por separado: cada
extracción elige una tienda y arrastra consigo su diferencia completa.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------- Parámetros del enunciado ---------------------------
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
CONF_LEVEL = 0.95

DATA_PATH = Path("data/model_errors.csv")
T2_RESULTS_PATH = Path("entrada/T2/resultados.json")
OUT_DIR = Path("output")
CSV_OUT = OUT_DIR / "store_differences.csv"
FIG_OUT = OUT_DIR / "store_differences.png"
FIG_REPRO_PATH = Path("bootstrap_reproducibilidad.png")
RESULTS_PATH = Path("resultados.json")

# ----------------------------- Carga y validación -------------------------------
df = pd.read_csv(DATA_PATH)

requeridas = ["store_id", "error_a", "error_b"]
faltantes = [c for c in requeridas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan las columnas {faltantes} en {DATA_PATH}")

if df["store_id"].isna().any():
    raise ValueError("store_id contiene valores nulos")
if df["store_id"].duplicated().any():
    duplicados = df.loc[df["store_id"].duplicated(), "store_id"].tolist()
    raise ValueError(f"store_id duplicado (identificador repetido): {duplicados}")

for col in ("error_a", "error_b"):
    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"La columna '{col}' no es numérica")

errores = df[["error_a", "error_b"]].to_numpy(dtype=float)
if not np.isfinite(errores).all():
    raise ValueError("Los errores contienen valores no finitos (NaN/inf)")
if (errores < 0).any():
    raise ValueError("Los errores deben ser no negativos")

# Orden determinista por identificador de tienda (S01..S16)
df = df.sort_values("store_id", kind="mergesort").reset_index(drop=True)

store_ids = df["store_id"].astype(str).tolist()
error_a = df["error_a"].to_numpy(dtype=float)
error_b = df["error_b"].to_numpy(dtype=float)
diff = error_a - error_b  # vector de diferencias pareadas: UNA por tienda
n = diff.size
if n < 2:
    raise ValueError("Se necesitan al menos 2 tiendas para el bootstrap pareado")

# ------------------------- Bootstrap pareado (tiendas completas) ----------------


def bootstrap_pareado(diferencias: np.ndarray, semilla: int, n_remuestras: int):
    """Remuestrea tiendas completas mediante el vector de diferencias.

    Cada extracción elige un índice de tienda; la diferencia de esa tienda
    entra como unidad indivisible. NUNCA se remuestrean error_a y error_b por
    separado (eso rompería el emparejamiento dentro de cada tienda).
    """
    rng = np.random.default_rng(semilla)
    n_tiendas = diferencias.size
    indices = rng.integers(0, n_tiendas, size=(n_remuestras, n_tiendas))
    medias = diferencias[indices].mean(axis=1)
    return medias, indices


# Dos ejecuciones independientes con la MISMA semilla (verificación de reproducibilidad)
boot1, idx1 = bootstrap_pareado(diff, SEED, BOOTSTRAP_RESAMPLES)
boot2, idx2 = bootstrap_pareado(diff, SEED, BOOTSTRAP_RESAMPLES)

# Intervalo percentil del 95 % (análisis de sensibilidad aproximado)
alfa = 1.0 - CONF_LEVEL
p_lo = 100.0 * (alfa / 2.0)        # 2.5
p_hi = 100.0 * (1.0 - alfa / 2.0)  # 97.5
ci1 = np.percentile(boot1, [p_lo, p_hi])
ci2 = np.percentile(boot2, [p_lo, p_hi])

# Verificación de que el remuestreo es de tiendas completas: con los MISMOS
# índices, media(error_a) - media(error_b) debe coincidir con la media del
# vector de diferencias remuestreado (si se remuestrearan A y B por separado,
# esta igualdad no se mantendría tienda a tienda).
boot_desde_pares = error_a[idx1].mean(axis=1) - error_b[idx1].mean(axis=1)
max_dif_pares = float(np.max(np.abs(boot_desde_pares - boot1)))
pares_ok = bool(np.allclose(boot_desde_pares, boot1, rtol=1e-12, atol=1e-12))

# Control: con una semilla DISTINTA el remuestreo sí cambia (descarta falsa
# reproducibilidad por un resultado constante)
boot_ctrl, idx_ctrl = bootstrap_pareado(diff, SEED + 1, BOOTSTRAP_RESAMPLES)
ci_ctrl = np.percentile(boot_ctrl, [p_lo, p_hi])
indices_difieren = bool(not np.array_equal(idx1, idx_ctrl))

# Comprobación de reproducibilidad (misma semilla -> mismo intervalo)
mismo_intervalo = bool(np.array_equal(ci1, ci2))
medias_identicas = bool(np.array_equal(boot1, boot2))
indices_identicos = bool(np.array_equal(idx1, idx2))

# ------------------------- Coherencia con T2 (informativa) ----------------------
coherencia_t2 = None
try:
    if T2_RESULTS_PATH.exists():
        t2 = json.loads(T2_RESULTS_PATH.read_text(encoding="utf-8"))
        filas = t2.get("tabla_pareada", [])
        t2_ids = [str(f["store_id"]) for f in filas]
        t2_diff = np.array([float(f["difference"]) for f in filas], dtype=float)
        if t2_ids == store_ids and t2_diff.size == n:
            coherencia_t2 = {
                "comparado_con": str(T2_RESULTS_PATH),
                "ids_iguales": True,
                "diferencias_coinciden": bool(
                    np.allclose(t2_diff, diff, rtol=0.0, atol=1e-12)
                ),
            }
except Exception as exc:  # informativa, no bloqueante
    coherencia_t2 = {"comparado_con": str(T2_RESULTS_PATH), "error_lectura": str(exc)}

# --------------------------------- Salida CSV -----------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
favors = np.where(diff > 0, "B", np.where(diff < 0, "A", "TIE"))
tabla_out = pd.DataFrame(
    {
        "store_id": store_ids,
        "error_a": error_a,
        "error_b": error_b,
        "difference": diff,
        "favors": favors,
    }
)
tabla_out.to_csv(CSV_OUT, index=False)

# ---------------------------------- Figuras -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

colores = np.where(diff > 0, "#1f77b4", "#d62728")
axes[0].bar(store_ids, diff, color=colores, edgecolor="black", linewidth=0.4)
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].axhline(
    float(diff.mean()), color="green", linestyle="--", linewidth=1.3,
    label=f"media = {diff.mean():.4f}",
)
axes[0].set_title("Diferencia pareada por tienda (error_a - error_b)")
axes[0].set_xlabel("store_id")
axes[0].set_ylabel("difference")
axes[0].tick_params(axis="x", rotation=90)
axes[0].legend(loc="lower right", fontsize=9)

axes[1].hist(boot1, bins=60, color="#9ecae1", edgecolor="white", linewidth=0.3)
axes[1].axvspan(
    float(ci1[0]), float(ci1[1]), color="red", alpha=0.15,
    label=f"IC 95% percentil = [{ci1[0]:.4f}, {ci1[1]:.4f}]",
)
axes[1].axvline(float(ci1[0]), color="red", linewidth=1.6)
axes[1].axvline(float(ci1[1]), color="red", linewidth=1.6)
axes[1].axvline(
    float(diff.mean()), color="green", linestyle="--", linewidth=1.6,
    label=f"media observada = {diff.mean():.4f}",
)
axes[1].set_title(
    f"Bootstrap pareado: {BOOTSTRAP_RESAMPLES} remuestras de tiendas\n(semilla {SEED})"
)
axes[1].set_xlabel("media de difference en la remuestra")
axes[1].set_ylabel("frecuencia")
axes[1].legend(loc="upper left", fontsize=9)

fig.tight_layout()
fig.savefig(FIG_OUT, dpi=120)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(5.8, 5.4))
ax2.scatter(boot1, boot2, s=4, alpha=0.25, color="#1f77b4")
lims = [float(min(boot1.min(), boot2.min())), float(max(boot1.max(), boot2.max()))]
ax2.plot(lims, lims, color="black", linewidth=1.0, linestyle="--")
ax2.set_xlabel("medias bootstrap - ejecución 1 (semilla 20260829)")
ax2.set_ylabel("medias bootstrap - ejecución 2 (semilla 20260829)")
ax2.set_title(
    f"Reproducibilidad: misma semilla -> mismas {BOOTSTRAP_RESAMPLES} medias\n"
    f"IC95 idéntico en ambas ejecuciones: {mismo_intervalo}"
)
fig2.tight_layout()
fig2.savefig(FIG_REPRO_PATH, dpi=120)
plt.close(fig2)

# ------------------------------- resultados.json --------------------------------
resultados = {
    "subtarea": "T6_bootstrap_pareado",
    "seed": SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "config": {
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "nivel_confianza": CONF_LEVEL,
        "metodo": "bootstrap percentil (análisis de sensibilidad aproximado)",
        "generador": "np.random.default_rng",
        "unidad_remuestreo": (
            "tiendas completas: se remuestrea el vector de diferencias "
            "(difference = error_a - error_b, una por tienda); error_a y error_b "
            "NUNCA se remuestrean por separado"
        ),
        "n_tiendas": int(n),
        "interpolacion_percentil": "linear (por defecto de numpy)",
    },
    "datos": {
        "csv_origen": str(DATA_PATH),
        "ids_tiendas": store_ids,
        "difference": [float(x) for x in diff],
        "media_difference_observada": float(diff.mean()),
        "mediana_difference_observada": float(np.median(diff)),
        "desv_est_difference_ddof1": float(diff.std(ddof=1)),
        "error_estandar_analitico_media": float(diff.std(ddof=1) / np.sqrt(n)),
    },
    "bootstrap": {
        "n_remuestras": BOOTSTRAP_RESAMPLES,
        "media_bootstrap": float(boot1.mean()),
        "error_estandar_bootstrap": float(boot1.std(ddof=1)),
        "sesgo_bootstrap": float(boot1.mean() - diff.mean()),
        "ic_percentil_95": {
            "percentiles": [p_lo, p_hi],
            "limite_inferior": float(ci1[0]),
            "limite_superior": float(ci1[1]),
        },
        "ic95_limite_inferior": float(ci1[0]),
        "ic95_limite_superior": float(ci1[1]),
        "incluye_cero": bool(ci1[0] <= 0.0 <= ci1[1]),
        "proporcion_remuestras_media_mayor_que_cero": float(np.mean(boot1 > 0)),
        "verificacion_remuestreo_tiendas_completas": {
            "descripcion": (
                "Con los mismos índices de tiendas remuestreadas, "
                "media(error_a)-media(error_b) coincide con la media del vector de "
                "diferencias remuestreado; confirma que la unidad de remuestreo es "
                "la tienda completa (pareado preservado)."
            ),
            "coincide": pares_ok,
            "max_diferencia_absoluta": max_dif_pares,
        },
    },
    "reproducibilidad": {
        "semilla": SEED,
        "n_ejecuciones_misma_semilla": 2,
        "ejecucion_1": {
            "limite_inferior": float(ci1[0]),
            "limite_superior": float(ci1[1]),
        },
        "ejecucion_2": {
            "limite_inferior": float(ci2[0]),
            "limite_superior": float(ci2[1]),
        },
        "mismo_intervalo": mismo_intervalo,
        "medias_bootstrap_identicas_bit_a_bit": medias_identicas,
        "indices_remuestreo_identicos": indices_identicos,
        "verificado": bool(mismo_intervalo and medias_identicas and indices_identicos),
        "control_semilla_distinta": {
            "seed": SEED + 1,
            "ic95": [float(ci_ctrl[0]), float(ci_ctrl[1])],
            "indices_difieren_de_ejecucion_1": indices_difieren,
        },
    },
    "coherencia_con_T2": coherencia_t2,
    "notas": (
        "El intervalo percentil del 95 % se reporta como análisis de sensibilidad "
        "aproximado. Remuestrear tiendas completas (y no los errores de A y B por "
        "separado) preserva el emparejamiento y la correlación dentro de cada tienda."
    ),
    "salidas": {
        "csv_tabla_diferencias": str(CSV_OUT),
        "figura_diferencias_y_bootstrap": str(FIG_OUT),
        "figura_reproducibilidad": str(FIG_REPRO_PATH),
        "resultados_json": str(RESULTS_PATH),
    },
}

RESULTS_PATH.write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ---------------------------------- Resumen -------------------------------------
print("=" * 68)
print("T6 — Bootstrap pareado (remuestreo de tiendas completas)")
print("=" * 68)
print(f"Datos: {DATA_PATH} ({n} tiendas)")
print(f"Semilla: {SEED} | Remuestras: {BOOTSTRAP_RESAMPLES} | Método: percentil")
print(f"Media observada de difference: {diff.mean():.6f}")
print(f"IC percentil 95%: [{ci1[0]:.6f}, {ci1[1]:.6f}]  (incluye 0: {ci1[0] <= 0 <= ci1[1]})")
print(
    "Reproducibilidad: ejecución 2 IC = "
    f"[{ci2[0]:.6f}, {ci2[1]:.6f}] -> mismo intervalo: {mismo_intervalo}"
)
print(f"Verificación pareado (tiendas completas): {pares_ok} (max dif {max_dif_pares:.3e})")
print(f"Control semilla distinta ({SEED + 1}): índices difieren = {indices_difieren}")
print(f"Salidas: {CSV_OUT} | {FIG_OUT} | {FIG_REPRO_PATH} | {RESULTS_PATH}")
