# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado del vector de diferencias por tienda.

- Remuestrea TIENDAS COMPLETAS: un solo sorteo de índices con reemplazo se aplica
  al vector de diferencias d = error_a - error_b (los pares nunca se separan;
  error_a y error_b jamás se remuestrean por separado).
- Reporta el intervalo percentil del 95 % con SEED = 20260829 y
  BOOTSTRAP_RESAMPLES = 10_000.
- Verifica que dos ejecuciones con la misma semilla produzcan el mismo intervalo
  (el generador se recrea desde cero en cada corrida).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ parámetros
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
PCT_LO, PCT_HI = 2.5, 97.5  # percentiles del intervalo del 95 %

RUTA_DATOS = Path("data/model_errors.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_CSV = Path("output/store_differences.csv")
RUTA_FIG = Path("output/store_differences.png")
RUTA_RESULTADOS = Path("resultados.json")

# ------------------------------------------------------- carga y validación
if not RUTA_DATOS.exists():
    raise FileNotFoundError(f"No se encontró el archivo de datos: {RUTA_DATOS}")

df = pd.read_csv(RUTA_DATOS)
columnas_esperadas = ["store_id", "error_a", "error_b"]
if list(df.columns) != columnas_esperadas:
    raise ValueError(
        f"Columnas inesperadas {list(df.columns)}; se exige exactamente {columnas_esperadas}"
    )
if (
    df["store_id"].isna().any()
    or df["store_id"].astype(str).str.strip().eq("").any()
    or df["store_id"].duplicated().any()
):
    raise ValueError(
        "store_id inválido: el identificador de tienda contiene nulos, vacíos o duplicados"
    )
for col in ("error_a", "error_b"):
    valores = pd.to_numeric(df[col], errors="coerce")
    if valores.isna().any():
        raise ValueError(f"{col}: contiene valores no numéricos o faltantes")
    if not np.isfinite(valores.to_numpy(dtype=float)).all():
        raise ValueError(f"{col}: contiene valores no finitos")
    if (valores < 0).any():
        raise ValueError(f"{col}: contiene errores negativos")
    df[col] = valores

store_ids = df["store_id"].astype(str).tolist()
error_a = df["error_a"].to_numpy(dtype=float)
error_b = df["error_b"].to_numpy(dtype=float)
d = error_a - error_b  # vector de diferencias pareadas: UNA diferencia por tienda
n = d.size
favors = np.where(d > 0, "B", np.where(d < 0, "A", "empate"))

# --------------------------------------------------------- bootstrap pareado
def bootstrap_pareado(diferencias: np.ndarray, semilla: int, n_remuestras: int):
    """Remuestrea tiendas completas.

    Un único sorteo de índices de tienda con reemplazo (rng.integers) se aplica
    al vector de diferencias: cada unidad remuestreada es la tienda completa
    (su par error_a/error_b viaja junto). Nunca se remuestrean error_a y
    error_b por separado.
    """
    rng = np.random.default_rng(semilla)  # generador recreado desde cero
    n_tiendas = diferencias.size
    indices = rng.integers(0, n_tiendas, size=(n_remuestras, n_tiendas))
    medias = diferencias[indices].mean(axis=1)
    lo, hi = np.percentile(medias, [PCT_LO, PCT_HI])
    return float(lo), float(hi), medias


# Corrida principal y corrida de verificación con la MISMA semilla
lo1, hi1, medias1 = bootstrap_pareado(d, SEED, BOOTSTRAP_RESAMPLES)
lo2, hi2, medias2 = bootstrap_pareado(d, SEED, BOOTSTRAP_RESAMPLES)

limites_identicos = bool(lo1 == lo2 and hi1 == hi2)
remuestras_identicas = bool(np.array_equal(medias1, medias2))
intervalos_identicos = bool(limites_identicos and remuestras_identicas)

# Diagnóstico informativo: una semilla distinta sí modifica el intervalo
lo_alt, hi_alt, _ = bootstrap_pareado(d, SEED + 1, BOOTSTRAP_RESAMPLES)
semilla_distinta_cambia = bool(not (lo_alt == lo1 and hi_alt == hi1))

# ------------------------------------------------- estadísticos observados
media_obs = float(d.mean())
mediana_obs = float(np.median(d))
sd_obs = float(d.std(ddof=1))
se_obs = float(sd_obs / np.sqrt(n))
media_boot = float(medias1.mean())
se_boot = float(medias1.std(ddof=1))
sesgo_boot = float(media_boot - media_obs)
contiene_cero = bool(lo1 <= 0.0 <= hi1)
n_positivas = int((medias1 > 0).sum())

# ------------------------------------- verificación cruzada con T2 (opcional)
verificacion_t2 = {
    "archivo_consultado": str(RUTA_T2),
    "disponible": False,
    "coincide": None,
    "nota": "Compara el vector de diferencias recalculado con la tabla pareada de T2.",
}
if RUTA_T2.exists():
    try:
        with open(RUTA_T2, "r", encoding="utf-8") as f:
            t2 = json.load(f)
        t2_ids = [fila["store_id"] for fila in t2["tabla_pareada"]]
        t2_dif = np.array([fila["difference"] for fila in t2["tabla_pareada"]], dtype=float)
        verificacion_t2["disponible"] = True
        verificacion_t2["n_pares_t2"] = int(len(t2_ids))
        verificacion_t2["coincide"] = bool(
            t2_ids == store_ids and np.allclose(t2_dif, d, rtol=0.0, atol=1e-9)
        )
    except Exception as exc:  # lectura tolerante: no es parte del contrato
        verificacion_t2["error_lectura"] = str(exc)

# ------------------------------------------------------- CSV exigido (T6/T2)
RUTA_CSV.parent.mkdir(parents=True, exist_ok=True)
tabla = pd.DataFrame(
    {
        "store_id": store_ids,
        "error_a": error_a,
        "error_b": error_b,
        "difference": d,
        "favors": favors,
    }
)
tabla.to_csv(RUTA_CSV, index=False)

# ------------------------------------------------------ figura exigida (PNG)
RUTA_FIG.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
colores = ["#1f77b4" if v > 0 else "#d62728" for v in d]
ax.bar(np.arange(n), d, color=colores, edgecolor="black", linewidth=0.4)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.axhline(media_obs, color="green", linestyle="--", linewidth=1.2,
           label=f"media = {media_obs:.4f}")
ax.set_xticks(np.arange(n))
ax.set_xticklabels(store_ids, rotation=90, fontsize=8)
ax.set_ylabel("error_a - error_b")
ax.set_title("Diferencias pareadas por tienda")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
ax.hist(medias1, bins=50, color="#9ecae1", edgecolor="white")
ax.axvline(lo1, color="red", linestyle="--", linewidth=1.4,
           label=f"IC95% percentil = [{lo1:.4f}, {hi1:.4f}]")
ax.axvline(hi1, color="red", linestyle="--", linewidth=1.4)
ax.axvline(media_obs, color="green", linewidth=1.4,
           label=f"media observada = {media_obs:.4f}")
ax.set_xlabel("Media de diferencias en remuestras bootstrap")
ax.set_ylabel("Frecuencia")
ax.set_title(f"Bootstrap pareado (B={BOOTSTRAP_RESAMPLES}, semilla={SEED})")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(str(RUTA_FIG), dpi=120)
plt.close(fig)

# ------------------------------------------------------------- resultados.json
diferencias_por_tienda = [
    {
        "store_id": store_ids[i],
        "error_a": float(error_a[i]),
        "error_b": float(error_b[i]),
        "difference": float(d[i]),
        "favors": str(favors[i]),
    }
    for i in range(n)
]

interpretacion = (
    f"IC95% percentil de la media de diferencias (error_a - error_b): "
    f"[{lo1:.4f}, {hi1:.4f}], con {BOOTSTRAP_RESAMPLES} remuestras de tiendas "
    f"completas y semilla {SEED}. "
    + (
        "El intervalo contiene 0: como análisis de sensibilidad aproximado, la "
        "diferencia media entre modelos no es robusta al nivel del 95 %."
        if contiene_cero
        else "El intervalo no contiene 0: como análisis de sensibilidad aproximado, "
             "la diferencia media entre modelos es robusta al nivel del 95 %."
    )
)

resultado = {
    "subtarea": "T6",
    "descripcion": (
        "Bootstrap pareado: remuestreo de tiendas completas del vector de "
        "diferencias (error_a - error_b) e intervalo percentil del 95 %"
    ),
    "semilla": SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "n_tiendas": int(n),
    "generador": f"np.random.default_rng({SEED})",
    "unidad_de_remuestreo": (
        "tienda completa: un solo sorteo de índices con reemplazo se aplica al "
        "vector de diferencias; error_a y error_b nunca se remuestrean por separado"
    ),
    "metodo_bootstrap": (
        "Para cada remuestra b se muestrean n índices de tienda con reemplazo "
        "(rng.integers) y se calcula media_b = mean(d[índices]); el IC percentil "
        "del 95 % son los percentiles 2.5 y 97.5 de las 10000 medias"
    ),
    "diferencias_por_tienda": diferencias_por_tienda,
    "estadistico_observado": {
        "media_diferencia": media_obs,
        "mediana_diferencia": mediana_obs,
        "desviacion_estandar_muestral": sd_obs,
        "error_estandar_media": se_obs,
    },
    "bootstrap": {
        "metodo": "percentil",
        "nivel": 0.95,
        "percentiles_usados": [PCT_LO, PCT_HI],
        "ic95_inferior": lo1,
        "ic95_superior": hi1,
        "ic95_percentil": [lo1, hi1],
        "media_bootstrap": media_boot,
        "error_estandar_bootstrap": se_boot,
        "sesgo_bootstrap": sesgo_boot,
        "ic_contiene_cero": contiene_cero,
        "remuestras_con_media_positiva": n_positivas,
        "proporcion_remuestras_con_media_positiva": float(n_positivas / BOOTSTRAP_RESAMPLES),
    },
    "verificacion_reproducibilidad": {
        "protocolo": (
            "cada corrida recrea el generador desde cero con "
            f"np.random.default_rng({SEED}) y repite las {BOOTSTRAP_RESAMPLES} remuestras"
        ),
        "corrida_1_ic95": [lo1, hi1],
        "corrida_2_ic95": [lo2, hi2],
        "limites_identicos": limites_identicos,
        "remuestras_identicas": remuestras_identicas,
        "intervalos_identicos": intervalos_identicos,
        "diagnostico_semilla_distinta": {
            "solo_informativo": True,
            "semilla_alternativa": SEED + 1,
            "ic95": [lo_alt, hi_alt],
            "difiere_de_la_semilla_original": semilla_distinta_cambia,
        },
    },
    "verificacion_con_T2": verificacion_t2,
    "interpretacion": interpretacion,
    "explicaciones": {
        "por_que_se_remuestrean_tiendas_completas": (
            "El diseño es pareado: la unidad muestral es la tienda y el estadístico "
            "de interés es la media de las diferencias intra-tienda (error_a - "
            "error_b). Remuestrear tiendas completas (un solo sorteo de índices con "
            "reemplazo aplicado al vector de diferencias) mantiene unido cada par "
            "(error_a, error_b) y preserva la correlación intra-tienda. Si se "
            "remuestrearan error_a y error_b por separado se romperían los pares: "
            "errores de modelos y tiendas distintos se combinarían entre sí, se "
            "estimaría la diferencia entre medias de muestras independientes en "
            "lugar de la media de diferencias pareadas, y la variabilidad quedaría "
            "mal estimada (al ignorar la correlación positiva intra-par, en general "
            "inflada)."
        ),
        "que_no_puede_corregir_el_bootstrap": (
            "El bootstrap remuestrea los datos observados y hereda sus limitaciones: "
            "no corrige una muestra no representativa o no aleatoria (p. ej., si las "
            "16 tiendas no son una muestra aleatoria de la población de interés), ni "
            "sesgos de medición o de emparejamiento, ni dependencias entre tiendas "
            "(clústeres, correlación espacial o temporal) que la unidad de "
            "remuestreo no refleje, ni la ausencia de valores de cola no observados "
            "en muestras pequeñas. Tampoco repara una especificación errónea del "
            "estadístico: solo aproxima su distribución muestral condicionando a la "
            "muestra disponible."
        ),
    },
    "archivos_generados": {
        "resultados": str(RUTA_RESULTADOS),
        "tabla_csv": str(RUTA_CSV),
        "figura_png": str(RUTA_FIG),
    },
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ resumen
print("=" * 72)
print("T6 — Bootstrap pareado (remuestreo de tiendas completas)")
print(f"Semilla: {SEED} | Remuestras: {BOOTSTRAP_RESAMPLES} | Tiendas: {n}")
print(f"Media observada de diferencias: {media_obs:.6f}")
print(f"IC95% percentil: [{lo1:.6f}, {hi1:.6f}] | contiene 0: {contiene_cero}")
print(f"Reproducibilidad (misma semilla): intervalos idénticos = {intervalos_identicos}")
print(f"  corrida 1: [{lo1:.6f}, {hi1:.6f}]")
print(f"  corrida 2: [{lo2:.6f}, {hi2:.6f}]")
print(f"Diagnóstico semilla {SEED + 1}: [{lo_alt:.6f}, {hi_alt:.6f}] (cambia: {semilla_distinta_cambia})")
print(f"Archivos escritos: {RUTA_RESULTADOS}, {RUTA_CSV}, {RUTA_FIG}")
print("=" * 72)
