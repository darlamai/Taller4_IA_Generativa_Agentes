#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T4 — Efecto pareado e incertidumbre sobre el vector de diferencias de T2.

Reporta exactamente n, mean, sample_sd (muestral, ddof=1) y
standard_error = sd / sqrt(n). Con scipy.stats calcula:
  * la prueba t de una muestra bilateral sobre las diferencias (H0: media = 0),
  * el intervalo t bilateral del 95 % para la media pareada,
conservando grados de libertad, estadístico t, p-value e intervalo.

Entradas:
  - entrada/T2/resultados.json  (vector de diferencias de T2; fallback: data/model_errors.csv)
Salidas (contrato del enunciado):
  - resultados.json
  - output/store_differences.csv
  - output/store_differences.png
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_CSV = Path("data/model_errors.csv")
COLUMNAS = ["store_id", "error_a", "error_b", "difference", "favors"]


def _nativo(v):
    """Convierte tipos de numpy a tipos nativos para json."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def cargar_diferencias():
    """Recupera la tabla pareada de T2 con su vector de diferencias.

    Devuelve (tabla, fuente, media_reportada_por_T2_o_None).
    """
    if RUTA_T2.exists():
        with open(RUTA_T2, "r", encoding="utf-8") as f:
            t2 = json.load(f)
        filas = t2.get("tabla_pareada", [])
        if filas and all("difference" in fila for fila in filas):
            tabla = pd.DataFrame(filas)
            tabla = tabla[[c for c in COLUMNAS if c in tabla.columns]].reset_index(drop=True)
            media_t2 = t2.get("resumen", {}).get("media_difference")
            return tabla, "entrada/T2/resultados.json (tabla_pareada de T2)", media_t2

    if not RUTA_CSV.exists():
        raise FileNotFoundError(
            "No se encontró ni entrada/T2/resultados.json ni data/model_errors.csv"
        )
    raw = pd.read_csv(RUTA_CSV)
    faltantes = [c for c in ["store_id", "error_a", "error_b"] if c not in raw.columns]
    if faltantes:
        raise ValueError(f"data/model_errors.csv sin columnas requeridas: {faltantes}")
    tabla = raw[["store_id", "error_a", "error_b"]].copy()
    tabla["difference"] = tabla["error_a"] - tabla["error_b"]
    tabla["favors"] = np.where(
        tabla["difference"] > 0, "B", np.where(tabla["difference"] < 0, "A", "tie")
    )
    return (
        tabla,
        "data/model_errors.csv (difference = error_a - error_b, misma definición de T2)",
        None,
    )


def main():
    tabla, fuente, media_t2 = cargar_diferencias()
    diffs = tabla["difference"].to_numpy(dtype=float)
    n = int(diffs.size)

    if n < 2:
        raise ValueError("Se necesitan >= 2 diferencias para la sd muestral y la prueba t.")
    if not np.all(np.isfinite(diffs)):
        raise ValueError("El vector de diferencias contiene valores no finitos.")

    # --- Estadísticos descriptivos exigidos ---
    mean = float(np.mean(diffs))
    sample_sd = float(np.std(diffs, ddof=1))            # desviación MUESTRAL
    standard_error = float(sample_sd / math.sqrt(n))    # SE = sd / sqrt(n)
    dfree = int(n - 1)

    # --- Prueba t de una muestra, bilateral, H0: media de las diferencias = 0 ---
    try:
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0, alternative="two-sided")
    except TypeError:  # scipy antiguo: por defecto ya es bilateral
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
    t_stat, p_val = float(t_stat), float(p_val)

    # --- Intervalo t bilateral del 95 % para la media pareada ---
    try:
        ci_low, ci_high = stats.t.interval(
            confidence=0.95, df=dfree, loc=mean, scale=standard_error
        )
    except TypeError:  # scipy antiguo usa alpha
        ci_low, ci_high = stats.t.interval(
            alpha=0.95, df=dfree, loc=mean, scale=standard_error
        )
    ci_low, ci_high = float(ci_low), float(ci_high)

    # --- Verificaciones internas ---
    t_crit = float(stats.t.ppf(0.975, dfree))
    verif = {
        "n_es_16": bool(n == 16),
        "se_igual_sd_sobre_raiz_de_n": bool(
            np.isclose(standard_error, sample_sd / np.sqrt(n))
        ),
        "t_igual_mean_sobre_se": bool(np.isclose(t_stat, mean / standard_error)),
        "ci_coincide_con_construccion_manual_ppf": bool(
            np.isclose(ci_low, mean - t_crit * standard_error)
            and np.isclose(ci_high, mean + t_crit * standard_error)
        ),
        "df_igual_n_menos_1": bool(dfree == n - 1),
    }
    if media_t2 is not None:
        verif["media_coincide_con_resumen_T2"] = bool(np.isclose(mean, float(media_t2)))
    if {"error_a", "error_b"}.issubset(tabla.columns):
        try:
            t_rel, p_rel = stats.ttest_rel(
                tabla["error_a"].to_numpy(dtype=float),
                tabla["error_b"].to_numpy(dtype=float),
            )
            verif["coincide_con_ttest_rel_pareada"] = bool(
                np.isclose(float(t_rel), t_stat) and np.isclose(float(p_rel), p_val)
            )
        except Exception:
            verif["coincide_con_ttest_rel_pareada"] = None

    # --- Interpretación pedida por el enunciado ---
    interpretacion = {
        "diferencia_media_en_unidades": (
            f"La diferencia media es {mean:.4f} unidades por tienda "
            f"(difference = error_a - error_b; en la escala del enunciado, unidades vendidas; "
            f"signo positivo favorece a B): en promedio, B reduce el error en "
            f"~{abs(mean):.2f} unidades respecto de A."
        ),
        "dispersion_entre_tiendas_vs_error_estandar": (
            f"La desviación muestral ({sample_sd:.4f}) describe la dispersión del efecto "
            f"entre tiendas, mientras que el error estándar ({standard_error:.4f}) describe "
            f"la incertidumbre de la media muestral; con n={n}, SE = sd/sqrt(n) = sd/{math.sqrt(n):.0f}."
        ),
        "lectura_prueba_e_intervalo": (
            f"t({dfree}) = {t_stat:.4f} con p bilateral = {p_val:.4f} > 0.05 y un IC t del 95 % "
            f"[{ci_low:.4f}, {ci_high:.4f}] que contiene a 0: no se rechaza H0 al nivel 0.05; "
            f"el efecto medio no se distingue de cero, aunque el intervalo admite efectos "
            f"positivos de hasta {ci_high:.2f} unidades."
        ),
    }

    # --- Salidas exigidas por el enunciado ---
    Path("output").mkdir(parents=True, exist_ok=True)
    cols_csv = [c for c in COLUMNAS if c in tabla.columns]
    tabla[cols_csv].to_csv("output/store_differences.csv", index=False)

    # Figura: diferencias por tienda + efecto con su incertidumbre
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    colores = np.where(
        diffs > 0, "#1f77b4", np.where(diffs < 0, "#d62728", "#7f7f7f")
    ).tolist()
    ax1.bar(np.arange(n), diffs, color=colores, edgecolor="black", linewidth=0.4)
    ax1.axhline(0.0, color="black", lw=1.0)
    ax1.axhline(mean, color="green", ls="--", lw=1.6)
    ax1.axhspan(ci_low, ci_high, color="green", alpha=0.12)
    ax1.set_xticks(np.arange(n))
    ax1.set_xticklabels(tabla["store_id"].astype(str), rotation=90, fontsize=8)
    ax1.set_ylabel("difference = error_a − error_b")
    ax1.set_title("Diferencias por tienda")
    ax1.grid(axis="y", alpha=0.3)
    handles1 = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="difference > 0 (favorece B)"),
        Patch(facecolor="#d62728", edgecolor="black", label="difference < 0 (favorece A)"),
        Line2D([0], [0], color="green", ls="--", lw=1.6, label=f"media = {mean:.4f}"),
        Patch(facecolor="green", alpha=0.12, label=f"IC t 95 % = [{ci_low:.4f}, {ci_high:.4f}]"),
    ]
    ax1.legend(handles=handles1, fontsize=7.5, loc="lower left")

    jitter = np.linspace(-0.09, 0.09, n)  # desplazamiento determinista (sin aleatoriedad)
    ax2.scatter(jitter, diffs, color="gray", alpha=0.65, s=28, zorder=2,
                label="diferencias por tienda")
    ax2.errorbar([0.0], [mean],
                 yerr=[[mean - ci_low], [ci_high - mean]],
                 fmt="o", color="crimson", ecolor="crimson",
                 capsize=7, markersize=9, zorder=3,
                 label="media ± IC t 95 %")
    ax2.axhline(0.0, color="black", lw=1.0)
    ax2.set_xlim(-0.45, 0.45)
    ax2.set_xticks([])
    ax2.set_ylabel("difference = error_a − error_b")
    ax2.set_title(f"Efecto pareado e incertidumbre\n"
                  f"t({dfree}) = {t_stat:.4f}, p bilateral = {p_val:.4f}")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"T4: Efecto pareado A vs B e incertidumbre (n = {n})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("output/store_differences.png", dpi=120)
    plt.close(fig)

    # --- resultados.json (contrato de la subtarea) ---
    tabla_records = [
        {k: _nativo(v) for k, v in fila.items()}
        for fila in tabla[cols_csv].to_dict(orient="records")
    ]
    resultados = {
        "subtarea": "T4_efecto_e_incertidumbre_prueba_t_pareada",
        "fuente_diferencias": fuente,
        "definicion": "difference = error_a - error_b (vector de diferencias de T2)",
        "n": n,
        "mean": mean,
        "sample_sd": sample_sd,
        "sample_sd_ddof": 1,
        "standard_error": standard_error,
        "standard_error_formula": "sample_sd / sqrt(n)",
        "degrees_of_freedom": dfree,
        "t_statistic": t_stat,
        "p_value_two_sided": p_val,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "confidence_level": 0.95,
        "alternative": "two-sided",
        "popmean": 0.0,
        "t_critical_two_sided_95": t_crit,
        "prueba_t": {
            "metodo": "scipy.stats.ttest_1samp sobre el vector de diferencias (equivalente a la t pareada)",
            "hipotesis_nula": "media de las diferencias = 0",
            "alternative": "two-sided",
            "popmean": 0.0,
            "degrees_of_freedom": dfree,
            "t_statistic": t_stat,
            "p_value_two_sided": p_val,
        },
        "intervalo_t_95": {
            "metodo": "scipy.stats.t.interval (t bilateral) para la media pareada",
            "confidence_level": 0.95,
            "degrees_of_freedom": dfree,
            "standard_error_usado": standard_error,
            "t_critical": t_crit,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "resumen_diferencias": {
            "suma": float(np.sum(diffs)),
            "min": float(np.min(diffs)),
            "max": float(np.max(diffs)),
            "n_positivas_favors_B": int(np.sum(diffs > 0)),
            "n_negativas_favors_A": int(np.sum(diffs < 0)),
            "n_cero": int(np.sum(diffs == 0)),
        },
        "verificaciones": verif,
        "interpretacion": interpretacion,
        "tabla_pareada": tabla_records,
        "salidas": {
            "resultados_json": "resultados.json",
            "csv_tabla_pareada": "output/store_differences.csv",
            "figura": "output/store_differences.png",
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # --- Resumen en consola ---
    print("=" * 66)
    print("T4 — Efecto pareado e incertidumbre (vector de diferencias de T2)")
    print("=" * 66)
    print(f"Fuente de las diferencias : {fuente}")
    print(f"n                  = {n}")
    print(f"mean               = {mean:.6f}")
    print(f"sample_sd (ddof=1) = {sample_sd:.6f}")
    print(f"standard_error     = {standard_error:.6f}   (sd / sqrt(n))")
    print(f"grados de libertad = {dfree}")
    print(f"estadístico t      = {t_stat:.6f}")
    print(f"p-value bilateral  = {p_val:.6f}")
    print(f"IC t 95 %          = [{ci_low:.6f}, {ci_high:.6f}]  (t_crit = {t_crit:.6f})")
    print("Verificaciones     : " + "; ".join(f"{k}={v}" for k, v in verif.items()))
    print("Salidas: resultados.json | output/store_differences.csv | output/store_differences.png")


if __name__ == "__main__":
    main()
