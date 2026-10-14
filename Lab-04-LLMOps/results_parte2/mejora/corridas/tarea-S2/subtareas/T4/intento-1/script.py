#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T4 — Efecto e incertidumbre sobre el vector de diferencias pareadas.

Sobre las diferencias error_a − error_b (emparejadas por store_id) reporta:
n, mean, sample_sd (ddof=1) y standard_error = sd/√n (verificado por una ruta
de cálculo independiente). Con scipy.stats calcula la prueba t de una muestra
bilateral (H0: media de las diferencias = 0) y el intervalo t bilateral del
95 % para la media pareada, conservando grados de libertad, estadístico,
p-value e intervalo. Genera output/store_differences.csv,
output/store_differences.png y resultados.json.
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
from matplotlib.patches import Patch

RUTA_DATOS = Path("data/model_errors.csv")
RUTA_T2 = Path("entrada/T2/results.json")
RUTA_CSV_SALIDA = Path("output/store_differences.csv")
RUTA_PNG_SALIDA = Path("output/store_differences.png")
RUTA_RESULTADOS = Path("resultados.json")

NIVEL_CONFIANZA = 0.95
ALFA = 1.0 - NIVEL_CONFIANZA
MU0 = 0.0


def cargar_tabla_pareada():
    """Carga store_id, error_a, error_b desde el CSV de datos (o respaldo T2)."""
    if RUTA_DATOS.exists():
        df = pd.read_csv(RUTA_DATOS)
        fuente = str(RUTA_DATOS)
    elif RUTA_T2.exists():
        with open(RUTA_T2, "r", encoding="utf-8") as f:
            previo = json.load(f)
        df = pd.DataFrame(previo["tabla_pareada"])
        fuente = str(RUTA_T2)
    else:
        raise FileNotFoundError(
            "No se encontró data/model_errors.csv ni entrada/T2/results.json"
        )

    columnas = ["store_id", "error_a", "error_b"]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas {faltantes} en {fuente}")

    df = df[columnas].copy()
    if df["store_id"].isna().any():
        raise ValueError("store_id nulo: el emparejamiento 1:1 no está garantizado")
    df["store_id"] = df["store_id"].astype(str)
    if df["store_id"].duplicated().any():
        raise ValueError("store_id duplicado: el emparejamiento 1:1 no está garantizado")
    for c in ["error_a", "error_b"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[["error_a", "error_b"]].to_numpy(dtype=float)).all():
        raise ValueError("Los errores deben ser finitos")

    df = df.sort_values("store_id").reset_index(drop=True)
    return df, fuente


def main():
    df, fuente = cargar_tabla_pareada()

    # Vector de diferencias pareadas (difference = error_a − error_b)
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(df["difference"] > 0, "B",
                            np.where(df["difference"] < 0, "A", "tie"))
    diffs = df["difference"].to_numpy(dtype=float)

    # ---- Cifras del efecto ----
    n = int(diffs.size)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 diferencias para la sd muestral")
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))          # desviación MUESTRAL (ddof=1)
    se = float(sd / math.sqrt(n))              # standard_error = sd / sqrt(n)

    # Verificación independiente de SE = sd/√n
    ssd = float(np.sum((diffs - mean) ** 2))
    sd_verif = math.sqrt(ssd / (n - 1))
    se_verif = float(sd_verif / math.sqrt(n))
    se_ok = bool(math.isclose(se, se_verif, rel_tol=1e-9, abs_tol=1e-15))

    # ---- Prueba t de una muestra, bilateral, sobre las diferencias ----
    t_stat, p_val = stats.ttest_1samp(diffs, MU0)  # alternativa bilateral por defecto
    t_stat, p_val = float(t_stat), float(p_val)
    gl = n - 1
    t_verif = float(mean / se)
    t_ok = bool(math.isclose(t_stat, t_verif, rel_tol=1e-9))
    p_verif = float(2.0 * stats.t.sf(abs(t_stat), gl))

    # ---- Intervalo t bilateral del 95 % para la media pareada ----
    t_crit = float(stats.t.ppf(1.0 - ALFA / 2.0, gl))
    margen = float(t_crit * se)
    ci_lo = float(mean - margen)
    ci_hi = float(mean + margen)
    ci_lo_sp, ci_hi_sp = stats.t.interval(NIVEL_CONFIANZA, df=gl, loc=mean, scale=se)
    ci_ok = bool(math.isclose(ci_lo, float(ci_lo_sp), rel_tol=1e-9)
                 and math.isclose(ci_hi, float(ci_hi_sp), rel_tol=1e-9))
    incluye_cero = bool(ci_lo <= 0.0 <= ci_hi)
    decision = "no se rechaza H0" if p_val >= ALFA else "se rechaza H0"

    # ---- Tabla y conteos ----
    tabla = [
        {"store_id": str(r.store_id), "error_a": float(r.error_a),
         "error_b": float(r.error_b), "difference": float(r.difference),
         "favors": str(r.favors)}
        for r in df.itertuples(index=False)
    ]
    conteos = {
        "favorece_A": int((diffs < 0).sum()),
        "favorece_B": int((diffs > 0).sum()),
        "empates": int((diffs == 0).sum()),
    }

    # ---- Interpretación (signo y magnitud en unidades vendidas) ----
    signo_txt = "positivo" if mean > 0 else ("negativo" if mean < 0 else "cero")
    direccion_txt = (
        "a favor de la dirección positiva de la diferencia (error_a − error_b; "
        "en la convención de la T2, diferencia positiva favorece a B)"
        if mean > 0
        else "a favor de la dirección negativa de la diferencia (error_a − error_b; "
             "en la convención de la T2, diferencia negativa favorece a A)"
    )
    interpretacion = (
        f"Efecto: con n={n} tiendas, la diferencia media es {mean:+.6f} unidades vendidas "
        f"(signo {signo_txt}, magnitud {abs(mean):.6f} unidades por tienda en promedio), "
        f"{direccion_txt}. Incertidumbre: la desviación muestral ({sd:.6f}) mide la dispersión "
        f"de las diferencias entre tiendas, mientras que el error estándar ({se:.6f} = sd/√n) "
        f"mide la incertidumbre de la media; aquí sd ≈ {sd / se:.2f}·SE. Valores compatibles: "
        f"el IC t del 95 % va de {ci_lo:.6f} a {ci_hi:.6f} unidades "
        f"({'incluye' if incluye_cero else 'excluye'} el 0; p bilateral = {p_val:.6f}), de modo "
        f"que bajo el modelo t efectos medios en ese rango son compatibles con los datos; el "
        f"intervalo es frecuentista y no debe interpretarse como una probabilidad posterior "
        f"sobre el parámetro."
    )

    # ---- Figura: output/store_differences.png ----
    RUTA_PNG_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colores = np.where(diffs >= 0, "#2b8a3e", "#c92a2a")
    ax.bar(df["store_id"], diffs, color=colores, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(mean, color="#1c7ed6", linewidth=1.8,
               label=f"Media = {mean:.4f}")
    ax.axhspan(ci_lo, ci_hi, color="#1c7ed6", alpha=0.12,
               label=f"IC t 95 % = [{ci_lo:.4f}, {ci_hi:.4f}] (gl={gl})")
    for x, d in zip(df["store_id"], diffs):
        ax.annotate(f"{d:+.1f}", (x, d), ha="center", fontsize=7,
                    va="bottom" if d >= 0 else "top")
    ax.margins(y=0.22)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Tienda (store_id)")
    ax.set_ylabel("Diferencia pareada: error_a − error_b (unidades vendidas)")
    ax.set_title(f"T4 · Diferencias pareadas por tienda (n={n}) · "
                 f"t={t_stat:.4f}, p={p_val:.4f} (bilateral), gl={gl}")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="#2b8a3e", edgecolor="black", label="Diferencia > 0 (favors B)"),
                Patch(facecolor="#c92a2a", edgecolor="black", label="Diferencia < 0 (favors A)")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(RUTA_PNG_SALIDA), dpi=120)
    plt.close(fig)

    # ---- CSV: output/store_differences.csv ----
    RUTA_CSV_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_CSV_SALIDA, index=False, encoding="utf-8")

    # ---- resultados.json (contrato de la subtarea) ----
    resultados = {
        "subtarea": "T4",
        "descripcion": ("Efecto e incertidumbre: n, media, sd muestral y SE de las diferencias; "
                        "prueba t de una muestra bilateral e IC t del 95 % para la media pareada"),
        "fuente_datos": fuente,
        "definicion_diferencia": "difference = error_a − error_b, emparejada por store_id",
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "verificacion_se": {
            "formula": "standard_error = sample_sd / sqrt(n)",
            "se_recalculado_ruta_independiente": se_verif,
            "coincide": se_ok,
        },
        "grados_libertad": gl,
        "estadistico_t": t_stat,
        "p_value_bilateral": p_val,
        "ic95_inferior": ci_lo,
        "ic95_superior": ci_hi,
        "prueba_t_una_muestra_bilateral": {
            "hipotesis_nula": "media de las diferencias = 0",
            "alternativa": "two-sided (bilateral, por defecto de scipy.stats.ttest_1samp)",
            "mu0": MU0,
            "grados_libertad": gl,
            "estadistico": t_stat,
            "p_value": p_val,
            "p_value_verificado_2_sf": p_verif,
            "estadistico_verificado_mean_sobre_se": t_verif,
            "coincide_estadistico": t_ok,
            "alfa": ALFA,
            "decision_5pct": decision,
        },
        "intervalo_t_95": {
            "nivel_confianza": NIVEL_CONFIANZA,
            "metodo": ("media ± t_{0.975, gl} · SE (intervalo t bilateral para la media "
                       "pareada), verificado con scipy.stats.t.interval"),
            "grados_libertad": gl,
            "t_critico": t_crit,
            "margen_error": margen,
            "limite_inferior": ci_lo,
            "limite_superior": ci_hi,
            "verificado_con_scipy_interval": ci_ok,
            "incluye_cero": incluye_cero,
            "nota": ("El intervalo describe los valores del efecto medio compatibles con los "
                     "datos bajo el modelo t; no es una probabilidad posterior sobre el parámetro."),
        },
        "signo_y_magnitud_media": {
            "signo": signo_txt,
            "magnitud_unidades_vendidas": abs(mean),
            "media_unidades_vendidas": mean,
        },
        "dispersion_vs_error_estandar": {
            "sample_sd_descripcion": "dispersión de las diferencias entre tiendas",
            "standard_error_descripcion": "incertidumbre de la media de las diferencias",
            "razon_sd_se": float(sd / se),
            "nota": "sd ≈ sqrt(n) · SE; la dispersión entre tiendas no debe confundirse con el SE de la media.",
        },
        "tabla_pareada": tabla,
        "vector_diferencias": [float(x) for x in diffs],
        "conteos": conteos,
        "interpretacion": interpretacion,
        "archivos_generados": {
            "csv": str(RUTA_CSV_SALIDA),
            "figura": str(RUTA_PNG_SALIDA),
            "resultados": str(RUTA_RESULTADOS),
        },
    }
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ---- Resumen ----
    print("=== T4 · Efecto e incertidumbre (diferencias pareadas) ===")
    print(f"Fuente de datos: {fuente}")
    print(f"n = {n}")
    print(f"mean = {mean:.6f} unidades vendidas (signo {signo_txt})")
    print(f"sample_sd (ddof=1) = {sd:.6f}")
    print(f"standard_error = sd/sqrt(n) = {se:.6f}  (verificado: {se_ok})")
    print(f"gl = {gl} · t = {t_stat:.6f} · p (bilateral) = {p_val:.6f} · {decision} al 5 %")
    print(f"IC t 95 % = [{ci_lo:.6f}, {ci_hi:.6f}] "
          f"(t_crit={t_crit:.6f}, margen={margen:.6f}, verificado: {ci_ok})")
    print(f"El intervalo incluye 0: {incluye_cero}")
    print(f"Archivos generados: {RUTA_CSV_SALIDA}, {RUTA_PNG_SALIDA}, {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
