# T4: Efecto e incertidumbre de las diferencias pareadas
# - Calcula n, media, sd muestral (ddof=1) y SE = sd/sqrt(n) del vector de diferencias.
# - Prueba t de una muestra bilateral (scipy.stats) y IC t del 95 % para la media pareada.
# - Salidas: resultados.json, output/store_differences.csv, output/store_differences.png

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Cargar y validar los datos pareados
    # ------------------------------------------------------------------
    csv_path = Path("data/model_errors.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {csv_path}")

    df = pd.read_csv(csv_path)
    columnas_exigidas = ["store_id", "error_a", "error_b"]
    faltantes = [c for c in columnas_exigidas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas exigidas en {csv_path}: {faltantes}")

    if df["store_id"].isna().any() or df["store_id"].duplicated().any():
        raise ValueError("store_id debe ser único y no nulo para el emparejamiento pareado")

    for col in ("error_a", "error_b"):
        valores = pd.to_numeric(df[col], errors="coerce")
        if valores.isna().any():
            raise ValueError(f"La columna {col} contiene valores no numéricos")
        arr = valores.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"La columna {col} contiene valores no finitos")
        if (arr < 0).any():
            raise ValueError(f"La columna {col} contiene errores negativos")
        df[col] = arr

    # ------------------------------------------------------------------
    # 2) Vector de diferencias pareadas (mismo criterio que T2)
    # ------------------------------------------------------------------
    df = df.sort_values("store_id", kind="mergesort").reset_index(drop=True)
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(df["difference"] > 0, "B",
                            np.where(df["difference"] < 0, "A", "tie"))
    d = df["difference"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 3) Efecto e incertidumbre: n, media, sd muestral, SE
    # ------------------------------------------------------------------
    n = int(d.size)
    mean = float(np.mean(d))
    sample_sd = float(np.std(d, ddof=1))            # desviación MUESTRAL (ddof=1)
    standard_error = float(sample_sd / np.sqrt(n))  # SE = sd / sqrt(n)
    sem_scipy = float(stats.sem(d, ddof=1))         # verificación cruzada
    if abs(sem_scipy - standard_error) > 1e-12:
        raise AssertionError("SE manual y scipy.stats.sem no coinciden")

    # ------------------------------------------------------------------
    # 4) Prueba t de una muestra (bilateral) sobre las diferencias
    # ------------------------------------------------------------------
    try:
        res = stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    except TypeError:  # scipy antiguo: el valor por defecto ya es bilateral
        res = stats.ttest_1samp(d, popmean=0.0)
    t_stat = float(res.statistic)
    p_value = float(res.pvalue)
    dfree = int(n - 1)
    res_df = getattr(res, "df", None)
    if res_df is not None:
        dfree = int(res_df)
    p_check = float(2.0 * stats.t.sf(abs(t_stat), dfree))

    # ------------------------------------------------------------------
    # 5) IC t bilateral del 95 % para la media pareada
    # ------------------------------------------------------------------
    conf = 0.95
    try:
        ci_low, ci_high = stats.t.interval(confidence=conf, df=dfree,
                                           loc=mean, scale=standard_error)
    except TypeError:
        try:
            ci_low, ci_high = stats.t.interval(alpha=conf, df=dfree,
                                               loc=mean, scale=standard_error)
        except TypeError:
            tc = float(stats.t.ppf(0.5 + conf / 2.0, dfree))
            ci_low, ci_high = mean - tc * standard_error, mean + tc * standard_error
    ci_low, ci_high = float(ci_low), float(ci_high)
    t_crit = float(stats.t.ppf(0.5 + conf / 2.0, dfree))

    # ------------------------------------------------------------------
    # 6) Verificación opcional contra los resultados de T2
    # ------------------------------------------------------------------
    t2_check = {"disponible": False}
    t2_path = Path("entrada/T2/resultados.json")
    if t2_path.exists():
        try:
            with open(t2_path, "r", encoding="utf-8") as fh:
                t2 = json.load(fh)
            tabla_t2 = t2.get("tabla_pareada", [])
            if tabla_t2:
                d_t2 = np.array([float(r["difference"]) for r in tabla_t2], dtype=float)
                t2_check = {
                    "disponible": True,
                    "n_T2": int(t2.get("n_pares_tabla_pareada", d_t2.size)),
                    "media_diferencias_T2": float(np.mean(d_t2)),
                    "n_coincide": bool(int(d_t2.size) == n),
                    "diferencias_coinciden": bool(np.allclose(np.sort(d_t2), np.sort(d), atol=1e-9)),
                }
        except Exception as exc:  # la verificación es opcional, no bloquea
            t2_check = {"disponible": False, "nota": f"No se pudo contrastar con T2: {exc}"}

    # ------------------------------------------------------------------
    # 7) Archivos exigidos por el enunciado (rutas exactas)
    # ------------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = df[["store_id", "error_a", "error_b", "difference", "favors"]].copy()
    csv_out = out_dir / "store_differences.csv"
    df_out.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colores = ["#1f77b4" if v > 0 else ("#d62728" if v < 0 else "#7f7f7f") for v in d]
    ax.bar(df["store_id"], d, color=colores, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhspan(ci_low, ci_high, color="green", alpha=0.12,
               label=f"IC t 95 % = [{ci_low:.4f}, {ci_high:.4f}]")
    ax.axhline(mean, color="green", linestyle="--", linewidth=1.4,
               label=f"media = {mean:.4f}")
    ax.set_xlabel("Tienda (store_id)")
    ax.set_ylabel("Diferencia pareada error_a − error_b (unidades)")
    ax.set_title("T4 · Diferencias pareadas por tienda, media e IC t del 95 %")
    ax.text(
        0.995, 0.02,
        (f"n = {n}\nmedia = {mean:.6f}\nsd muestral = {sample_sd:.6f}\n"
         f"SE = {standard_error:.6f}\nt({dfree}) = {t_stat:.6f}\n"
         f"p (bilateral) = {p_value:.6f}"),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("output/store_differences.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) results.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    signo = "positiva" if mean > 0 else ("negativa" if mean < 0 else "nula")
    contiene_cero = bool(ci_low <= 0.0 <= ci_high)
    resultados = {
        "subtarea": "T4: efecto e incertidumbre de las diferencias pareadas",
        "fuente_datos": "data/model_errors.csv",
        "definicion_diferencia": "difference = error_a - error_b (positiva favorece a B)",
        "n": n,
        "n_esperado": 16,
        "n_coincide_con_esperado": bool(n == 16),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "regla_se": "standard_error = sample_sd / sqrt(n)",
        "ddof_usado_para_sd": 1,
        "sem_scipy_verificacion": sem_scipy,
        "t_statistic": t_stat,
        "degrees_of_freedom": dfree,
        "p_value_two_sided": p_value,
        "alternative": "two-sided",
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "confidence_level": 0.95,
        "prueba_t": {
            "nombre": "prueba t de una muestra sobre el vector de diferencias (equivalente a la t pareada)",
            "funcion_scipy": "scipy.stats.ttest_1samp",
            "media_nula": 0.0,
            "alternativa": "two-sided",
            "statistic": t_stat,
            "degrees_of_freedom": dfree,
            "p_value_two_sided": p_value,
            "p_value_verificacion_sf": p_check,
        },
        "intervalo_t": {
            "nombre": "IC t bilateral para la media de las diferencias (media pareada)",
            "funcion_scipy": "scipy.stats.t.interval",
            "confidence_level": 0.95,
            "low": ci_low,
            "high": ci_high,
            "t_critico": t_crit,
        },
        "interpretacion": {
            "signo_media": signo,
            "lectura_media": (
                f"La diferencia media es {mean:.6f} unidades: como difference = error_a - error_b, "
                f"un valor {signo} indica que, en promedio, el modelo B comete "
                f"{abs(mean):.6f} unidades {'menos' if mean > 0 else 'más'} de error que el A."
            ),
            "dispersion_vs_error_estandar": (
                f"La sd muestral ({sample_sd:.6f}) describe la dispersión del efecto entre tiendas; "
                f"el error estándar ({standard_error:.6f}) describe la incertidumbre de la media y "
                f"se obtiene como sd/sqrt(n) con n={n}."
            ),
            "lectura_intervalo": (
                f"El IC t del 95 % es [{ci_low:.6f}, {ci_high:.6f}] y "
                f"{'contiene' if contiene_cero else 'no contiene'} el valor 0 "
                f"(p bilateral = {p_value:.6f}); los efectos dentro del intervalo son compatibles "
                f"con los datos bajo el modelo t. El intervalo es un procedimiento frecuentista de "
                f"cobertura, no una probabilidad posterior sobre el parámetro."
            ),
        },
        "verificacion_con_T2": t2_check,
        "differences": [float(v) for v in d],
        "tabla_pareada": df_out.to_dict(orient="records"),
        "salidas": {
            "resultados_json": "resultados.json",
            "tabla_csv": "output/store_differences.csv",
            "figura_png": "output/store_differences.png",
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 9) Resumen
    # ------------------------------------------------------------------
    print("=" * 64)
    print("T4 · Efecto e incertidumbre de las diferencias pareadas")
    print("=" * 64)
    print(f"n tiendas               : {n}")
    print(f"diferencia media        : {mean:.6f} unidades")
    print(f"sd muestral (ddof=1)    : {sample_sd:.6f}")
    print(f"error estándar (sd/√n)  : {standard_error:.6f}")
    print(f"estadístico t({dfree})     : {t_stat:.6f}")
    print(f"p-value (bilateral)     : {p_value:.6f}")
    print(f"IC t 95 %               : [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"¿IC contiene 0?         : {'sí' if contiene_cero else 'no'}")
    print(f"CSV  -> {csv_out}")
    print("PNG  -> output/store_differences.png")
    print("JSON -> resultados.json")


if __name__ == "__main__":
    main()
