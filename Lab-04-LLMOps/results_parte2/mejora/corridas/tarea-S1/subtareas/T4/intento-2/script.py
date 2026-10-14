# -*- coding: utf-8 -*-
"""T4 — Resumen por servidor.

Agrupa el DataFrame resultante de T3 (con load_score y requires_review) por la
columna `server` y construye una tabla con una fila por servidor y EXACTAMENTE
estas columnas, en este orden:
    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

donde:
    observations       -> número de filas del servidor
    mean_power_w       -> media de power_w
    max_temperature_c  -> máximo de temperature_c
    mean_load          -> media de load_score
    review_count       -> conteo de filas con requires_review=True

load_score se define (enunciado) como:  zscores @ [0.50, 0.30, 0.20]
    0.50*z(gpu_utilization) + 0.30*z(cpu_utilization) + 0.20*z(memory_gb)
Si el DataFrame de T3 no está disponible en entrada/T3/, se recalcula SIEMPRE
con esa combinación ponderada (nunca con z-scores de power_w).

Salidas:
- output/server_analysis.parquet (tabla resumen; ruta exigida por el enunciado)
- resultados.json                (cifras de la subtarea; contrato de salida)
- t4_resumen_servidores.png      (figura resumen)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_COLUMNS = [
    "server",
    "observations",
    "mean_power_w",
    "max_temperature_c",
    "mean_load",
    "review_count",
]

# Valores de referencia dados en el enunciado para AI-SRV-01
EXPECTED_SRV01 = {
    "observations": 50,
    "mean_power_w": 363.4598,
    "max_temperature_c": 86.50,
    "mean_load": -0.614527,
    "review_count": 2,
}

# Fórmula del enunciado: load_score = zscores @ [0.50, 0.30, 0.20]
Z_COLS = ["gpu_utilization", "cpu_utilization", "memory_gb"]
LOAD_WEIGHTS = np.array([0.50, 0.30, 0.20])


def _to_native(obj):
    """Convierte tipos de numpy a tipos nativos para serializar a JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def t3_metadata():
    """Lee de entrada/T3/resultados.json los load_score conocidos y n_filas.

    Tolera JSON truncado: si json.loads falla, recupera los pares
    (indice, load_score) mediante regex sobre el texto crudo.
    """
    p = Path("entrada") / "T3" / "resultados.json"
    pairs, n_filas = [], None
    if not p.exists():
        return pairs, n_filas
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return pairs, n_filas
    # 1) Parseo JSON completo
    try:
        res = json.loads(text)
        if isinstance(res.get("n_filas"), int):
            n_filas = res["n_filas"]
        regs = []
        if isinstance(res.get("fila_23"), dict):
            regs.append(res["fila_23"])
        marcadas = res.get("filas_marcadas")
        if isinstance(marcadas, list):
            regs.extend(r for r in marcadas if isinstance(r, dict))
        for r in regs:
            try:
                pairs.append((int(r["indice"]), float(r["load_score"])))
            except (KeyError, TypeError, ValueError):
                continue
        if pairs:
            return pairs, n_filas
    except json.JSONDecodeError:
        pass  # JSON truncado -> salvamento por regex
    # 2) Salvamento por regex (tolera el archivo truncado)
    m = re.search(r'"n_filas"\s*:\s*(\d+)', text)
    if m:
        n_filas = int(m.group(1))
    for m in re.finditer(r'"indice"\s*:\s*(\d+)', text):
        idx = int(m.group(1))
        seg = text[m.end(): m.end() + 500]
        lm = re.search(r'"load_score"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', seg)
        if lm:
            pairs.append((idx, float(lm.group(1))))
    return pairs, n_filas


def _normalize_review(d):
    """Garantiza que requires_review sea booleano."""
    if d["requires_review"].dtype != bool:
        d["requires_review"] = (
            d["requires_review"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
        )
    return d


def load_t3_dataframe():
    """Carga el DataFrame de T3 desde entrada/T3/ (parquet primero, luego csv).

    Valida columnas, número de filas y los valores de load_score conocidos.
    """
    needed = {"server", "power_w", "temperature_c", "load_score", "requires_review"}
    t3_dir = Path("entrada") / "T3"
    files = []
    if t3_dir.is_dir():
        files = sorted(t3_dir.rglob("*.parquet")) + sorted(t3_dir.rglob("*.csv"))
    pairs, n_esperado = t3_metadata()
    for f in files:
        try:
            d = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        except Exception:
            continue
        if not needed.issubset(d.columns) or len(d) == 0:
            continue
        if n_esperado is not None and len(d) != n_esperado:
            continue
        d = d.reset_index(drop=True)
        d["load_score"] = pd.to_numeric(d["load_score"], errors="coerce")
        if pairs:
            checks = [(i, v) for i, v in pairs if 0 <= i < len(d)]
            if not checks:
                continue
            vals = d["load_score"].to_numpy(dtype=float)
            if not all(np.isclose(vals[i], v, rtol=0.0, atol=1e-6) for i, v in checks):
                continue
        d = _normalize_review(d)
        return d, f"entrada/T3/{f.name}"
    return None, None


def weighted_load_score(raw, ddof):
    """load_score = zscores @ [0.50, 0.30, 0.20] (z de gpu, cpu y memoria)."""
    X = raw[Z_COLS].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=ddof)
    zscores = (X - mu) / sd
    return pd.Series(zscores @ LOAD_WEIGHTS, index=raw.index)


def recompute_t3_dataframe():
    """Reconstruye el DataFrame de T3 desde el CSV original.

    load_score se calcula SIEMPRE como zscores @ [0.50, 0.30, 0.20] sobre
    gpu_utilization, cpu_utilization y memory_gb (sin fallback a power_w).
    Si hay valores de referencia de T3, se usa la variante (ddof) que los
    reproduce; por defecto, ddof=1.
    """
    try:
        raw = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
    except Exception:
        raw = pd.read_csv("data/server_measurements.csv")
    pairs, _ = t3_metadata()
    options = {
        "zscores @ [0.50, 0.30, 0.20] (ddof=1)": weighted_load_score(raw, 1),
        "zscores @ [0.50, 0.30, 0.20] (ddof=0)": weighted_load_score(raw, 0),
    }
    chosen_name, chosen = None, None
    if pairs:
        for name, ls in options.items():
            checks = [(i, v) for i, v in pairs if 0 <= i < len(ls)]
            if checks and all(
                np.isclose(float(ls.iloc[i]), v, rtol=0.0, atol=1e-6) for i, v in checks
            ):
                chosen_name, chosen = name, ls
                break
    if chosen is None:
        chosen_name = "zscores @ [0.50, 0.30, 0.20] (ddof=1) [defecto del enunciado]"
        chosen = options["zscores @ [0.50, 0.30, 0.20] (ddof=1)"]
    d = raw.copy()
    d["load_score"] = chosen
    d["requires_review"] = (d["load_score"] > 1.5) | (d["temperature_c"] > 80.0)
    return d, f"recomputado desde data/server_measurements.csv (load_score = {chosen_name})"


def main():
    # 1) Obtener el DataFrame de T3 (load_score y requires_review incluidos)
    df, fuente = load_t3_dataframe()
    if df is None:
        df, fuente = recompute_t3_dataframe()
    print(f"Fuente de datos: {fuente}")

    df = df.copy()
    df["_review_int"] = df["requires_review"].astype(bool).astype(int)

    # 2) Agrupar por server y construir la tabla resumen
    g = df.groupby("server", sort=True)
    resumen = pd.DataFrame(
        {
            "observations": g.size(),
            "mean_power_w": g["power_w"].mean(),
            "max_temperature_c": g["temperature_c"].max(),
            "mean_load": g["load_score"].mean(),
            "review_count": g["_review_int"].sum(),
        }
    ).reset_index()
    resumen = resumen[EXPECTED_COLUMNS]  # orden exacto de columnas exigido
    resumen["observations"] = resumen["observations"].astype(int)
    resumen["review_count"] = resumen["review_count"].astype(int)

    if len(resumen) != 6:
        print(f"ADVERTENCIA: se esperaban 6 servidores, se obtuvieron {len(resumen)}")

    # 3) Imprimir la tabla completa de los seis servidores
    print("\nTabla resumen por servidor (formato de lectura):")
    tabla_print = resumen.copy()
    tabla_print["mean_power_w"] = tabla_print["mean_power_w"].map(lambda v: f"{v:.4f}")
    tabla_print["max_temperature_c"] = tabla_print["max_temperature_c"].map(lambda v: f"{v:.2f}")
    tabla_print["mean_load"] = tabla_print["mean_load"].map(lambda v: f"{v:.6f}")
    print(tabla_print.to_string(index=False))
    print("\nTabla con precisión completa:")
    print(resumen.to_string(index=False))

    # 4) Verificación contra el resultado esperado del enunciado (AI-SRV-01)
    fila01 = resumen.loc[resumen["server"] == "AI-SRV-01"].iloc[0]
    verificacion = {}
    for k, v_exp in EXPECTED_SRV01.items():
        v_obt = fila01[k]
        if k in ("observations", "review_count"):
            ok = int(v_obt) == int(v_exp)
            verificacion[k] = {"esperado": v_exp, "obtenido": int(v_obt), "ok": bool(ok)}
        else:
            ok = np.isclose(float(v_obt), float(v_exp), rtol=0.0, atol=1e-3)
            verificacion[k] = {"esperado": v_exp, "obtenido": float(v_obt), "ok": bool(ok)}

    # 5) Parquet exigido por el enunciado
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    resumen.to_parquet("output/server_analysis.parquet", index=False)

    # 6) Figura resumen
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(len(resumen))
    panels = [
        ("mean_power_w", "Potencia media (W)"),
        ("max_temperature_c", "Temperatura máxima (°C)"),
        ("mean_load", "Carga media (load_score)"),
        ("review_count", "Observaciones marcadas para revisión"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.bar(x, resumen[col], color="#4C72B0")
        ax.set_xticks(x)
        ax.set_xticklabels(resumen["server"], rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("T4 — Resumen por servidor")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("t4_resumen_servidores.png", dpi=120)
    plt.close(fig)

    # 7) resultados.json (contrato de la subtarea, valores sin redondear)
    tabla_redondeada = resumen.copy()
    tabla_redondeada["mean_power_w"] = tabla_redondeada["mean_power_w"].round(4)
    tabla_redondeada["max_temperature_c"] = tabla_redondeada["max_temperature_c"].round(2)
    tabla_redondeada["mean_load"] = tabla_redondeada["mean_load"].round(6)

    resultados = {
        "subtarea": "T4_resumen_por_servidor",
        "fuente_datos": fuente,
        "load_score_formula": "zscores @ [0.50, 0.30, 0.20] sobre "
        "gpu_utilization, cpu_utilization, memory_gb",
        "pesos_load_score": [0.50, 0.30, 0.20],
        "n_filas_entrada": int(len(df)),
        "n_servidores": int(len(resumen)),
        "columnas": EXPECTED_COLUMNS,
        "orden_columnas_correcto": list(resumen.columns) == EXPECTED_COLUMNS,
        "tabla": _to_native(resumen.to_dict(orient="records")),
        "tabla_redondeada": _to_native(tabla_redondeada.to_dict(orient="records")),
        "AI-SRV-01": {
            "observations": int(fila01["observations"]),
            "mean_power_w": float(fila01["mean_power_w"]),
            "max_temperature_c": float(fila01["max_temperature_c"]),
            "mean_load": float(fila01["mean_load"]),
            "review_count": int(fila01["review_count"]),
        },
        "verificacion_AI-SRV-01": verificacion,
        "salidas": ["output/server_analysis.parquet", "t4_resumen_servidores.png"],
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(_to_native(resultados), fh, indent=2, ensure_ascii=False)

    # 8) Resumen breve en consola
    print("\nVerificación AI-SRV-01 frente al enunciado:")
    for k, v in verificacion.items():
        print(
            f"  {k}: obtenido={v['obtenido']} | esperado={v['esperado']} "
            f"-> {'OK' if v['ok'] else 'DIFERENTE'}"
        )
    print(f"\nFilas de entrada: {len(df)} | Servidores en el resumen: {len(resumen)}")
    print("Archivos escritos: output/server_analysis.parquet, resultados.json, t4_resumen_servidores.png")


if __name__ == "__main__":
    main()
