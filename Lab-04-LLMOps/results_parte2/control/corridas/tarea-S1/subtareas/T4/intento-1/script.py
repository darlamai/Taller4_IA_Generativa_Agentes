#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Construir el resumen por servidor.

Agrupa el DataFrame de T3 por `server` y construye una tabla con una fila por
servidor y EXACTAMENTE estas columnas, en este orden:

    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

donde:
    observations      -> número de filas del servidor
    mean_power_w      -> media de power_w
    max_temperature_c -> máximo de temperature_c
    mean_load         -> media de load_score
    review_count      -> conteo de filas del servidor con requires_review=True

Imprime la tabla completa de los seis servidores, escribe resultados.json,
garantiza el parquet exigido por el enunciado (output/server_analysis.parquet,
con el DataFrame de T3) y guarda una figura PNG de apoyo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_PARQUET = Path("output/server_analysis.parquet")
ORDEN_COLUMNAS = [
    "server", "observations", "mean_power_w",
    "max_temperature_c", "mean_load", "review_count",
]
PESOS = np.array([0.50, 0.30, 0.20])
ESPERADO_01 = {
    "observations": 50,
    "mean_power_w": 363.4598,
    "max_temperature_c": 86.50,
    "mean_load": -0.614527,
    "review_count": 2,
}


def _lista_numerica(obj, n):
    return (
        isinstance(obj, list)
        and len(obj) == n
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj)
    )


def _buscar_lista(obj, n):
    if _lista_numerica(obj, n):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = _buscar_lista(v, n)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _buscar_lista(v, n)
            if r is not None:
                return r
    return None


def _leer_json(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cargar_dataframe_t3():
    """Devuelve (DataFrame de T3 con load_score y requires_review, fuente)."""
    necesarias = {"server", "load_score", "requires_review"}

    # 1) Parquet producido por T3: copia en entrada/T3/ o salida original en output/
    candidatos = []
    if Path("entrada/T3").is_dir():
        candidatos += sorted(Path("entrada/T3").glob("*.parquet"))
    candidatos.append(RUTA_PARQUET)
    for p in candidatos:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if necesarias.issubset(df.columns):
            return df, str(p)

    # 2) Reconstrucción: CSV base + load_score de T2 (json o npy)
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
    n = len(df)
    load_score, fuente = None, None

    t2 = _leer_json(Path("entrada/T2/resultados.json"))
    if isinstance(t2, dict):
        ls = t2.get("load_score")
        if _lista_numerica(ls, n):
            load_score = np.asarray(ls, dtype=float)
            fuente = "entrada/T2/resultados.json (load_score)"
        else:
            z = t2.get("zscores")
            if isinstance(z, list) and len(z) == n and isinstance(z[0], list) and len(z[0]) == 3:
                load_score = np.asarray(z, dtype=float) @ PESOS
                fuente = "entrada/T2/resultados.json (zscores @ [0.50, 0.30, 0.20])"
            else:
                ls = _buscar_lista(t2, n)
                if ls is not None:
                    load_score = np.asarray(ls, dtype=float)
                    fuente = "entrada/T2/resultados.json (lista de 300 valores)"

    if load_score is None and Path("entrada/T2").is_dir():
        for p in sorted(Path("entrada/T2").glob("*.npy")):
            try:
                arr = np.load(p, allow_pickle=False)
            except Exception:
                continue
            if arr.shape == (n,):
                load_score = arr.astype(float)
                fuente = str(p)
                break
            if arr.shape == (n, 3):
                load_score = arr.astype(float) @ PESOS
                fuente = f"{p} @ [0.50, 0.30, 0.20]"
                break

    if load_score is None:
        # Último recurso: recalcular zscores([gpu, cpu, memory]) @ pesos,
        # calibrando ddof contra el resumen y los índices de revisión de T3.
        t3 = _leer_json(Path("entrada/T3/resultados.json")) or {}
        ref = t3.get("load_score_resumen", {}) if isinstance(t3, dict) else {}
        idx_ref = set(t3.get("indices_requieren_revision", [])) if isinstance(t3, dict) else set()
        X = df[["gpu_utilization", "cpu_utilization", "memory_gb"]].to_numpy(dtype=float)
        temp = df["temperature_c"].to_numpy(dtype=float)

        def costo(ls):
            c = (
                abs(float(np.std(ls)) - float(ref.get("desviacion", 0.0)))
                + abs(float(np.max(ls)) - float(ref.get("max", 0.0)))
                + abs(float(np.min(ls)) - float(ref.get("min", 0.0)))
            )
            if idx_ref:
                idx = set(np.flatnonzero((ls > 1.5) | (temp > 80.0)).tolist())
                c += 0.1 * len(idx.symmetric_difference(idx_ref))
            return c

        opciones = []
        for ddof in (0, 1):
            Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=ddof)
            opciones.append((costo(Z @ PESOS), ddof, Z @ PESOS))
        opciones.sort(key=lambda t: t[0])
        load_score = opciones[0][2]
        fuente = f"recalculado (zscores ddof={opciones[0][1]} @ [0.50, 0.30, 0.20])"

    df = df.copy()
    df["load_score"] = np.asarray(load_score, dtype=float)
    df["requires_review"] = ((df["load_score"] > 1.5) | (df["temperature_c"] > 80.0)).astype(bool)
    return df, fuente


def main():
    df, fuente = cargar_dataframe_t3()
    df = df.reset_index(drop=True)
    df["load_score"] = df["load_score"].astype(float)
    df["requires_review"] = df["requires_review"].astype(bool)

    # ---------- Tabla resumen por servidor (una fila por server) ----------
    g = df.groupby("server", sort=True)
    resumen = pd.DataFrame(
        {
            "observations": g.size(),
            "mean_power_w": g["power_w"].mean(),
            "max_temperature_c": g["temperature_c"].max(),
            "mean_load": g["load_score"].mean(),
            "review_count": g["requires_review"].sum(),
        }
    ).reset_index()
    resumen["observations"] = resumen["observations"].astype(int)
    resumen["review_count"] = resumen["review_count"].astype(int)
    resumen = resumen[ORDEN_COLUMNAS]  # orden exacto de columnas exigido

    # ---------- Impresión de la tabla completa (formato del enunciado) ----------
    mostrada = resumen.copy()
    mostrada["mean_power_w"] = mostrada["mean_power_w"].map("{:.4f}".format)
    mostrada["max_temperature_c"] = mostrada["max_temperature_c"].map("{:.2f}".format)
    mostrada["mean_load"] = mostrada["mean_load"].map("{:.6f}".format)
    print("Tabla resumen por servidor (T4):")
    print(mostrada.to_string(index=False))

    # ---------- Verificaciones ----------
    fila01 = resumen.loc[resumen["server"] == "AI-SRV-01"]
    tiene_01 = len(fila01) == 1
    verif_01 = {"existe_fila": bool(tiene_01)}
    if tiene_01:
        f = fila01.iloc[0]
        verif_01.update(
            {
                "observations": int(f["observations"]),
                "mean_power_w": float(f["mean_power_w"]),
                "max_temperature_c": float(f["max_temperature_c"]),
                "mean_load": float(f["mean_load"]),
                "review_count": int(f["review_count"]),
                "observations_ok": int(f["observations"]) == ESPERADO_01["observations"],
                "mean_power_w_ok": abs(float(f["mean_power_w"]) - ESPERADO_01["mean_power_w"]) <= 5.1e-5,
                "max_temperature_c_ok": abs(float(f["max_temperature_c"]) - ESPERADO_01["max_temperature_c"]) <= 5.1e-3,
                "mean_load_ok": abs(float(f["mean_load"]) - ESPERADO_01["mean_load"]) <= 5.1e-7,
                "review_count_ok": int(f["review_count"]) == ESPERADO_01["review_count"],
            }
        )

    # Contraste con los índices de revisión reportados por T3
    t3 = _leer_json(Path("entrada/T3/resultados.json")) or {}
    idx_t3 = t3.get("indices_requieren_revision", []) if isinstance(t3, dict) else []
    coincide_t3 = None
    if idx_t3 and max(idx_t3) < len(df):
        conteo_t3 = df.loc[sorted(idx_t3), "server"].value_counts().to_dict()
        conteo_tabla = dict(zip(resumen["server"], resumen["review_count"]))
        coincide_t3 = all(int(conteo_t3.get(s, 0)) == int(c) for s, c in conteo_tabla.items())

    checks = {
        "seis_filas": int(len(resumen)) == 6,
        "orden_columnas_exacto": list(resumen.columns) == ORDEN_COLUMNAS,
        "existe_fila_ai_srv_01": bool(tiene_01),
    }
    checks.update({k: bool(v) for k, v in verif_01.items() if k.endswith("_ok")})
    checks["total_observaciones_300"] = int(resumen["observations"].sum()) == 300
    checks["total_revision_49_coincide_t3"] = (
        int(resumen["review_count"].sum()) == 49 and (coincide_t3 is not False)
    )
    criterio_exitoso = bool(all(checks.values()))

    # ---------- Registros para resultados.json ----------
    tabla, tabla_redondeada = [], []
    for _, f in resumen.iterrows():
        tabla.append(
            {
                "server": str(f["server"]),
                "observations": int(f["observations"]),
                "mean_power_w": float(f["mean_power_w"]),
                "max_temperature_c": float(f["max_temperature_c"]),
                "mean_load": float(f["mean_load"]),
                "review_count": int(f["review_count"]),
            }
        )
        tabla_redondeada.append(
            {
                "server": str(f["server"]),
                "observations": int(f["observations"]),
                "mean_power_w": round(float(f["mean_power_w"]), 4),
                "max_temperature_c": round(float(f["max_temperature_c"]), 2),
                "mean_load": round(float(f["mean_load"]), 6),
                "review_count": int(f["review_count"]),
            }
        )

    # ---------- Figura de apoyo ----------
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    ejes[0].bar(resumen["server"], resumen["observations"], color="#4C72B0")
    ejes[0].set_title("Observaciones por servidor")
    ejes[0].set_ylabel("observations")
    ejes[0].tick_params(axis="x", rotation=45)
    ejes[1].bar(resumen["server"], resumen["review_count"], color="#C44E52")
    ejes[1].set_title("Filas con requires_review=True por servidor")
    ejes[1].set_ylabel("review_count")
    ejes[1].tick_params(axis="x", rotation=45)
    for e in ejes:
        e.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("t4_resumen_servidores.png", dpi=120)
    plt.close(fig)

    # ---------- Parquet exigido por el enunciado (DataFrame de T3) ----------
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RUTA_PARQUET, index=False)

    # ---------- resultados.json ----------
    resultados = {
        "subtarea": "T4",
        "descripcion": (
            "Tabla resumen por servidor (una fila por server) con observations, "
            "mean_power_w, max_temperature_c, mean_load y review_count, en ese orden"
        ),
        "fuente_dataframe": fuente,
        "n_filas_dataframe": int(len(df)),
        "n_servidores": int(len(resumen)),
        "servidores": [str(s) for s in resumen["server"]],
        "columnas": ORDEN_COLUMNAS,
        "orden_columnas_correcto": bool(list(resumen.columns) == ORDEN_COLUMNAS),
        "definiciones": {
            "observations": "número de filas por servidor",
            "mean_power_w": "media de power_w por servidor",
            "max_temperature_c": "máximo de temperature_c por servidor",
            "mean_load": "media de load_score por servidor",
            "review_count": "conteo de filas con requires_review=True por servidor",
        },
        "tabla_resumen": tabla,
        "tabla_resumen_redondeada": tabla_redondeada,
        "total_observaciones": int(resumen["observations"].sum()),
        "total_requieren_revision": int(resumen["review_count"].sum()),
        "coincide_revision_con_t3": coincide_t3,
        "verificacion_ai_srv_01": verif_01,
        "checks": checks,
        "criterio_exitoso": criterio_exitoso,
        "figura": "t4_resumen_servidores.png",
        "parquet_escrito": str(RUTA_PARQUET),
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ---------- Resumen breve ----------
    print("\nResumen T4:")
    print(f"  Fuente del DataFrame de T3 : {fuente}")
    print(f"  Servidores                 : {len(resumen)} | Observaciones: {int(resumen['observations'].sum())}")
    print(f"  Filas con revisión (total) : {int(resumen['review_count'].sum())}")
    print(f"  Criterio de éxito          : {criterio_exitoso}")
    print("  Salidas: resultados.json, t4_resumen_servidores.png, output/server_analysis.parquet")


if __name__ == "__main__":
    main()
