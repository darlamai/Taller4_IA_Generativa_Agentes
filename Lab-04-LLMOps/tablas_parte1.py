"""Tablas de la Parte 1 a partir de spans.jsonl (lo que escribe exportar_trazas.py).

    python tablas_parte1.py spans.jsonl                    # imprime en Markdown
    python tablas_parte1.py spans.jsonl --csv carpeta/     # y deja un CSV por tabla

Complementa a resumir_trazas.py (costo con fila y fecha) con: latencia por corrida (p50/p95),
tiempo por tipo de span, tokens y tiempo POR AGENTE y la respuesta a «¿dónde se va el tiempo?».

Dos cuidados al sumar tiempos:
  * `agent` y `corrida` CONTIENEN a los demás spans: no se suman con los hijos.
  * Los spans `llm` de un mismo agente pueden correr en PARALELO (hilos del GraphRAG): su suma
    puede superar el tiempo real. Por eso cada agente muestra su tiempo de pared (lo que dura
    su span `agent`) y, aparte, la suma de sus llm con su "concurrencia".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    filas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, f in df.iterrows():
        filas.append("| " + " | ".join(
            (f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}") if isinstance(v, float) else str(v)
            for v in f) + " |")
    return "\n".join(filas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spans")
    ap.add_argument("--csv", help="carpeta donde guardar cada tabla")
    a = ap.parse_args()
    df = pd.DataFrame([json.loads(l) for l in Path(a.spans).read_text(encoding="utf-8").splitlines() if l.strip()])
    for c in ("duracion_ms", "tokens_entrada", "tokens_salida"):
        df[c] = pd.to_numeric(df[c] if c in df else pd.Series(dtype=float), errors="coerce")
    df["seg"] = df["duracion_ms"] / 1000
    df["tiene_error"] = df["error"].notna() if "error" in df else False
    tablas: dict[str, pd.DataFrame] = {}

    # --- 1. corridas: una fila por traza
    llm = df[df["span"] == "llm"]
    por = df.groupby("traza")
    corr = df[df["span"] == "corrida"].set_index("traza")
    runs = pd.DataFrame({
        "duracion_s": corr["seg"],
        "tokens_entrada": llm.groupby("traza")["tokens_entrada"].sum(),
        "tokens_salida": llm.groupby("traza")["tokens_salida"].sum(),
        "llamadas_llm": llm.groupby("traza").size(),
        "spans_con_error": por["tiene_error"].sum().astype(int),
    }).fillna(0).reset_index()
    tablas["corridas"] = runs
    dur = runs["duracion_s"]
    resumen = pd.DataFrame([{
        "corridas": len(runs), "tokens_entrada": int(runs["tokens_entrada"].sum()),
        "tokens_salida": int(runs["tokens_salida"].sum()),
        "latencia_p50_s": float(dur.quantile(0.5)), "latencia_p95_s": float(dur.quantile(0.95)),
        "latencia_max_s": float(dur.max())}])
    tablas["resumen_de_corridas"] = resumen

    # --- 2. tiempo por tipo de span
    por_tipo = (df.groupby("span").agg(spans=("span", "size"), total_s=("seg", "sum"), promedio_s=("seg", "mean"),
                                       errores=("tiene_error", "sum")).reset_index())
    por_tipo["errores"] = por_tipo["errores"].astype(int)
    tablas["tiempo_por_tipo_de_span"] = por_tipo.sort_values("total_s", ascending=False)

    # --- 3. por agente: tiempo de pared (span agent) + tokens y tiempo de sus llm
    total_corridas = float(dur.sum())
    pared = (df[df["span"] == "agent"].groupby("nombre")
             .agg(activaciones=("seg", "size"), tiempo_pared_s=("seg", "sum")))
    gasto = (llm.groupby("nombre")
             .agg(llamadas_llm=("seg", "size"), tokens_entrada=("tokens_entrada", "sum"),
                  tokens_salida=("tokens_salida", "sum"), suma_llm_s=("seg", "sum")))
    ag = pared.join(gasto, how="outer").fillna(0).reset_index().rename(columns={"nombre": "agente"})
    ag["pct_del_tiempo_de_corridas"] = 100 * ag["tiempo_pared_s"] / total_corridas if total_corridas else 0
    ag["tokens_totales"] = ag["tokens_entrada"] + ag["tokens_salida"]
    ag["concurrencia_llm"] = (ag["suma_llm_s"] / ag["tiempo_pared_s"]).where(ag["tiempo_pared_s"] > 0, 0)
    for c in ("activaciones", "llamadas_llm", "tokens_entrada", "tokens_salida", "tokens_totales"):
        ag[c] = ag[c].astype(int)
    tablas["tokens_y_tiempo_por_agente"] = ag.sort_values("tiempo_pared_s", ascending=False)

    # --- 4. dónde se va el tiempo: spans hoja (llm, herramienta, guardrail)
    hojas = df[df["span"].isin(["llm", "herramienta", "guardrail"])]
    cuello = (hojas.groupby(["span", "nombre"]).agg(spans=("seg", "size"), total_s=("seg", "sum"),
                                                    promedio_s=("seg", "mean")).reset_index())
    cuello["pct_de_la_suma_hoja"] = 100 * cuello["total_s"] / cuello["total_s"].sum()
    tablas["donde_se_va_el_tiempo_hojas"] = cuello.sort_values("total_s", ascending=False)

    # --- salida
    print(f"{len(runs)} corridas · {int(df['tiene_error'].sum())} spans con error\n")
    for nombre, t in tablas.items():
        print(f"### {nombre}\n\n{md(t)}\n")
    top_ag = tablas["tokens_y_tiempo_por_agente"].iloc[0]
    print(f"Agente que más TIEMPO consume (pared): {top_ag['agente']} con {top_ag['tiempo_pared_s']:.1f} s "
          f"({top_ag['pct_del_tiempo_de_corridas']:.0f} % del tiempo de las corridas).")
    top_tok = tablas["tokens_y_tiempo_por_agente"].sort_values("tokens_totales", ascending=False).iloc[0]
    print(f"Agente que más TOKENS gasta: {top_tok['agente']} con {top_tok['tokens_totales']:,} "
          f"({top_tok['tokens_entrada']:,} de entrada, {top_tok['tokens_salida']:,} de salida).")
    top = tablas["donde_se_va_el_tiempo_hojas"].iloc[0]
    prom = tablas["donde_se_va_el_tiempo_hojas"].sort_values("promedio_s", ascending=False).iloc[0]
    print(f"Cuello de botella entre spans hoja (mayor tiempo TOTAL): {top['span']}:{top['nombre']} con "
          f"{top['total_s']:.1f} s ({top['pct_de_la_suma_hoja']:.0f} % de la suma de hojas).")
    if (prom["span"], prom["nombre"]) != (top["span"], top["nombre"]):
        print(f"OJO: el de mayor promedio es otro ({prom['span']}:{prom['nombre']}, {prom['promedio_s']:.1f} s "
              f"por llamada); no es el cuello de botella.")
    suma_hojas = float(hojas["seg"].sum())
    if total_corridas and suma_hojas > 1.05 * total_corridas:
        print(f"AVISO: la suma de spans hoja ({suma_hojas:.0f} s) supera el tiempo de las corridas "
              f"({total_corridas:.0f} s): hay llamadas en paralelo. Para el reparto del tiempo real usa "
              "la tabla por agente (tiempo de pared).")
    if a.csv:
        Path(a.csv).mkdir(parents=True, exist_ok=True)
        for nombre, t in tablas.items():
            t.to_csv(Path(a.csv) / f"{nombre}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())