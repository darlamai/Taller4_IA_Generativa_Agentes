"""Parte 3 — lista los spans `guardrail` de una sesión de Langfuse, con su decisión y su latencia,
y comprueba que ningún dato personal de tus casos haya llegado a Langfuse.

    uv run python scripts/resumir_guardrails.py --sesion grupo-07-3-guardrails \\
        --prohibidas "Montoya" "1700000019" --salida results_parte3

Escribe `guardrails_spans.csv` (un renglón por span guardrail) y `guardrails_spans.md` (agregado por
guardrail: cuántas veces, qué decidió, latencia mediana y p95) y cuenta, sobre TODAS las observations de la
sesión (entrada, salida y metadata ya enmascaradas, que es lo que Langfuse guardó), cuántas contienen cada
literal de --prohibidas. Lo esperado: 0.
Necesita .env con LANGFUSE_* (lo mismo que exportar_trazas.py, del que reutiliza `traer`).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ.parent))          # donde vive exportar_trazas.py (carpeta del Lab)
sys.path.insert(0, str(RAIZ))


def _dato(valor):
    if isinstance(valor, str):
        try:
            return json.loads(valor)
        except ValueError:
            return valor
    return valor


def _ms(o: dict):
    ini, fin = o.get("startTime"), o.get("endTime")
    return round((fin - ini).total_seconds() * 1000, 2) if ini and fin else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesion", required=True)
    ap.add_argument("--prohibidas", nargs="*", default=[], help="literales que NO deben aparecer en ningún span")
    ap.add_argument("--salida", default=str(RAIZ / "results_parte3"))
    ap.add_argument("--desde", default=None, help="solo observations desde esta fecha ISO (UTC), p. ej. 2026-10-16T02:00:00+00:00")
    a = ap.parse_args()
    from exportar_trazas import traer
    obs = traer(tag=None, sesion=a.sesion, desde=a.desde, hasta=None)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    filas = []
    for o in obs:
        if str(o.get("type")).upper() != "GUARDRAIL":
            continue
        out = _dato(o.get("output"))
        out = out if isinstance(out, dict) else {"decision": str(out)}
        filas.append({"traza": o["traceId"], "guardrail": o.get("name"), "decision": out.get("decision", ""),
                      "latencia_ms": _ms(o), "nivel": o.get("level"),
                      "detalle": json.dumps({k: v for k, v in out.items() if k not in {"decision", "hallazgos"}},
                                            ensure_ascii=False)[:300],
                      "motivo": (out.get("violaciones") or [out.get("motivo", "")])[0] if
                                isinstance(out.get("violaciones", []), list) else ""})
    filas.sort(key=lambda f: (f["guardrail"], f["traza"]))
    if filas:
        with (salida / "guardrails_spans.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)

    por = defaultdict(list)
    for f in filas:
        por[f["guardrail"]].append(f)
    md = [f"# Spans guardrail de la sesión {a.sesion}", "",
          f"{len(obs)} observations en la sesión; {len(filas)} son guardrails.", "",
          "| guardrail | spans | decisiones | latencia mediana (ms) | p95 (ms) |", "|---|---|---|---|---|"]
    for nombre, lista in sorted(por.items()):
        lat = sorted(f["latencia_ms"] for f in lista if f["latencia_ms"] is not None) or [0]
        dec = ", ".join(f"{k}×{v}" for k, v in Counter(f["decision"] for f in lista).items())
        md.append(f"| {nombre} | {len(lista)} | {dec} | {statistics.median(lat):.2f} | {lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.2f} |")
    md += ["", "## Datos personales en los spans", ""]
    crudo = [json.dumps(o, ensure_ascii=False, default=str) for o in obs]
    for lit in a.prohibidas:
        n = sum(lit.lower() in c.lower() for c in crudo)
        md.append(f"- «{lit[:3]}…» aparece en {n} observation(s)" + (" ✓" if n == 0 else "  ← FUGA"))
    if not a.prohibidas:
        md.append("(no pasaste --prohibidas: no se comprobó)")
    (salida / "guardrails_spans.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n✓ {salida}/guardrails_spans.csv y .md")


if __name__ == "__main__":
    main()
