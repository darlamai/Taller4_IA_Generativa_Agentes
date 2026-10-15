"""Parte 4.b — sacar los umbrales de las alertas de TUS spans exportados.

    uv run python scripts/umbrales_4b.py ../results_parte2/base/spans.jsonl ../results_parte2/control/spans.jsonl ../results_parte2/mejora/spans.jsonl > ../results_parte4_umbrales.md

Una fila de spans.jsonl es un span (columnas de exportar_trazas.py). Por cada corrida (traza):
  duración = su span `corrida`; tokens = suma de entrada+salida de sus spans `llm`;
  fallida  = su span `corrida` trae `error` (el orquestador marca error si status == fallido).
Imprime, por archivo y en total: n, mediana, p95 y máximo de duración y de tokens (solo corridas sin error,
que son «lo normal»), y la fracción de corridas fallidas.
"""
import json
import statistics
import sys
from collections import defaultdict


def cargar(ruta):
    corridas = defaultdict(lambda: {"s": None, "tok": 0, "error": False})
    for linea in open(ruta, encoding="utf-8"):
        s = json.loads(linea)
        c = corridas[s["traza"]]
        if s["span"] == "corrida":
            c["s"] = (s.get("duracion_ms") or 0) / 1000
            c["error"] = bool(s.get("error"))
        elif s["span"] == "llm":
            c["tok"] += (s.get("tokens_entrada") or 0) + (s.get("tokens_salida") or 0)
    return [c for c in corridas.values() if c["s"] is not None]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def fila(nombre, cs):
    ok = [c for c in cs if not c["error"]] or cs
    d, t = [c["s"] for c in ok], [c["tok"] for c in ok]
    return (f"| {nombre} | {len(cs)} | {sum(c['error'] for c in cs)} | "
            f"{statistics.median(d):.0f} / {pct(d, .95):.0f} / {max(d):.0f} | "
            f"{statistics.median(t):,.0f} / {pct(t, .95):,.0f} / {max(t):,.0f} |")


todas = []
print("| archivo | corridas | con error | duración s (mediana / p95 / máx, sin errores) | tokens (mediana / p95 / máx, sin errores) |")
print("|---|---|---|---|---|")
for ruta in sys.argv[1:]:
    cs = cargar(ruta)
    todas += cs
    print(fila(ruta.split("/")[-2] if "/" in ruta else ruta, cs))
print(fila("TODAS", todas))
print(f"\nfallidas: {sum(c['error'] for c in todas)} de {len(todas)} = {sum(c['error'] for c in todas) / len(todas):.1%}")