"""Parte 3 — mide los guardrails como clasificadores y escribe los entregables.

    python scripts/medir_guardas.py                      # desde la carpeta Taller3-opcionB
    python scripts/medir_guardas.py --lab ..             # si el Lab no está justo encima

Corre, sobre los mismos bancos, (1) los guardrails de `solver/guardrails.py` y (2) los de ejemplo del
Lab (`guardrails.py`), y el guardián del sandbox (`solver/sandbox/guarda.py`) sobre `ataques_codigo.json`.
Escribe en la carpeta de salida (por defecto `results_parte3/`) la medición EXTENDIDA (los entregables
`guardrails.csv` y `guardrails_ejemplo.csv` salen de probar_guardrails.py, ver el paso a paso):

    medicion_mios.csv        los míos, una fila por caso (banco, guardrail, esperado, obtenido, veredicto, latencia)
    medicion_ejemplo.csv   los de ejemplo del Lab, mismas columnas
    medicion_sandbox.csv   el guardián del sandbox, una fila por script
    medicion_resumen.md    falsos negativos / falsos positivos / latencia, por conjunto y por guardrail
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

REPETICIONES = 30


def _cargar(ruta: Path, nombre: str):
    espec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(espec)
    espec.loader.exec_module(modulo)
    return modulo


def _latencia_us(fn, texto: str) -> float:
    medidas = []
    for _ in range(REPETICIONES):
        t0 = time.perf_counter_ns()
        try:
            fn(texto)
        except Exception:
            pass
        medidas.append((time.perf_counter_ns() - t0) / 1000)
    return round(statistics.median(medidas), 1)


def medir_texto(probar, casos: list[dict], v_entrada, v_salida, banco: str) -> list[dict]:
    filas = probar.probar(casos, v_entrada, v_salida)
    por_id = {c["id"]: c for c in casos}
    for f in filas:
        c = por_id[f["id"]]
        validador = v_entrada if c["canal"] == "entrada" else v_salida
        f["banco"] = banco
        f["guardrail"] = c.get("guardrail", "")
        f["latencia_us"] = _latencia_us(validador, c["texto"])
    return filas


def medir_sandbox(casos: list[dict]) -> list[dict]:
    from solver.sandbox.guarda import revisar
    filas = []
    for c in casos:
        v = revisar(c["codigo"])
        obtenido = "bloquear" if not v.permitido else "confirmar" if v.descargas else "pasar"
        esperado = c["esperado"]
        if esperado == obtenido:
            ver = "correcto"
        elif esperado == "pasar":
            ver = "falso_positivo"
        elif obtenido == "pasar":
            ver = "falso_negativo"
        else:
            ver = "otra_accion"
        filas.append({"id": c["id"], "esperado": esperado, "obtenido": obtenido, "veredicto": ver,
                      "violaciones": " | ".join(v.violaciones + v.descargas),
                      "latencia_us": _latencia_us(revisar, c["codigo"]), "por_que": c.get("por_que", "")})
    return filas


def _tasas(filas: list[dict]) -> dict:
    ataques = [f for f in filas if f["esperado"] != "pasar"]
    legit = [f for f in filas if f["esperado"] == "pasar"]
    c = lambda lista, v: sum(f["veredicto"] == v for f in lista)          # noqa: E731
    lat = sorted(f["latencia_us"] for f in filas) or [0]
    return {"casos": len(filas), "correctos": c(filas, "correcto"),
            "fn": c(ataques, "falso_negativo"), "de_ataques": len(ataques),
            "fp": c(legit, "falso_positivo"), "de_legitimas": len(legit),
            "otra": c(filas, "otra_accion"), "excepciones": c(filas, "excepcion"),
            "lat_mediana_us": round(statistics.median(lat), 1),
            "lat_p95_us": lat[min(len(lat) - 1, int(0.95 * len(lat)))]}


def _tabla(titulo: str, grupos: dict[str, list[dict]]) -> str:
    out = [f"### {titulo}", "",
           "| grupo | casos | correctos | falsos negativos | falsos positivos | otra acción | latencia mediana (µs) | p95 (µs) |",
           "|---|---|---|---|---|---|---|---|"]
    for nombre, filas in grupos.items():
        t = _tasas(filas)
        out.append(f"| {nombre} | {t['casos']} | {t['correctos']} | {t['fn']}/{t['de_ataques']} | "
                   f"{t['fp']}/{t['de_legitimas']} | {t['otra']} | {t['lat_mediana_us']} | {t['lat_p95_us']} |")
    return "\n".join(out) + "\n"


def _escribir(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", default=str(RAIZ.parent), help="carpeta del Lab (donde están probar_guardrails.py y guardrails.py)")
    ap.add_argument("--salida", default=str(RAIZ / "results_parte3"))
    a = ap.parse_args()
    lab, salida = Path(a.lab), Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    probar = _cargar(lab / "probar_guardrails.py", "probar_guardrails")
    ejemplo = _cargar(lab / "guardrails.py", "guardrails_ejemplo")
    from solver import guardrails as mio

    bancos = {"ataques": RAIZ / "ataques.json", "extra": RAIZ / "ataques_extra.json"}
    casos = {n: json.loads(r.read_text(encoding="utf-8"))["casos"] for n, r in bancos.items() if r.exists()}

    mis, ej = [], []
    for nombre, lista in casos.items():
        mis += medir_texto(probar, lista, mio.validar_entrada, mio.validar_salida, nombre)
        ej += medir_texto(probar, lista, ejemplo.validate_input, ejemplo.validate_output, nombre)
    sandbox = medir_sandbox(json.loads((RAIZ / "ataques_codigo.json").read_text(encoding="utf-8"))["casos"])

    _escribir(salida / "medicion_mios.csv", mis)
    _escribir(salida / "medicion_ejemplo.csv", ej)
    _escribir(salida / "medicion_sandbox.csv", sandbox)

    md = ["# Parte 3 — guardrails medidos como clasificadores", "",
          "Falsos negativos = ataques/datos que pasaron; falsos positivos = legítimas bloqueadas o reescritas.", ""]
    for titulo, filas in (("Mis guardrails (solver/guardrails.py)", mis), ("Guardrails de ejemplo del Lab", ej)):
        md.append(_tabla(f"{titulo} — por banco", {n: [f for f in filas if f["banco"] == n] for n in casos}))
        md.append(_tabla(f"{titulo} — por guardrail (todos los bancos)",
                         {g: [f for f in filas if f["guardrail"] == g] for g in ("inyeccion", "pii", "secretos")}))
    md.append(_tabla("Sandbox como guardrail (guarda.py, ataques_codigo.json)", {"sandbox": sandbox}))
    md.append("## Casos que fallan\n")
    for titulo, filas in (("míos", mis), ("ejemplo", ej), ("sandbox", sandbox)):
        malos = [f for f in filas if f["veredicto"] != "correcto"]
        md.append(f"**{titulo}**: {len(malos)} de {len(filas)}")
        for f in malos:
            md.append(f"- {f.get('banco', 'codigo')}/{f['id']}: esperado {f['esperado']}, obtuvo {f['obtenido']} ({f['veredicto']})")
        md.append("")
    (salida / "medicion_resumen.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✓ escrito en {salida}")


if __name__ == "__main__":
    main()
