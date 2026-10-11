"""Prueba de humo de la Parte 1: ¿llegan los spans a Langfuse, con su jerarquía y sin PII?"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import solver.config  # noqa: E402,F401  (carga el .env)
os.environ["SOLVER_SESION"] = "humo-solver"

from solver import observabilidad as obs  # noqa: E402


def main() -> int:
    if not obs.activo():
        print("Langfuse no está activo: revisa LANGFUSE_* en .env (ver el aviso de arriba).")
        return 1
    pregunta = "Contacto: ana.perez@usfq.edu.ec, tel. 0991234567, cédula: 1712345678"
    with obs.corrida(pregunta, variante="humo") as raiz:
        with obs.guardrail("validar_entrada") as g:
            g.actualizar(output={"decision": "permitir"})
        with obs.agente("planificador"):
            with obs.generacion("planificador", modelo="modelo-de-juguete", input=pregunta) as gen:
                time.sleep(0.2)
                obs.usar_uso(gen, {"prompt_tokens": 120, "completion_tokens": 45}, output="plan de prueba")
        with obs.agente("ejecutor"):
            try:
                with obs.herramienta("sandbox_python", input={"huella": "abc"}) as t:
                    time.sleep(0.1)
                    t.actualizar(output={"returncode": 1, "tam_stderr": 20})
                    raise RuntimeError("fallo a propósito: este span debe salir en nivel ERROR")
            except RuntimeError:
                pass
        raiz.actualizar(output={"status": "ok"}, metadata={"status": "ok"})
        traza = obs.traza_id_actual()
    obs.vaciar()
    print(f"traza_id = {traza}\nsesión   = humo-solver\nEn unos segundos aparece en Langfuse → Tracing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())