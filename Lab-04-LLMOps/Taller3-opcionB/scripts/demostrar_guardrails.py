"""Parte 3 — hace disparar, sin modelo, los guardrails que una corrida real del golden set no dispara.

    SOLVER_SESION=grupo-07-3-guardrails SOLVER_PROMPT_VERSION=v3 uv run python scripts/demostrar_guardrails.py

Los dos guardrails de ENTRADA se disparan de verdad con las tareas I1 (instrucción escondida) y P1
(datos personales) del golden set. Este script cubre lo que depende de qué escriba el modelo:
  · el sandbox rechazando scripts (`guarda_sandbox` con decisión «bloquear» y sus violaciones),
  · el guardrail de salida redactando un informe con un correo y una credencial (inventados aquí).
Usa el código REAL del solver (ejecutor.ejecutar y Solver._sanear_salida), solo que con entradas fijas.
Todo ocurre dentro de UNA corrida, así que en Langfuse aparece como una traza con sus spans `guardrail`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from solver import observabilidad                                  # noqa: E402
from solver.agentes import ejecutor                                # noqa: E402
from solver.orquestador import Solver                              # noqa: E402
from solver.traza import Traza                                     # noqa: E402

SCRIPTS = {
    "demo-permitido": "print(sum(range(10)))\n",
    "demo-proceso": "import os\nos.system('echo hola')\n",
    "demo-entorno": "import os\nprint(os.environ['HOME'])\n",
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with observabilidad.corrida("demostración de guardrails (Parte 3)", variante="demo"):
            for nombre, codigo in SCRIPTS.items():
                r = ejecutor.ejecutar(codigo, tmp / nombre, subtarea=nombre)
                print(f"{nombre:16s} returncode={r['returncode']} violaciones={r['violaciones']}")
            solver = object.__new__(Solver)
            solver.traza = Traza(tmp / "traza.jsonl")
            # inventados en tiempo de ejecución: ningún valor con forma de clave vive en el repositorio
            informe = ("Resultados listos. Contacto: ana.demo@ejemplo.ec. "
                       + "OPENAI_API_KEY=sk-" + "proj-" + "x" * 30)
            limpio = solver._sanear_salida(informe)
            print("salida:", limpio)
    observabilidad.vaciar()
    print("✓ listo: espera ~10 s y exporta la sesión con exportar_trazas.py")


if __name__ == "__main__":
    main()
