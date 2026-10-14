"""Parte 2.c — mejora v2: el freno de presupuesto mira lo que CUESTA la llamada, no solo que quede algo.

    uv run python scripts/mejora_2c.py            # aplica el cambio (idempotente)
    uv run python scripts/mejora_2c.py --ver      # solo dice si ya está aplicado

Qué falló (caso S2 de la base): `ClienteLLM.chat` solo comprobaba «restante > 0» ANTES de cada
llamada. Con 50 k por delante, el programador pidió 32 768 tokens de salida, volvió vacío por
longitud, se reintentó con 65 536 y juntas gastaron ~77 k: el consumo llegó a 520 683 de 500 000, la
reserva del redactor desapareció y S2 se quedó sin informe.md (2/12). El freno existía, pero no
protegía la reserva.

Qué cambia (solver/cliente_llm.py y solver/config.py), detrás de SOLVER_FRENO_SALIDA=1:
  1. el tope de salida de cada llamada = lo que queda − lo que costará el prompt (≈ caracteres/3);
  2. si ese tope es menor que SOLVER_SALIDA_MINIMA (4 096), la llamada NO se hace: decision
     `presupuesto_agotado`, y el orquestador pasa al redactor con la reserva intacta;
  3. si una llamada vuelve vacía por longitud con el tope rebajado por presupuesto, no se reintenta
     (decision `tope_de_presupuesto`): repetirla solo gastaría la reserva.
Sin la variable, el comportamiento es el de v1 (por eso el control y las pruebas no cambian).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CLIENTE = RAIZ / "solver" / "cliente_llm.py"
CONFIG = RAIZ / "solver" / "config.py"
MARCA = "freno_salida"


def leer(p: Path) -> tuple[str, str]:
    crudo = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in crudo else "\n"
    return crudo.replace("\r\n", "\n"), nl


def escribir(p: Path, texto: str, nl: str) -> None:
    p.write_bytes(texto.replace("\n", nl).encode("utf-8"))


def reemplazar(texto: str, viejo: str, nuevo: str, que: str) -> str:
    if texto.count(viejo) != 1:
        sys.exit(f"No encuentro el bloque «{que}» exactamente una vez en el archivo: no toco nada más. "
                 "Pásame ese trozo de código y lo ajusto.")
    return texto.replace(viejo, nuevo)


# --------------------------------------------------------------------------- config.py
CFG_VIEJO = '''    max_intentos: int = field(default_factory=lambda: _entero("SOLVER_MAX_INTENTOS", 3))
'''
CFG_NUEVO = '''    # v2 (2.c): el freno mira lo que CUESTA la llamada. Apagado por omisión = comportamiento de v1.
    freno_salida: bool = field(default_factory=lambda: _entero("SOLVER_FRENO_SALIDA", 0) == 1)
    salida_minima: int = field(default_factory=lambda: _entero("SOLVER_SALIDA_MINIMA", 4_096))
    max_intentos: int = field(default_factory=lambda: _entero("SOLVER_MAX_INTENTOS", 3))
'''

# --------------------------------------------------------------------------- cliente_llm.py
CL_1_VIEJO = '''            with observabilidad.guardrail("presupuesto_tokens", input={"agente": agente}) as g:
                restante = self.disponible(agente)
                g.actualizar(output={"decision": "bloquear" if restante <= 0 else "permitir",
                                     "restante": restante, "consumido": self.consumido,
                                     "presupuesto": self.cfg.presupuesto_tokens})
            if restante <= 0:                         # freno: ANTES de la llamada
                self.traza.decision(agente, "presupuesto_agotado", subtarea=subtarea,
                                    motivo=f"consumido={self.consumido} de "
                                           f"{self.cfg.presupuesto_tokens} (reserva "
                                           f"{self.cfg.reserva_redactor})")
'''
CL_1_NUEVO = '''            with observabilidad.guardrail("presupuesto_tokens", input={"agente": agente}) as g:
                restante = self.disponible(agente)
                techo = restante                      # v1: solo importa que quede algo
                if self.cfg.freno_salida:             # v2: lo que queda menos el prompt (≈ caracteres/3)
                    techo = restante - sum(len(str(m.get("content") or "")) for m in mensajes) // 3
                bloquear = restante <= 0 or (self.cfg.freno_salida and techo < self.cfg.salida_minima)
                g.actualizar(output={"decision": "bloquear" if bloquear else "permitir",
                                     "restante": restante, "techo_salida": techo,
                                     "consumido": self.consumido,
                                     "presupuesto": self.cfg.presupuesto_tokens})
            if bloquear:                              # freno: ANTES de la llamada
                self.traza.decision(agente, "presupuesto_agotado", subtarea=subtarea,
                                    motivo=f"consumido={self.consumido} de "
                                           f"{self.cfg.presupuesto_tokens} (reserva "
                                           f"{self.cfg.reserva_redactor})"
                                           + (f"; queda {techo} para la salida y el mínimo es "
                                              f"{self.cfg.salida_minima}" if restante > 0 else ""))
'''
CL_2_VIEJO = '''            intento += 1
            t0 = time.perf_counter()
'''
CL_2_NUEVO = '''            limite_salida = min(max_tokens, techo) if self.cfg.freno_salida else max_tokens
            intento += 1
            t0 = time.perf_counter()
'''
CL_3_VIEJO = '''                                               metadata={"subtarea": subtarea, "intento": intento,
                                                         "max_tokens": max_tokens, "json": json_mode}) as gen:
                    r = self.llm.chat(mensajes, tools=tools, json_mode=json_mode, max_tokens=max_tokens)
'''
CL_3_NUEVO = '''                                               metadata={"subtarea": subtarea, "intento": intento,
                                                         "max_tokens": limite_salida, "json": json_mode}) as gen:
                    r = self.llm.chat(mensajes, tools=tools, json_mode=json_mode, max_tokens=limite_salida)
'''
CL_4_VIEJO = '''                fin=r.get("fin"), max_tokens=max_tokens,
'''
CL_4_NUEVO = '''                fin=r.get("fin"), max_tokens=limite_salida,
'''
CL_5_VIEJO = '''            if r.get("fin") == "length" and max_tokens >= tope:
'''
CL_5_NUEVO = '''            if self.cfg.freno_salida and r.get("fin") == "length" and limite_salida < max_tokens:
                self.traza.decision(agente, "tope_de_presupuesto", subtarea=subtarea,
                                    motivo=f"vacío por longitud con la salida rebajada a {limite_salida} "
                                           f"(pedía {max_tokens}): repetirlo gastaría la reserva del redactor")
                raise LLMVacio(f"{agente}: contenido vacío con la salida limitada por presupuesto "
                               f"({limite_salida} tokens)")
            if r.get("fin") == "length" and max_tokens >= tope:
'''


def main() -> int:
    cfg, nl_cfg = leer(CONFIG)
    cli, nl_cli = leer(CLIENTE)
    if MARCA in cfg and MARCA in cli:
        print("Ya está aplicado (solver/config.py y solver/cliente_llm.py).")
        return 0
    if "--ver" in sys.argv:
        print("Sin aplicar.")
        return 0
    if MARCA in cfg or MARCA in cli:
        sys.exit("Aplicado a medias (solo uno de los dos archivos). No toco nada: pásame el estado.")
    cfg = reemplazar(cfg, CFG_VIEJO, CFG_NUEVO, "config: max_intentos")
    for i, (v, n) in enumerate([(CL_1_VIEJO, CL_1_NUEVO), (CL_2_VIEJO, CL_2_NUEVO), (CL_3_VIEJO, CL_3_NUEVO),
                                (CL_4_VIEJO, CL_4_NUEVO), (CL_5_VIEJO, CL_5_NUEVO)], 1):
        cli = reemplazar(cli, v, n, f"cliente_llm, bloque {i}")
    escribir(CONFIG, cfg, nl_cfg)
    escribir(CLIENTE, cli, nl_cli)
    print("Aplicado: solver/config.py (freno_salida, salida_minima) y solver/cliente_llm.py (5 bloques).")
    print("Se activa con SOLVER_FRENO_SALIDA=1; sin la variable todo sigue como en v1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())