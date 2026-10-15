"""Parte 3 — conecta los guardrails de solver/guardrails.py al solver.

    uv run python scripts/aplicar_parte3.py            # aplica el cambio (idempotente)
    uv run python scripts/aplicar_parte3.py --ver      # solo dice qué falta

Qué cambia (cada guardrail abre su span `guardrail`; SOLVER_GUARDRAILS=0 los apaga para comparar):
  solver/orquestador.py  `_leer`      -> inyeccion_en_pdf y pii_en_entrada, sobre el texto que sacó el lector
                         `_redactar`  -> salida_pii_secretos, sobre el informe antes de escribirlo
  solver/traza.py        `enmascarar` -> además de las claves, redacta datos personales (arregla P107:
                         la traza local conservaba el nombre que sí se enmascaraba en Langfuse)
  solver/observabilidad.py `enmascarar_pii` -> delega en solver/guardrails.py (un solo detector, no dos)
El sandbox (`guarda_sandbox`, solver/agentes/ejecutor.py) YA era un guardrail con su span: no se toca.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORQ = RAIZ / "solver" / "orquestador.py"
TRAZA = RAIZ / "solver" / "traza.py"
OBS = RAIZ / "solver" / "observabilidad.py"


def leer(p: Path) -> tuple[str, str]:
    crudo = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in crudo else "\n"
    return crudo.replace("\r\n", "\n"), nl


def escribir(p: Path, texto: str, nl: str) -> None:
    p.write_bytes(texto.replace("\n", nl).encode("utf-8"))


def reemplazar(texto: str, viejo: str, nuevo: str, que: str) -> str:
    if texto.count(viejo) != 1:
        sys.exit(f"No encuentro el bloque «{que}» exactamente una vez: no toco nada más. "
                 "Pásame ese trozo de código y lo ajusto.")
    return texto.replace(viejo, nuevo)


# --------------------------------------------------------------------------- orquestador.py
ORQ_IMPORT_V = "from solver import formatos, graphrag, observabilidad\n"
ORQ_IMPORT_N = "from solver import formatos, graphrag, guardrails, observabilidad\n"

ORQ_LEER_V = '''        ruta = _guardar(self._salida(estado) / INTERNO / "enunciado_leido.json", doc)
        trabajo = [s["clave"] for s in doc["secciones"] if s["trabajo"]]
'''
ORQ_LEER_N = '''        doc = self._sanear_entrada(doc)                  # Parte 3: guardrails 1 y 2, ANTES de que lo vea un agente
        ruta = _guardar(self._salida(estado) / INTERNO / "enunciado_leido.json", doc)
        trabajo = [s["clave"] for s in doc["secciones"] if s["trabajo"]]
'''

ORQ_METODOS_V = '''    def _indexar(self, estado: Estado) -> dict:
'''
ORQ_METODOS_N = '''    def _sanear_entrada(self, doc: dict) -> dict:
        """Parte 3, guardrails 1 y 2: lo que extrae el lector es texto NO confiable (un PDF puede traer
        órdenes dirigidas al modelo) y puede traer datos personales. Se limpia antes de indexarlo,
        planificar o escribirlo en disco. Los spans llevan decisiones y CONTEOS, nunca el dato."""
        if not guardrails.activos():
            return doc
        guardrails.reiniciar_sensibles()
        with observabilidad.guardrail("inyeccion_en_pdf",
                                      input={"caracteres": len(str(doc.get("texto", "")))}) as g:
            doc, inj = guardrails.sanear_inyeccion(doc)
            g.actualizar(output={"decision": "redactar" if inj["retiradas"] else "permitir", **inj})
        with observabilidad.guardrail("pii_en_entrada",
                                      input={"caracteres": len(str(doc.get("texto", "")))}) as g:
            doc, pii = guardrails.sanear_pii(doc)
            g.actualizar(output={"decision": "redactar" if pii["total"] else "permitir", **pii})
        self.traza.decision("guardrail", "entrada_saneada",
                            f"{inj['retiradas']} frase(s) retirada(s); datos personales: {pii['pii'] or 'ninguno'}")
        return doc

    def _sanear_salida(self, texto):
        """Parte 3, guardrail de salida: el informe no sale con credenciales ni datos personales."""
        if not guardrails.activos() or not isinstance(texto, str):
            return texto
        with observabilidad.guardrail("salida_pii_secretos", input={"caracteres": len(texto)}) as g:
            limpio, inf = guardrails.sanear_salida(texto)
            g.actualizar(output={"decision": "redactar" if inf["cambio"] else "permitir", **inf})
        if inf["cambio"]:
            self.traza.decision("guardrail", "salida_saneada",
                                f"secretos={inf['secretos']}; datos personales: {inf['pii'] or 'ninguno'}")
        return limpio

    def _indexar(self, estado: Estado) -> dict:
'''

ORQ_RED_V = '''                                  estado.get("borrador"), self._notas(estado))
        destino = self._construir(estado, doc, plan, aprobadas, texto)
'''
ORQ_RED_N = '''                                  estado.get("borrador"), self._notas(estado))
        texto = self._sanear_salida(texto)               # Parte 3: guardrail de salida
        destino = self._construir(estado, doc, plan, aprobadas, texto)
'''

# --------------------------------------------------------------------------- traza.py
TRAZA_V = '''            texto = patron.sub("***", texto)
    return texto
'''
TRAZA_N = '''            texto = patron.sub("***", texto)
    from solver import guardrails                          # Parte 3: la traza local tampoco guarda datos personales
    return guardrails.redactar_pii(texto) if guardrails.activos() else texto
'''

# --------------------------------------------------------------------------- observabilidad.py
OBS_V = '''def enmascarar_pii(texto: str) -> str:
    texto = _EMAIL.sub("<CORREO>", texto)
    texto = _CEDULA.sub(lambda m: f"{m.group(1)}{m.group(2)}<CEDULA>", texto)
    texto = _TELEFONO.sub("<TELEFONO>", texto)
    return _TARJETA.sub(_tarjeta, texto)
'''
OBS_N = '''def enmascarar_pii(texto: str) -> str:
    """Un solo detector: el de solver/guardrails.py (el mismo que mide medir_guardas.py)."""
    from solver import guardrails
    return guardrails.redactar_pii(texto)
'''


def main() -> None:
    solo_ver = "--ver" in sys.argv
    plan = [
        (ORQ, [(ORQ_IMPORT_V, ORQ_IMPORT_N, "import de observabilidad", "guardrails, observabilidad"),
               (ORQ_LEER_V, ORQ_LEER_N, "_leer", "self._sanear_entrada(doc)"),
               (ORQ_METODOS_V, ORQ_METODOS_N, "def _indexar", "def _sanear_entrada"),
               (ORQ_RED_V, ORQ_RED_N, "_redactar", "self._sanear_salida(texto)")]),
        (TRAZA, [(TRAZA_V, TRAZA_N, "enmascarar() de traza.py", "guardrails.redactar_pii")]),
        (OBS, [(OBS_V, OBS_N, "enmascarar_pii() de observabilidad.py", "Un solo detector")]),
    ]
    faltan = 0
    for ruta, cambios in plan:
        texto, nl = leer(ruta)
        for viejo, nuevo, que, marca in cambios:
            if marca in texto:
                print(f"  ya está : {ruta.name}: {que}")
                continue
            faltan += 1
            print(f"  falta   : {ruta.name}: {que}")
            if not solo_ver:
                texto = reemplazar(texto, viejo, nuevo, que)
        if not solo_ver:
            escribir(ruta, texto, nl)
    if not (RAIZ / "solver" / "guardrails.py").exists():
        sys.exit("Falta solver/guardrails.py: cópialo primero.")
    print("✓ aplicado" if not solo_ver and faltan else "✓ nada que aplicar" if not faltan else f"faltan {faltan} cambio(s)")


if __name__ == "__main__":
    main()
