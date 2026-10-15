"""Taller 04, Parte 1: observabilidad con Langfuse. Todo lo que toca Langfuse vive aquí.

Sin claves, sin el paquete o con OBSERVABILIDAD=0 los spans son no-ops: el solver corre igual.
La duración se mide al cerrar el `with` (== finally); la raíz envuelve a los guardrails;
la máscara se aplica antes de enviar nada y recorre dict, list y tuple.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import re
import sys
from typing import Any, Callable, Iterator

SESION_POR_DEFECTO = "grupo-07-base"
ETIQUETA = "taller-04"
VERSION_PROMPTS_POR_DEFECTO = "v1"
_MAX_TEXTO = 4_000

# ------------------------------------------------------------------ PII
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_TELEFONO = re.compile(r"(?<![\w.])(?:(?:\+|00)593[\s-]?\(?0?\)?[\s-]?\d(?:[\s-]?\d){7,8}"
                       r"|09\d[\s-]?\d{3}[\s-]?\d{4})(?!\d)")
_CEDULA = re.compile(r"(?i)\b(c[eé]dula|c\.?i\.?|identificaci[oó]n|ruc)\b(\s*(?:n[°ºo.]*\s*)?[:#]?\s*)\d{10}(?:\d{3})?(?!\d)")
_TARJETA = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")


def _luhn(digitos: str) -> bool:
    suma, doble = 0, False
    for c in reversed(digitos):
        d = int(c)
        if doble:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        suma += d
        doble = not doble
    return suma % 10 == 0


def _tarjeta(m: re.Match) -> str:
    solo = re.sub(r"\D", "", m.group(0))
    return "<TARJETA>" if 13 <= len(solo) <= 19 and _luhn(solo) else m.group(0)

def enmascarar_pii(texto: str) -> str:
    """Un solo detector: el de solver/guardrails.py."""
    from solver import guardrails
    return guardrails.redactar_pii(texto)


def _limpiar_texto(texto: str) -> str:
    from solver.traza import enmascarar as secretos     # claves de API, bearer, valores del entorno
    return enmascarar_pii(secretos(texto))


def mask(*, data: Any, **_: Any) -> Any:
    """Langfuse(mask=...): corre en esta máquina ANTES de enviar. Recorre str, dict, list Y tuple."""
    if isinstance(data, str):
        return _limpiar_texto(data)
    if isinstance(data, dict):
        return {k: mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [mask(data=v) for v in data]
    return data


# ------------------------------------------------------------------ cliente
_estado: dict[str, Any] = {"listo": False, "cliente": None}


def _aviso(texto: str) -> None:
    print(f"[observabilidad] {texto}", file=sys.stderr)


def cliente():
    """El cliente de Langfuse (con la máscara) o None si está apagado. Se crea una vez."""
    if _estado["listo"]:
        return _estado["cliente"]
    _estado["listo"] = True
    if os.environ.get("OBSERVABILIDAD", "1") == "0":
        return None
    faltan = [v for v in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
              if not os.environ.get(v)]
    if faltan:
        _aviso(f"Langfuse DESACTIVADO: falta {', '.join(faltan)} en el entorno/.env")
        return None
    try:
        from langfuse import Langfuse, get_client
        Langfuse(mask=mask)
        lf = get_client()
        if not lf.auth_check():
            _aviso("Langfuse DESACTIVADO: auth_check() falló (¿servidor caído o claves de otro proyecto?)")
            return None
    except Exception as err:
        _aviso(f"Langfuse DESACTIVADO: {type(err).__name__}: {err}")
        return None
    _estado["cliente"] = lf
    return lf


def activo() -> bool:
    return cliente() is not None


def vaciar() -> None:
    """flush(): un script corto termina antes de que el envío en segundo plano acabe."""
    lf = _estado["cliente"]
    if lf is not None:
        try:
            lf.flush()
        except Exception as err:
            _aviso(f"flush falló: {type(err).__name__}: {err}")


def version_prompts() -> str:
    return os.environ.get("SOLVER_PROMPT_VERSION", VERSION_PROMPTS_POR_DEFECTO)


def sesion() -> str:
    return os.environ.get("SOLVER_SESION", SESION_POR_DEFECTO)


# ------------------------------------------------------------------ spans
def _recortar(valor: Any, limite: int = _MAX_TEXTO) -> Any:
    if isinstance(valor, str):
        return valor if len(valor) <= limite else valor[:limite] + f"… [+{len(valor) - limite} car.]"
    if isinstance(valor, dict):
        return {k: _recortar(v, limite) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_recortar(v, limite) for v in valor[:40]]
    return valor


class Manejador:
    """Lo que recibe el cuerpo del `with`: actualizar el span sin preocuparse de si existe."""

    def __init__(self, obs: Any = None):
        self._obs = obs

    def actualizar(self, **campos: Any) -> None:
        if self._obs is None:
            return
        for clave in ("input", "output", "metadata"):
            if clave in campos:
                campos[clave] = _recortar(campos[clave])
        try:
            self._obs.update(**campos)
        except Exception as err:
            _aviso(f"update falló: {type(err).__name__}: {err}")

    def error(self, err: BaseException | str) -> None:
        texto = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
        self.actualizar(level="ERROR", status_message=texto[:500])


@contextlib.contextmanager
def _span(nombre: str, tipo: str, **campos: Any) -> Iterator[Manejador]:
    """Abre un span y lo cierra SIEMPRE. Si el cuerpo lanza, queda en nivel ERROR y la excepción sigue."""
    lf = cliente()
    if lf is None:
        yield Manejador()
        return
    for clave in ("input", "output", "metadata"):
        if clave in campos:
            campos[clave] = _recortar(campos[clave])
    try:
        cm = lf.start_as_current_observation(name=nombre, as_type=tipo, **campos)
        obs = cm.__enter__()
    except Exception as err:
        _aviso(f"no pude abrir el span {nombre!r}: {type(err).__name__}: {err}")
        yield Manejador()
        return
    h = Manejador(obs)
    try:
        yield h
    except BaseException as err:
        h.error(err)
        with contextlib.suppress(Exception):
            cm.__exit__(type(err), err, err.__traceback__)
        raise
    else:
        with contextlib.suppress(Exception):
            cm.__exit__(None, None, None)


@contextlib.contextmanager
def corrida(pregunta: str, *, variante: str = "completo", metadata: dict | None = None) -> Iterator[Manejador]:
    """La raíz. Fija sesión, etiqueta y prompt_version en TODOS los spans de la corrida."""
    lf = cliente()
    if lf is None:
        yield Manejador()
        return
    meta = {"prompt_version": version_prompts(), "variante": variante,
            **{k: str(v) for k, v in (metadata or {}).items()}}
    try:
        from langfuse import propagate_attributes
        prop = propagate_attributes(session_id=sesion(), tags=[ETIQUETA], metadata=meta)
    except Exception as err:
        _aviso(f"propagate_attributes falló: {type(err).__name__}: {err}")
        prop = contextlib.nullcontext()
    with prop, _span("corrida", "span", input={"question": pregunta}, metadata=meta) as h:
        yield h


def agente(nombre: str, **campos: Any):
    return _span(nombre, "agent", **campos)


def herramienta(nombre: str, **campos: Any):
    return _span(nombre, "tool", **campos)


def guardrail(nombre: str, **campos: Any):
    return _span(nombre, "guardrail", **campos)


def generacion(nombre: str, *, modelo: str, **campos: Any):
    return _span(nombre, "generation", model=modelo, **campos)


def usar_uso(h: Manejador, uso: dict | None, **extra: Any) -> None:
    """Tokens en la forma de Langfuse. completion_tokens YA incluye el razonamiento: no se manda
    aparte como output_reasoning_tokens (el exportador los suma y se contarían dos veces)."""
    uso = uso or {}
    h.actualizar(usage_details={"input": int(uso.get("prompt_tokens", 0) or 0),
                                "output": int(uso.get("completion_tokens", 0) or 0)}, **extra)


def con_contexto(fn: Callable) -> Callable:
    """Para ThreadPoolExecutor: los hilos no heredan el contexto y sus spans saldrían huérfanos."""
    base = contextvars.copy_context()

    def envuelto(*args: Any, **kw: Any):
        return base.copy().run(fn, *args, **kw)
    return envuelto


def traza_id_actual() -> str | None:
    lf = _estado["cliente"]
    if lf is None:
        return None
    try:
        return lf.get_current_trace_id()
    except Exception:
        return None