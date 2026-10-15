"""Taller 04, Parte 3 — guardrails de texto del solver (opción B).

Dos guardrails de texto y el sandbox (que ya vive en solver/sandbox/guarda.py):

  1. INYECCIÓN en la entrada. En la opción B la entrada es el enunciado que saca el lector del PDF.
     Compara sobre texto NORMALIZADO (minúsculas, sin tildes, sin signos, sin caracteres invisibles)
     y por INTENCIÓN, no por frase exacta: un verbo que anula + algo que se anula («ignora … las
     instrucciones»), un verbo que extrae + algo interno («traduce … tus reglas internas»), un cambio
     de rol («actúa como si no tuvieras restricciones») o el secuestro de la tarea (dos de tres
     señales: va dirigida al sistema, manda no hacer la tarea, manda escribir una frase fija).
     Una negación delante («no ignores las instrucciones») la desactiva.
  2. DATOS PERSONALES en la entrada, la salida y la traza: correo, teléfono de Ecuador, cédula/RUC
     (solo con su palabra clave delante: un número de diez dígitos suelto es un número normal),
     tarjeta (Luhn) y nombre (solo si lo anuncia una etiqueta: «Solicitado por: …»). Un nombre sin
     etiqueta se escapa: haría falta un modelo de entidades (Presidio u otro); aquí no se usa.
  3. SECRETOS en la salida, por su FORMA (sk-…, AKIA…, ghp_…, JWT, «clave = valor»), no por su largo:
     un hash de 32 caracteres o un UUID pasan.

Contrato del Lab (`probar_guardrails.py`): f(texto) -> (ok, texto_resultante).
  ok=False                      -> BLOQUEAR (no llega al modelo / al usuario)
  ok=True y texto distinto      -> REDACTAR
  ok=True y el mismo texto      -> PASAR

`validar_entrada` bloquea cuando el mensaje ES el ataque (queda menos del 30 % de texto legítimo) y
redacta —retira solo la frase— cuando el ataque va metido en un documento largo. El solver nunca
bloquea el enunciado entero: usa `sanear_documento`, que retira la frase, avisa y sigue.

Este módulo no importa nada del solver: lo usan el orquestador, la traza y el banco de pruebas.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field

PROPORCION_MINIMA = 0.30     # menos texto legítimo que esto: el mensaje ES el ataque -> bloquear

# =========================================================================== normalización
_INVISIBLES = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_HOMOGLIFOS = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "і": "i",
                             "ѕ": "s", "ј": "j", "ԁ": "d", "ɡ": "g", "ո": "n", "ѵ": "v"})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni signos, sin caracteres invisibles, espacios colapsados."""
    t = unicodedata.normalize("NFKD", _INVISIBLES.sub("", texto)).lower().translate(_HOMOGLIFOS)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    # «i g n o r a  l a s» (letras sueltas; dos espacios separan palabras): se unen ANTES de colapsar espacios
    t = re.sub(r"(?<![a-z0-9])(?:[a-z] ){3,}[a-z](?![a-z0-9])", lambda m: m.group(0).replace(" ", ""), t)
    return re.sub(r"\s+", " ", t).strip()


def _desofuscar(texto: str) -> str:
    """Segunda pasada: «i g n o r a» -> «ignora»; «1gn0ra» -> «ignora» (solo en palabras con letras)."""
    t = re.sub(r"\b(?:\w )+\w\b", lambda m: m.group(0).replace(" ", ""), texto)

    def palabra(m: re.Match) -> str:
        w = m.group(0)
        return w.translate(_LEET) if re.search(r"[a-z]", w) and re.search(r"\d|@|\$", w) else w
    return re.sub(r"[\w@$]+", palabra, t)


# =========================================================================== 1. inyección
_V_ANULA = (r"(?:ignor\w*|olvid\w*|descart\w*|omit\w*|desobedec\w*|salt\w*|anul\w*|desoy\w*|"
            r"deja\w* de (?:seguir|obedecer|cumplir|respetar)|disregard\w*|forget\w*|ignore\w*|override\w*|"
            r"bypass\w*|skip\w*|discard\w*|"
            r"no (?:tengas|tomes|hagas|des) (?:en cuenta|caso a|caso de|bola a)|no (?:sigas|obedezcas|cumplas|respetes))")
_O_ANULA = (r"(?:instruccion\w*|prompt|lo anterior|todo lo anterior|lo de arriba|everything above|"
            r"(?:previous|prior|above|earlier) (?:instructions?|rules|guidelines|prompts?)|"
            r"instructions?|guidelines|"
            r"(?:las |tus |todas las )?(?:reglas|instrucciones|ordenes) que (?:te |se te )?(?:dieron|dije|di|dimos|dio|han dado|recibiste)|"
            r"todo (?:lo )?que (?:te |se te )?(?:pidieron|pidio|dijeron|dije|dieron|dio|han dicho)|"
            r"(?:tus|tu|todas tus|todas las|las) (?:reglas|restricciones|limites|filtros|politicas|directrices)"
            r"(?: (?:anteriores|previas|de seguridad|internas|del sistema|de arriba))?|"
            r"(?:reglas|restricciones|limites|filtros|politicas|directrices) "
            r"(?:anteriores|previas|de seguridad|internas|del sistema|de arriba))")
_ANULAR = re.compile(rf"\b{_V_ANULA}\b (?:\w+ ){{0,6}}?\b{_O_ANULA}\b")

_V_EXTRAE = (r"(?:revel\w*|muestr\w*|imprim\w*|repit\w*|traduc\w*|copi\w*|dame|dime|entreg\w*|"
             r"comparte\w*|filtr\w*|escrib\w*|cita\w*|expon\w*|exhib\w*|print\w*|reveal\w*|show\w*|"
             r"repeat\w*|translat\w*|leak\w*|output\w*|display\w*|tell me|give me|send me|envi\w*)")
_O_INTERNO = (r"(?:system prompt|prompt del sistema|prompt inicial|prompt oculto|"
              r"instrucciones (?:internas|del sistema|iniciales|ocultas|secretas)|"
              r"(?:tus|tu|your) (?:instrucciones|prompt|directrices|configuracion|instructions)|"
              r"(?:las |tus )?instrucciones que (?:recibiste|te (?:dieron|dije|han dado))|"
              r"(?:tus |tu |your )?(?:reglas|rules|directrices) (?:internas|ocultas|secretas|del sistema|internal)|"
              r"api key|apikey|clave (?:de la |del |de )?api|clave secreta|llave secreta|credenciales|"
              r"contrasena|password|secret key|variables de entorno|environment variables)")
_EXTRAER = re.compile(rf"\b{_V_EXTRAE}\b (?:\w+ ){{0,7}}?\b{_O_INTERNO}\b")

_ROL = [
    re.compile(r"\b(?:actua|comportate|finge|simula|act|behave|pretend)\w* (?:como|as|like)(?: si)? "
               r"(?:no (?:tuvieras|tienes|existieran|hubiera|hay)|you (?:have|had) no|without|sin)"
               r"(?: \w+){0,3} (?:restriccion\w*|regla\w*|limite\w*|filtro\w*|restrictions|rules|limits|filters)\b"),
    re.compile(r"\b(?:desde ahora|a partir de ahora|de ahora en adelante|ahora) (?:eres|seras|ya no eres) "
               r"(?:\w+ ){0,5}?(?:sin (?:reglas|restricciones|limites|filtros|etica)|libre de (?:reglas|restricciones))"),
    re.compile(r"\b(?:eres|seras) (?:un|una) (?:\w+ ){0,3}sin (?:reglas|restricciones|limites|filtros|etica)\b"),
    re.compile(r"\byou are now (?:a |an |the )?(?:\w+ ){0,3}?(?:unrestricted|jailbroken|dan|evil|uncensored|without)\b"),
    re.compile(r"\b(?:activa|enable|enter|entra en|modo)\w* (?:el )?(?:modo )?(?:desarrollador|developer mode|dios|god mode)\b"),
    re.compile(r"\b(?:responde|contesta|respond|answer)\w* sin (?:filtros|restricciones|limites|censura|reglas)\b"),
    re.compile(r"\bjailbreak\b"),
]

# secuestro de la tarea: hacen falta DOS de las tres señales en la misma frase
_DIRIGIDA = re.compile(r"\b(?:nota|mensaje|aviso|instruccion\w*|atencion|importante|note|notice)"
                       r"(?: \w+){0,2} (?:para|dirigid\w* a|to) (?:quien|el|la|los|las|todo\w*|any|whoever)? ?"
                       r"(?:procese|procesa|lea|lee|analice|resuma|asistente|modelo|ia|sistema|agente|llm|solver|"
                       r"revisor|automat\w*|process\w*|ai|assistant|model)\b"
                       r"|\bsi eres (?:un |una )?(?:ia|modelo|asistente|llm|agente|bot)\b"
                       r"|\bif you are (?:an? )?(?:ai|llm|assistant|language model|bot)\b"
                       r"|\b(?:asistente|assistant|ia|ai|modelo|llm|solver)s?\b(?: \w+){0,2} "
                       r"(?:debe|debes|deben|should|must|tiene que|tienes que)\b")
_NO_HACER = re.compile(r"\bno (?:calcules|resuelvas|hagas|realices|ejecutes|respondas|escribas|analices|corras)"
                       r"(?: \w+){0,2} (?:nada|la tarea|el ejercicio|ningun\w*|codigo|lo que (?:se )?pide)\b"
                       r"|\b(?:sin|en lugar de|en vez de|instead of) (?:calcular|resolver|hacer|ejecutar|analizar)"
                       r"(?: \w+){0,3} (?:nada|la tarea|el ejercicio|cualquier\w*)\b"
                       r"|\bdo not (?:calculate|solve|run|execute|analy[sz]e)(?: \w+){0,2} (?:anything|the task|any code)\b")
_FRASE_FIJA = re.compile(r"\b(?:escrib|respond|devuelv|imprim|report|entreg|pon|write|reply|output|return|print)\w*"
                         r"(?: \w+){0,4} (?:unicamente|solamente|solo|exclusivamente|nada mas que|only|just)"
                         r"(?: \w+){0,2} (?:la frase|el texto|la palabra|la cadena|esta frase|lo siguiente|"
                         r"the phrase|the text|the word|the string|this phrase)\b")


@dataclass
class Hallazgo:
    intencion: str                 # anular_instrucciones | extraer_interno | cambio_de_rol | secuestro_de_tarea
    fragmento: str                 # la frase original (recortada)
    senales: list[str] = field(default_factory=list)
    inicio: int = 0                # posición de la frase en el texto original
    fin: int = 0


_NEGADORES = re.compile(r"\b(?:no|nunca|jamas|tampoco|never|do not|dont|don t)\s+(?:\w+\s+){0,1}$")


def _negado(norm: str, inicio: int) -> bool:
    return bool(_NEGADORES.search(norm[max(0, inicio - 28):inicio]))


def _intenciones(norm: str) -> list[tuple[str, list[str]]]:
    """Intenciones que hay en UNA frase ya normalizada (y su evidencia)."""
    hallazgos: list[tuple[str, list[str]]] = []
    for m in _ANULAR.finditer(norm):
        if not _negado(norm, m.start()):
            hallazgos.append(("anular_instrucciones", [m.group(0)]))
            break
    for m in _EXTRAER.finditer(norm):
        if not _negado(norm, m.start()):
            hallazgos.append(("extraer_interno", [m.group(0)]))
            break
    for p in _ROL:
        m = p.search(norm)
        if m and not _negado(norm, m.start()):
            hallazgos.append(("cambio_de_rol", [m.group(0)]))
            break
    señales = [(n, p.search(norm)) for n, p in (("dirigida_al_sistema", _DIRIGIDA),
                                                ("manda_no_hacer_la_tarea", _NO_HACER),
                                                ("manda_escribir_una_frase", _FRASE_FIJA))]
    señales = [(n, m) for n, m in señales if m and not _negado(norm, m.start())]
    if len(señales) >= 2:
        hallazgos.append(("secuestro_de_tarea", [f"{n}: {m.group(0)}" for n, m in señales]))
    return hallazgos


_FRASES = re.compile(r".+?(?:(?<=[.!?»”])\s+|\n\s*\n|$)", re.S)


def _frases(texto: str):
    for m in _FRASES.finditer(texto):
        if m.group(0).strip():
            yield m.start(), m.end(), m.group(0)


def clasificar_inyeccion(texto: str) -> list[Hallazgo]:
    """Frases del texto que dan una orden al sistema en vez de describir una tarea."""
    salida: list[Hallazgo] = []
    for ini, fin, frase in _frases(texto):
        norm = normalizar(frase)
        vistas = _intenciones(norm)
        if not vistas:                                   # segunda pasada: ofuscación (i g n o r a, 1gn0ra)
            vistas = _intenciones(_desofuscar(norm))
        for intencion, señales in vistas:
            salida.append(Hallazgo(intencion, " ".join(frase.split())[:200], señales, ini, fin))
            break                                         # una intención por frase basta para retirarla
    return salida


def retirar_inyecciones(texto: str, marca: str = "[FRASE RETIRADA POR GUARDRAIL: instrucción dirigida al sistema]"
                        ) -> tuple[str, list[Hallazgo]]:
    hallazgos = clasificar_inyeccion(texto)
    for h in sorted(hallazgos, key=lambda x: x.inicio, reverse=True):
        trozo = texto[h.inicio:h.fin]
        cola = trozo[len(trozo.rstrip()):] or " "          # conserva el salto de línea (los «## Parte N» siguen en su línea)
        texto = texto[:h.inicio] + marca + cola + texto[h.fin:]
    return texto, hallazgos


# =========================================================================== 2. datos personales
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Ecuador: móvil 09x xxx xxxx, +593 / 00593 y cualquier +código internacional con espacios o guiones.
_TELEFONO = re.compile(r"(?<![\w.,/-])(?:(?:\+|00)593[\s-]?\(?0?\)?[\s-]?\d(?:[\s-]?\d){7,8}"
                       r"|09\d[\s-]?\d{3}[\s-]?\d{4}"
                       r"|\+\d{1,3}[\s-]\d{1,4}(?:[\s-]\d{2,4}){2,3})(?![\d])")
# Cédula / RUC: solo con la palabra clave delante. «Mi cédula es 1712345678», «C.I.: 1712345678».
_CEDULA = re.compile(r"(?i)\b(c[eé]dula(?: de (?:identidad|ciudadan[ií]a))?|c\.\s?i\.?|identificaci[oó]n|ruc|documento)"
                     r"((?:\s+(?:es|n[°º.o]*|n[uú]mero|:|#|-))*\s*[:#-]?\s*)(?P<num>\d(?:[ .-]?\d){9}(?:[ .-]?\d{3})?)(?!\d)")
_TARJETA = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_NOMBRE_PROPIO = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:de |del |de la |y )?[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}"
_ETIQUETA_NOMBRE = re.compile(r"(?i:\b(solicitado por|solicitante|a nombre de|dirigido a|contacto|firmado por|"
                              r"nombre completo|nombre|cliente|me llamo|mi nombre es)\b)(\*{0,2}\s*(?:es|:|-|—)?\s*\*{0,2}\s*(?:(?:Sr|Sra|Srta|Dr|Dra|Ing|Lic|Mgs|Msc|Prof)\.?\s+)?)"
                              rf"({_NOMBRE_PROPIO})")
_SENSIBLES: set[str] = set()


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


def contar_pii(texto: str) -> dict[str, int]:
    """Cuántos datos de cada tipo hay (nunca devuelve los valores: el conteo sí puede ir a un span)."""
    n = {"correo": len(_EMAIL.findall(texto)), "telefono": len(_TELEFONO.findall(texto)),
         "cedula": len(_CEDULA.findall(texto)), "nombre": len(_ETIQUETA_NOMBRE.findall(texto)),
         "tarjeta": sum(1 for m in _TARJETA.finditer(texto) if _tarjeta(m) == "<TARJETA>")}
    return {k: v for k, v in n.items() if v}


def extraer_pii(texto: str) -> list[str]:
    """Los valores literales (para que la traza los reconozca aunque aparezcan sin etiqueta)."""
    valores = _EMAIL.findall(texto) + [m.group(0) for m in _TELEFONO.finditer(texto)]
    valores += [m.group("num") for m in _CEDULA.finditer(texto)]
    valores += [m.group(3).strip() for m in _ETIQUETA_NOMBRE.finditer(texto)]
    return sorted({v for v in valores if len(v) >= 6}, key=len, reverse=True)


def registrar_sensibles(valores: list[str]) -> None:
    _SENSIBLES.update(v for v in valores if len(v) >= 6)


def reiniciar_sensibles() -> None:
    _SENSIBLES.clear()


def redactar_pii(texto: str) -> str:
    for v in sorted(_SENSIBLES, key=len, reverse=True):          # lo que esta corrida ya sabe que es personal
        texto = re.sub(re.escape(v).replace(r"\ ", r"\s+"), "<DATO-PERSONAL>", texto, flags=re.I)
    texto = _EMAIL.sub("<CORREO>", texto)
    texto = _CEDULA.sub(lambda m: f"{m.group(1)}{m.group(2)}<CEDULA>", texto)
    texto = _TELEFONO.sub("<TELEFONO>", texto)
    texto = _ETIQUETA_NOMBRE.sub(lambda m: f"{m.group(1)}{m.group(2)}<NOMBRE>", texto)
    return _TARJETA.sub(_tarjeta, texto)


# =========================================================================== 3. secretos por su forma
_SECRETOS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)\b(?:contrase[ñn]a|password|passwd)\b[^.\n]{0,25}?(?:\bes\b|\bis\b|[:=])\s*(?=\S*[\d!@#$%^&*])\S{6,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|contrasena|clave)\b\s*[=:]\s*['\"]?[^\s'\",;]{8,}"),
]


def hay_secreto(texto: str) -> bool:
    return any(p.search(texto) for p in _SECRETOS)


def retirar_secretos(texto: str) -> str:
    for p in _SECRETOS:
        texto = p.sub("[SECRETO RETIRADO]", texto)
    return texto


# =========================================================================== contrato del Lab
def validar_entrada(texto: str) -> tuple[bool, str]:
    """Bloquea si el mensaje ES un ataque; retira la frase si va dentro de algo legítimo; redacta PII."""
    hallazgos = clasificar_inyeccion(texto)
    if hallazgos:
        limpio, _ = retirar_inyecciones(texto)
        resto = re.sub(r"\[FRASE RETIRADA[^\]]*\]", "", limpio)
        if len(resto.strip()) < PROPORCION_MINIMA * len(texto.strip()):
            return False, "Solicitud rechazada: intento de dar órdenes al sistema (" + hallazgos[0].intencion + ")."
        return True, redactar_pii(limpio)
    return True, redactar_pii(texto)


def validar_salida(texto: str) -> tuple[bool, str]:
    """Bloquea una credencial por su forma; redacta datos personales; deja pasar lo demás."""
    if hay_secreto(texto):
        return False, "Salida bloqueada: contiene algo con forma de credencial."
    return True, redactar_pii(texto)


# =========================================================================== lo que usa el solver
def activos() -> bool:
    """SOLVER_GUARDRAILS=0 los apaga (para medir el solver con y sin ellos)."""
    return os.environ.get("SOLVER_GUARDRAILS", "1") != "0"


def _recorrer(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _recorrer(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_recorrer(v, fn) for v in obj]
    return obj


def sanear_inyeccion(doc: dict) -> tuple[dict, dict]:
    """Guardrail 1 (entrada): retira del texto que sacó el lector las frases que dan órdenes al sistema.

    Devuelve (documento limpio, informe). El informe lleva la frase retirada (es un ataque, no un dato)."""
    t0 = time.perf_counter()
    vistos: dict[str, Hallazgo] = {}

    def limpiar(texto: str) -> str:
        limpio, hallazgos = retirar_inyecciones(texto)
        for h in hallazgos:
            vistos.setdefault(h.fragmento, h)
        return limpio

    nuevo = _recorrer(doc, limpiar)
    informe = {"retiradas": len(vistos),
               "hallazgos": [{"intencion": h.intencion, "senales": h.senales, "fragmento": h.fragmento[:200]}
                             for h in vistos.values()],
               "latencia_ms": round((time.perf_counter() - t0) * 1000, 2)}
    return nuevo, informe


def sanear_pii(doc: dict) -> tuple[dict, dict]:
    """Guardrail 2 (entrada): redacta los datos personales del documento y los REGISTRA, para que la
    traza los reconozca después aunque aparezcan sin etiqueta. El informe lleva solo CONTEOS."""
    t0 = time.perf_counter()
    textos: list[str] = []
    _recorrer(doc, lambda s: textos.append(s) or s)
    registrar_sensibles([v for t in textos for v in extraer_pii(t)])
    conteo = contar_pii(doc["texto"]) if isinstance(doc.get("texto"), str) else {}
    if not conteo:
        for t in textos:
            for k, v in contar_pii(t).items():
                conteo[k] = conteo.get(k, 0) + v
    nuevo = _recorrer(doc, redactar_pii)
    return nuevo, {"pii": conteo, "total": sum(conteo.values()),
                   "latencia_ms": round((time.perf_counter() - t0) * 1000, 2)}


def sanear_salida(texto: str) -> tuple[str, dict]:
    """Guardrail de salida: retira secretos por su forma y redacta datos personales."""
    t0 = time.perf_counter()
    secretos = hay_secreto(texto)
    pii = contar_pii(texto)
    limpio = redactar_pii(retirar_secretos(texto)) if (secretos or pii or _SENSIBLES) else texto
    return limpio, {"secretos": secretos, "pii": pii, "cambio": limpio != texto,
                    "latencia_ms": round((time.perf_counter() - t0) * 1000, 2)}
