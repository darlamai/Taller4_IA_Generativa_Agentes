# -*- coding: utf-8 -*-
"""
T3 — Parte 3: análisis por consulta del corpus mínimo (Tarea C, MMIA 6013).

A partir de los rankings de T1 (TF-IDF, línea base léxica) y T2 (LSA con
TruncatedSVD de 4 dimensiones):
  1) identifica, por método, en qué consultas falla cada uno (el documento
     relevante queda fuera del top 3);
  2) calcula, para cada consulta, qué términos comparten la consulta y su
     documento relevante (intersección de tokens tras la tokenización de
     TF-IDF), junto con la evidencia de los competidores del top 3.

Salidas (carpeta actual):
  * resultados.json
  * T3_terminos_compartidos.png
  * T3_puesto_relevante_por_metodo.png
"""

import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (textuales del enunciado)
# ---------------------------------------------------------------------------
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
DOCUMENTOS = {
    "d01": "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "d02": "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "d03": "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "d04": "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "d05": "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "d06": "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "d07": "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "d08": "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "d09": "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "d10": "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
}
QUERY_IDS = ["q1", "q2", "q3", "q4", "q5", "q6"]
CONSULTAS = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

# ---------------------------------------------------------------------------
# 2) Reproducción de TF-IDF y LSA con los parámetros exactos de T1 y T2
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, idéntico a la Parte 1
X_docs = vectorizer.fit_transform([DOCUMENTOS[d] for d in DOC_IDS])
X_cons = vectorizer.transform([CONSULTAS[q][0] for q in QUERY_IDS])
S_tfidf = cosine_similarity(X_cons, X_docs)  # (6 consultas, 10 documentos)

svd = TruncatedSVD(n_components=4, random_state=0)  # idéntico a la Parte 2
L_docs = svd.fit_transform(X_docs)
L_cons = svd.transform(X_cons)
S_lsa = cosine_similarity(L_cons, L_docs)


def ranking_desc(fila):
    """Documentos por similitud descendente; empates resueltos en orden d01..d10."""
    orden = np.argsort(-np.asarray(fila), kind="stable")
    return [DOC_IDS[i] for i in orden]


rankings_locales = {
    nombre: {q: ranking_desc(S[i]) for i, q in enumerate(QUERY_IDS)}
    for nombre, S in (("TF-IDF", S_tfidf), ("LSA", S_lsa))
}
sims_locales = {
    nombre: {
        q: {d: float(S[i, j]) for j, d in enumerate(DOC_IDS)}
        for i, q in enumerate(QUERY_IDS)
    }
    for nombre, S in (("TF-IDF", S_tfidf), ("LSA", S_lsa))
}

# ---------------------------------------------------------------------------
# 3) Rankings oficiales de T1/T2 (si están disponibles y completos)
# ---------------------------------------------------------------------------
def cargar_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extraer_previo(res, clave_lista):
    """{qid: {'ranking': [...], 'sims': {...}}} desde un resultados.json previo; None si no es utilizable."""
    if not isinstance(res, dict):
        return None
    filas = res.get(clave_lista)
    if not isinstance(filas, list) or len(filas) != len(QUERY_IDS):
        return None
    out = {}
    for fila in filas:
        if not isinstance(fila, dict):
            return None
        qid = fila.get("consulta")
        ranking = fila.get("ranking_completo_10_docs")
        if qid not in CONSULTAS or not isinstance(ranking, list) or len(ranking) != len(DOC_IDS):
            return None
        sims = fila.get("similitudes_por_documento")
        out[qid] = {
            "ranking": [str(x) for x in ranking],
            "sims": sims if isinstance(sims, dict) else None,
        }
    return out


raw_t1 = cargar_json("entrada/T1/resultados.json")
raw_t2 = cargar_json("entrada/T2/resultados.json")
prev_t1 = extraer_previo(raw_t1, "por_consulta")
prev_t2 = extraer_previo(raw_t2, "lsa_por_consulta")


def construir_metodo(nombre, previo, rankings_loc, sims_loc, descripcion_local, ruta_previa):
    if previo is not None:
        rankings = {q: list(previo[q]["ranking"]) for q in QUERY_IDS}
        sims = {}
        for q in QUERY_IDS:
            sims_prev = previo[q]["sims"]
            if sims_prev is not None and all(d in sims_prev for d in DOC_IDS):
                sims[q] = {d: float(sims_prev[d]) for d in DOC_IDS}
            else:
                sims[q] = dict(sims_loc[q])
        fuente = f"{ruta_previa} (rankings oficiales de la subtarea previa)"
        coincide = all(rankings[q] == rankings_loc[q] for q in QUERY_IDS)
    else:
        rankings = {q: list(rankings_loc[q]) for q in QUERY_IDS}
        sims = {q: dict(sims_loc[q]) for q in QUERY_IDS}
        fuente = f"recalculado localmente: {descripcion_local} (archivo previo no disponible o incompleto)"
        coincide = None
    ranks = {q: rankings[q].index(CONSULTAS[q][1]) + 1 for q in QUERY_IDS}
    return {
        "metodo": nombre,
        "fuente_rankings": fuente,
        "rankings": rankings,
        "similitudes": sims,
        "rank_del_relevante": ranks,
        "rankings_locales_coinciden_con_previa": coincide,
    }


info_tfidf = construir_metodo(
    "TF-IDF", prev_t1, rankings_locales["TF-IDF"], sims_locales["TF-IDF"],
    "TfidfVectorizer por defecto + similitud coseno (Parte 1)",
    "entrada/T1/resultados.json",
)
info_lsa = construir_metodo(
    "LSA", prev_t2, rankings_locales["LSA"], sims_locales["LSA"],
    "TruncatedSVD(n_components=4, random_state=0) + similitud coseno (Parte 2)",
    "entrada/T2/resultados.json",
)

# ---------------------------------------------------------------------------
# 4) Tokenización de TF-IDF e intersección consulta–documento relevante
# ---------------------------------------------------------------------------
analizador = vectorizer.build_analyzer()
tokens_consulta = {q: set(analizador(CONSULTAS[q][0])) for q in QUERY_IDS}
tokens_doc = {d: set(analizador(DOCUMENTOS[d])) for d in DOC_IDS}
vocabulario = set(vectorizer.vocabulary_.keys())
df_counts = np.asarray((X_docs > 0).sum(axis=0)).ravel()  # nº de docs que contienen cada término


def df_de(termino):
    return int(df_counts[vectorizer.vocabulary_[termino]])


matriz_coincidencias = np.zeros((len(QUERY_IDS), len(DOC_IDS)), dtype=int)
for i, q in enumerate(QUERY_IDS):
    for j, d in enumerate(DOC_IDS):
        matriz_coincidencias[i, j] = len(tokens_consulta[q] & tokens_doc[d])

tabla_compartidos = []
for q in QUERY_IDS:
    texto_q, rel = CONSULTAS[q]
    tq, td = tokens_consulta[q], tokens_doc[rel]
    compartidos = sorted(tq & td)
    fuera_vocab = sorted(tq - vocabulario)
    tabla_compartidos.append({
        "consulta": q,
        "texto_consulta": texto_q,
        "documento_relevante": rel,
        "texto_documento_relevante": DOCUMENTOS[rel],
        "tokens_unicos_consulta": sorted(tq),
        "tokens_unicos_documento_relevante": sorted(td),
        "n_tokens_unicos_consulta": len(tq),
        "n_tokens_unicos_documento_relevante": len(td),
        "terminos_compartidos": compartidos,
        "n_terminos_compartidos": len(compartidos),
        "cobertura_de_la_consulta": (len(compartidos) / len(tq)) if tq else 0.0,
        "tokens_consulta_fuera_de_vocabulario": fuera_vocab,
        "n_tokens_consulta_fuera_de_vocabulario": len(fuera_vocab),
        "tokens_consulta_sin_coincidencia_en_relevante": sorted(tq - td),
        "df_de_terminos_compartidos": {t: df_de(t) for t in compartidos},
    })

idf = vectorizer.idf_
detalle_terminos = []
for fila in tabla_compartidos:
    q, rel = fila["consulta"], fila["documento_relevante"]
    i, j = QUERY_IDS.index(q), DOC_IDS.index(rel)
    for t in fila["terminos_compartidos"]:
        col = vectorizer.vocabulary_[t]
        detalle_terminos.append({
            "consulta": q,
            "documento_relevante": rel,
            "termino": t,
            "df_en_corpus": int(df_counts[col]),
            "idf": float(idf[col]),
            "peso_tfidf_en_consulta": float(X_cons[i, col]),
            "peso_tfidf_en_documento_relevante": float(X_docs[j, col]),
        })

# ---------------------------------------------------------------------------
# 5) Consultas falladas por método (relevante fuera del top 3)
# ---------------------------------------------------------------------------
def analizar_fallos(info):
    ranks, rankings, sims = info["rank_del_relevante"], info["rankings"], info["similitudes"]
    fallos = [q for q in QUERY_IDS if ranks[q] > 3]
    detalle = []
    for q in fallos:
        rel = CONSULTAS[q][1]
        inter = tokens_consulta[q] & tokens_doc[rel]
        detalle.append({
            "consulta": q,
            "texto_consulta": CONSULTAS[q][0],
            "documento_relevante": rel,
            "rank_del_relevante": ranks[q],
            "top_3": rankings[q][:3],
            "similitud_del_relevante": sims[q][rel],
            "terminos_compartidos_con_relevante": sorted(inter),
            "n_terminos_compartidos_con_relevante": len(inter),
            "df_de_terminos_compartidos": {t: df_de(t) for t in inter},
            "tokens_consulta_fuera_de_vocabulario": sorted(tokens_consulta[q] - vocabulario),
        })
    return {
        "metodo": info["metodo"],
        "consultas_falladas": fallos,
        "n_consultas_falladas": len(fallos),
        "consultas_con_relevante_en_top3": [q for q in QUERY_IDS if ranks[q] <= 3],
        "Hit@1": float(np.mean([1.0 if ranks[q] == 1 else 0.0 for q in QUERY_IDS])),
        "Hit@3": float(np.mean([1.0 if ranks[q] <= 3 else 0.0 for q in QUERY_IDS])),
        "MRR": float(np.mean([1.0 / ranks[q] for q in QUERY_IDS])),
        "detalle_por_fallo": detalle,
    }


fallos_tfidf = analizar_fallos(info_tfidf)
fallos_lsa = analizar_fallos(info_lsa)

# ---------------------------------------------------------------------------
# 6) Evidencia de los fallos: competidores del top 3 y su solape con la consulta
# ---------------------------------------------------------------------------
evidencia = {}
for info in (info_tfidf, info_lsa):
    metodo = info["metodo"]
    for q in [q for q in QUERY_IDS if info["rank_del_relevante"][q] > 3]:
        rel = CONSULTAS[q][1]
        ranking = info["rankings"][q]
        ev = evidencia.setdefault(q, {
            "consulta": q,
            "texto_consulta": CONSULTAS[q][0],
            "documento_relevante": rel,
            "terminos_compartidos_con_relevante": sorted(tokens_consulta[q] & tokens_doc[rel]),
            "terminos_compartidos_con_cada_documento": {
                d: sorted(tokens_consulta[q] & tokens_doc[d]) for d in DOC_IDS
            },
            "por_metodo": {},
        })
        competidores = []
        for pos, d in enumerate(ranking[:3]):
            competidores.append({
                "posicion": pos + 1,
                "documento": d,
                "similitud": info["similitudes"][q][d],
                "terminos_compartidos_con_consulta": sorted(tokens_consulta[q] & tokens_doc[d]),
                "n_terminos_compartidos_con_consulta": len(tokens_consulta[q] & tokens_doc[d]),
            })
        ev["por_metodo"][metodo] = {
            "rank_del_relevante": info["rank_del_relevante"][q],
            "similitud_del_relevante": info["similitudes"][q][rel],
            "top_3": ranking[:3],
            "competidores_del_top3": competidores,
        }

# ---------------------------------------------------------------------------
# 7) Explicación (generada a partir de los resultados)
# ---------------------------------------------------------------------------
frases = []
for info, fallos in ((info_tfidf, fallos_tfidf), (info_lsa, fallos_lsa)):
    metodo = info["metodo"]
    if not fallos["detalle_por_fallo"]:
        frases.append(f"{metodo}: no falla ninguna consulta; el relevante queda dentro del top 3 en las seis consultas.")
    for det in fallos["detalle_por_fallo"]:
        q = det["consulta"]
        ev = evidencia[q]["por_metodo"][metodo]
        df_txt = ", ".join(
            f"'{t}' (df={d}/10)" for t, d in sorted(det["df_de_terminos_compartidos"].items())
        ) or "ninguno"
        comp_txt = "; ".join(
            "{} (puesto {}, similitud {:.4f}) comparte {} término(s) [{}]".format(
                c["documento"], c["posicion"], c["similitud"],
                c["n_terminos_compartidos_con_consulta"],
                ", ".join(c["terminos_compartidos_con_consulta"]) or "ninguno",
            )
            for c in ev["competidores_del_top3"]
        )
        oov = det["tokens_consulta_fuera_de_vocabulario"]
        oov_txt = (
            f" Además, {len(oov)} de los {len(tokens_consulta[q])} tokens de la consulta no aparecen en ningún "
            f"documento del corpus ({', '.join(oov)}), con lo que no pueden emparejar con nada."
            if oov else ""
        )
        frases.append(
            f"{metodo} falla en {q}: el relevante {det['documento_relevante']} queda en el puesto "
            f"{det['rank_del_relevante']} (similitud {det['similitud_del_relevante']:.4f}) y la intersección de "
            f"tokens con la consulta se reduce a {det['n_terminos_compartidos_con_relevante']} término(s): {df_txt}."
            f"{oov_txt} Los documentos que lo anteceden comparten más vocabulario con la consulta: {comp_txt}."
        )

orden_solape = sorted(tabla_compartidos, key=lambda f: f["n_terminos_compartidos"])
min_f, max_f = orden_solape[0], orden_solape[-1]
fallos_conj = set(fallos_tfidf["consultas_falladas"]) | set(fallos_lsa["consultas_falladas"])
if fallos_conj and fallos_conj.issubset({min_f["consulta"]}):
    cierre = (
        "El fallo se explica porque el solape se reduce a términos muy frecuentes en el corpus "
        "(df alto, idf bajo), que apenas discriminan, y a tokens de la consulta ausentes del vocabulario."
    )
else:
    cierre = (
        "La relación entre solape y fallo puede leerse consulta a consulta en la tabla de términos "
        "compartidos y en la evidencia de cada fallo."
    )
frases.append(
    f"Solape consulta–relevante en el corpus: mínimo {min_f['n_terminos_compartidos']} término(s) "
    f"({min_f['consulta']} vs {min_f['documento_relevante']}: "
    f"{', '.join(min_f['terminos_compartidos']) or 'ninguno'}; "
    f"{min_f['n_tokens_consulta_fuera_de_vocabulario']} de sus {min_f['n_tokens_unicos_consulta']} tokens de "
    f"consulta son ajenos al vocabulario) y máximo {max_f['n_terminos_compartidos']} "
    f"({max_f['consulta']} vs {max_f['documento_relevante']}). {cierre}"
)

# ---------------------------------------------------------------------------
# 8) Verificación contra las subtareas previas
# ---------------------------------------------------------------------------
def _coincide(recap, previa, clave):
    if not isinstance(previa, dict) or clave not in previa:
        return None
    try:
        return bool(abs(float(recap) - float(previa[clave])) < 1e-9)
    except Exception:
        return None


m_t1 = raw_t1.get("metricas_agregadas") if isinstance(raw_t1, dict) else None
m_t2 = raw_t2.get("metricas_lsa") if isinstance(raw_t2, dict) else None
tam_vocab_t1 = raw_t1.get("tam_vocabulario_tfidf") if isinstance(raw_t1, dict) else None

verificacion = {
    "T1": {
        "archivo": "entrada/T1/resultados.json",
        "disponible_y_completo": prev_t1 is not None,
        "rankings_locales_coinciden_con_T1": info_tfidf["rankings_locales_coinciden_con_previa"],
        "Hit@3_recapitulado": fallos_tfidf["Hit@3"],
        "Hit@3_coincide_con_T1": _coincide(fallos_tfidf["Hit@3"], m_t1, "Hit@3"),
        "MRR_recapitulado": fallos_tfidf["MRR"],
        "MRR_coincide_con_T1": _coincide(fallos_tfidf["MRR"], m_t1, "MRR"),
        "ranks_del_relevante_usados": info_tfidf["rank_del_relevante"],
        "ranks_del_relevante_recalculados_localmente": {
            q: rankings_locales["TF-IDF"][q].index(CONSULTAS[q][1]) + 1 for q in QUERY_IDS
        },
    },
    "T2": {
        "archivo": "entrada/T2/resultados.json",
        "disponible_y_completo": prev_t2 is not None,
        "rankings_locales_coinciden_con_T2": info_lsa["rankings_locales_coinciden_con_previa"],
        "Hit@3_recapitulado": fallos_lsa["Hit@3"],
        "Hit@3_coincide_con_T2": _coincide(fallos_lsa["Hit@3"], m_t2, "Hit@3"),
        "MRR_recapitulado": fallos_lsa["MRR"],
        "MRR_coincide_con_T2": _coincide(fallos_lsa["MRR"], m_t2, "MRR"),
        "ranks_del_relevante_usados": info_lsa["rank_del_relevante"],
        "ranks_del_relevante_recalculados_localmente": {
            q: rankings_locales["LSA"][q].index(CONSULTAS[q][1]) + 1 for q in QUERY_IDS
        },
    },
}

# ---------------------------------------------------------------------------
# 9) Figuras
# ---------------------------------------------------------------------------
vmax = max(1, int(matriz_coincidencias.max()))

fig, ax = plt.subplots(figsize=(9.5, 5))
im = ax.imshow(matriz_coincidencias, cmap="Blues", vmin=0, vmax=vmax)
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS, fontsize=9)
ax.set_yticks(range(len(QUERY_IDS)))
ax.set_yticklabels([f"{q} (rel: {CONSULTAS[q][1]})" for q in QUERY_IDS], fontsize=9)
for i in range(len(QUERY_IDS)):
    for j in range(len(DOC_IDS)):
        val = int(matriz_coincidencias[i, j])
        ax.text(j, i, str(val), ha="center", va="center", fontsize=9,
                color="white" if val > 0.6 * vmax else "black")
for i, q in enumerate(QUERY_IDS):
    j = DOC_IDS.index(CONSULTAS[q][1])
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="crimson", linewidth=2))
ax.set_xticks(np.arange(-0.5, len(DOC_IDS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(QUERY_IDS), 1), minor=True)
ax.grid(which="minor", color="lightgray", linewidth=0.4)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("Documentos del corpus")
ax.set_ylabel("Consultas (con su documento relevante)")
ax.set_title("T3 · Términos compartidos consulta–documento\n"
             "(intersección de tokens de la tokenización TF-IDF; cuadro rojo = relevante)")
fig.colorbar(im, ax=ax, label="número de tokens compartidos")
fig.tight_layout()
fig.savefig("T3_terminos_compartidos.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(QUERY_IDS))
ancho = 0.38
r_tf = [info_tfidf["rank_del_relevante"][q] for q in QUERY_IDS]
r_lsa = [info_lsa["rank_del_relevante"][q] for q in QUERY_IDS]
ax.bar(x - ancho / 2, r_tf, ancho, edgecolor="black", linewidth=0.4,
       color=["crimson" if r > 3 else "steelblue" for r in r_tf])
ax.bar(x + ancho / 2, r_lsa, ancho, edgecolor="black", linewidth=0.4,
       color=["crimson" if r > 3 else "seagreen" for r in r_lsa])
for xi, (a, b) in enumerate(zip(r_tf, r_lsa)):
    ax.text(xi - ancho / 2, a + 0.12, str(a), ha="center", fontsize=8)
    ax.text(xi + ancho / 2, b + 0.12, str(b), ha="center", fontsize=8)
ax.axhline(3, color="black", linestyle="--", linewidth=1)
ax.text(len(QUERY_IDS) - 0.55, 3.18, "umbral del top 3", ha="right", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(QUERY_IDS)
ax.set_yticks(range(0, 11))
ax.set_ylim(0, 10.8)
ax.set_ylabel("puesto (rank) del documento relevante")
ax.set_xlabel("Consultas")
ax.set_title("T3 · Puesto del documento relevante por consulta y método\n"
             "(rojo = fallo: relevante fuera del top 3)")
ax.legend(handles=[
    Patch(facecolor="steelblue", edgecolor="black", linewidth=0.4, label="TF-IDF (T1) · en top 3"),
    Patch(facecolor="seagreen", edgecolor="black", linewidth=0.4, label="LSA (T2) · en top 3"),
    Patch(facecolor="crimson", edgecolor="black", linewidth=0.4, label="fallo: relevante fuera del top 3"),
], fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("T3_puesto_relevante_por_metodo.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 10) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": (
        "Parte 3, análisis por consulta: consultas falladas por método (documento relevante fuera "
        "del top 3) y términos compartidos entre cada consulta y su documento relevante (intersección "
        "de tokens tras la tokenización de TF-IDF), como evidencia que explica los fallos."
    ),
    "criterio_de_fallo": "el documento relevante queda fuera del top 3 (rank > 3) en el ranking del método",
    "configuracion": {
        "tokenizacion_para_la_interseccion": (
            "build_analyzer() de TfidfVectorizer por defecto: minúsculas, token_pattern "
            "r'(?u)\\b\\w\\w+\\b' (tokens de 2+ caracteres), sin stop words ni stemming; "
            "intersección de conjuntos de tokens únicos"
        ),
        "tfidf": "TfidfVectorizer() por defecto, ajustado con los 10 documentos (idéntico a T1/T2)",
        "lsa": "TruncatedSVD(n_components=4, random_state=0), ajustado con los 10 documentos (idéntico a T2)",
        "n_documentos": 10,
        "n_consultas": 6,
        "tam_vocabulario_tfidf": len(vocabulario),
        "tam_vocabulario_reportado_en_T1": tam_vocab_t1,
    },
    "fuente_de_los_rankings": {
        "TF-IDF": info_tfidf["fuente_rankings"],
        "LSA": info_lsa["fuente_rankings"],
    },
    "verificacion_con_subtareas_previas": verificacion,
    "ranks_del_relevante_por_metodo": {
        "TF-IDF": info_tfidf["rank_del_relevante"],
        "LSA": info_lsa["rank_del_relevante"],
    },
    "top3_por_consulta_por_metodo": {
        "TF-IDF": {q: info_tfidf["rankings"][q][:3] for q in QUERY_IDS},
        "LSA": {q: info_lsa["rankings"][q][:3] for q in QUERY_IDS},
    },
    "metricas_recapituladas_desde_los_ranks": {
        "TF-IDF": {"Hit@1": fallos_tfidf["Hit@1"], "Hit@3": fallos_tfidf["Hit@3"], "MRR": fallos_tfidf["MRR"]},
        "LSA": {"Hit@1": fallos_lsa["Hit@1"], "Hit@3": fallos_lsa["Hit@3"], "MRR": fallos_lsa["MRR"]},
    },
    "fallos_por_metodo": {"TF-IDF": fallos_tfidf, "LSA": fallos_lsa},
    "tabla_terminos_compartidos": tabla_compartidos,
    "detalle_pesos_de_los_terminos_compartidos": detalle_terminos,
    "matriz_n_terminos_compartidos": {
        "filas_consultas": QUERY_IDS,
        "columnas_documentos": DOC_IDS,
        "nota": "cada celda es |tokens(consulta) ∩ tokens(documento)| con la tokenización de TF-IDF",
        "valores": matriz_coincidencias.tolist(),
    },
    "evidencia_de_los_fallos": evidencia,
    "rankings_y_similitudes_usados": {
        nombre: {
            q: {
                "ranking": info["rankings"][q],
                "similitudes_por_documento": info["similitudes"][q],
                "rank_del_relevante": info["rank_del_relevante"][q],
            }
            for q in QUERY_IDS
        }
        for nombre, info in (("TF-IDF", info_tfidf), ("LSA", info_lsa))
    },
    "explicacion": frases,
    "notas_metodologicas": [
        "Intersección calculada sobre tokens únicos tras el análisis de TfidfVectorizer por defecto: "
        "minúsculas, patrón r'(?u)\\b\\w\\w+\\b' (tokens de 2+ caracteres), sin stop words ni stemming; "
        "por eso 'que' y 'qué' son tokens distintos y variantes morfológicas como 'añaden'/'añade', "
        "'recuperados'/'recuperación' o 'texto'/'textos' no cuentan como compartidos.",
        "Los tokens de la consulta ausentes del vocabulario (ajustado solo con los 10 documentos) no "
        "aportan nada a la similitud TF-IDF ni a la proyección LSA; se listan por consulta.",
        "df = número de documentos del corpus que contienen el término; un término compartido con df "
        "alto (p. ej. 'la') tiene idf bajo y apenas discrimina.",
        "Los rankings usados son los oficiales de T1/T2 cuando sus archivos están disponibles y "
        "completos; en caso contrario se recalcularon localmente con los mismos parámetros "
        "(ver 'verificacion_con_subtareas_previas').",
    ],
    "figuras": ["T3_terminos_compartidos.png", "T3_puesto_relevante_por_metodo.png"],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 11) Resumen
# ---------------------------------------------------------------------------
print("=" * 78)
print("T3 · Análisis por consulta — fallos por método y términos compartidos")
print("=" * 78)
print(f"Rankings TF-IDF: {info_tfidf['fuente_rankings']}")
print(f"Rankings LSA   : {info_lsa['fuente_rankings']}")
if info_tfidf["rankings_locales_coinciden_con_previa"] is not None:
    print(f"Recálculo local coincide con T1: {info_tfidf['rankings_locales_coinciden_con_previa']} | "
          f"con T2: {info_lsa['rankings_locales_coinciden_con_previa']}")
for nombre, info in (("T1/TF-IDF", info_tfidf), ("T2/LSA", info_lsa)):
    if info["rankings_locales_coinciden_con_previa"] is False:
        print(f"AVISO: el recálculo local no coincide con los rankings oficiales de {nombre}; se usan los oficiales.")
for metodo, fallos in (("TF-IDF", fallos_tfidf), ("LSA", fallos_lsa)):
    lista = ", ".join(fallos["consultas_falladas"]) if fallos["consultas_falladas"] else "ninguna"
    print(f"{metodo}: Hit@1={fallos['Hit@1']:.4f}  Hit@3={fallos['Hit@3']:.4f}  MRR={fallos['MRR']:.4f}  "
          f"consultas falladas (relevante fuera del top 3): {lista}")
print("Términos compartidos consulta -> documento relevante (intersección de tokens TF-IDF):")
for fila in tabla_compartidos:
    terminos_txt = ", ".join(fila["terminos_compartidos"]) if fila["terminos_compartidos"] else "—"
    extra = (f" | {fila['n_tokens_consulta_fuera_de_vocabulario']} tokens de consulta fuera de vocabulario"
             if fila["n_tokens_consulta_fuera_de_vocabulario"] else "")
    print(f"  {fila['consulta']} -> {fila['documento_relevante']}: "
          f"{fila['n_terminos_compartidos']} compartidos [{terminos_txt}]{extra}")
print("Escritos: resultados.json | T3_terminos_compartidos.png | T3_puesto_relevante_por_metodo.png")
