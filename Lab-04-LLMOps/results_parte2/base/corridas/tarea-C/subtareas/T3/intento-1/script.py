#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Análisis por consulta (Tarea C, MMIA 6013).

A partir de los rankings de ambos métodos (TF-IDF de T1 y LSA k=4 de T2):
  1) identifica las consultas donde cada método falla: el documento relevante
     queda fuera del top 3 (posición >= 4);
  2) extrae los términos que comparte cada consulta con su documento relevante
     y con los documentos que la superan en el ranking, para sustentar la
     explicación de los fallos en la Parte 3 del reporte.

Salidas (carpeta actual):
  resultados.json
  T3_heatmap_terminos_compartidos.png
  T3_rank_del_relevante_por_metodo.png
  T3_tabla_fallos.csv
  T3_tabla_terminos_compartidos.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 3  # criterio del enunciado: fallo si el relevante queda fuera del top 3

# ---------------------------------------------------------------------------
# 1) Corpus y consultas (idénticos al enunciado; sin cambios respecto a T1/T2)
# ---------------------------------------------------------------------------
DOCS = {
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
DOC_IDS = sorted(DOCS)

QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
QIDS = sorted(QUERIES)

# ---------------------------------------------------------------------------
# 2) Reproducción exacta de T1 (TF-IDF + coseno) y T2 (LSA k=4 + coseno)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que en T1/T2
X_docs = vectorizer.fit_transform([DOCS[d] for d in DOC_IDS])
X_queries = vectorizer.transform([QUERIES[q][0] for q in QIDS])

S_tfidf = cosine_similarity(X_queries, X_docs)

svd = TruncatedSVD(n_components=4, random_state=0)
D_lsa = svd.fit_transform(X_docs)
Q_lsa = svd.transform(X_queries)
S_lsa = cosine_similarity(Q_lsa, D_lsa)


def ranking_estable(fila):
    """Orden descendente por similitud; desempate estable por id (d01..d10)."""
    return [DOC_IDS[i] for i in np.argsort(-np.asarray(fila).ravel(), kind="stable")]


rank_recomp = {
    "tfidf": {q: ranking_estable(S_tfidf[i]) for i, q in enumerate(QIDS)},
    "lsa": {q: ranking_estable(S_lsa[i]) for i, q in enumerate(QIDS)},
}

# ---------------------------------------------------------------------------
# 3) Rankings oficiales: resultados previos (entrada/T1, entrada/T2) si están
#    disponibles y completos; en su defecto, el recómputo con los mismos
#    parámetros. El recómputo siempre se contrasta contra lo previo.
# ---------------------------------------------------------------------------
def _cargar_json(ruta):
    try:
        with open(ruta, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def extraer_rankings(data, clave_bloque):
    try:
        bloque = data[clave_bloque]
        out = {}
        for q in QIDS:
            rk = bloque[q]["ranking_completo"]
            docs_orden = [e["doc"] for e in rk]
            if len(docs_orden) != len(DOC_IDS) or set(docs_orden) != set(DOC_IDS):
                return None
            out[q] = docs_orden
        return out
    except Exception:
        return None


def extraer_sims(data, clave_bloque, campo_sim):
    try:
        bloque = data[clave_bloque]
        out = {}
        for q in QIDS:
            rk = bloque[q]["ranking_completo"]
            if len(rk) != len(DOC_IDS):
                return None
            out[q] = {e["doc"]: float(e[campo_sim]) for e in rk}
        return out
    except Exception:
        return None


t1_json = _cargar_json(Path("entrada/T1/resultados.json"))
t2_json = _cargar_json(Path("entrada/T2/resultados.json"))

prev_rank = {
    "tfidf": extraer_rankings(t1_json, "metricas_por_consulta"),
    "lsa": extraer_rankings(t2_json, "metricas_por_consulta_lsa"),
}
prev_sims = {
    "tfidf": extraer_sims(t1_json, "metricas_por_consulta", "similitud_coseno"),
    "lsa": extraer_sims(t2_json, "metricas_por_consulta_lsa", "similitud_coseno_lsa"),
}

RANKS = {m: (prev_rank[m] if prev_rank[m] is not None else rank_recomp[m]) for m in ("tfidf", "lsa")}

SIMS = {}
for m in ("tfidf", "lsa"):
    S = S_tfidf if m == "tfidf" else S_lsa
    base = {q: {DOC_IDS[j]: float(S[i, j]) for j in range(len(DOC_IDS))} for i, q in enumerate(QIDS)}
    if prev_sims[m] is not None:
        for q in QIDS:
            base[q].update(prev_sims[m][q])  # similitudes oficiales de T1/T2 con prioridad
    SIMS[m] = base

verif = {
    "fuente_rankings_tfidf": "entrada/T1/resultados.json" if prev_rank["tfidf"] is not None
    else "recomputado con los parámetros de T1",
    "fuente_rankings_lsa": "entrada/T2/resultados.json" if prev_rank["lsa"] is not None
    else "recomputado con los parámetros de T2",
    "recomputo_tfidf_coincide_con_T1": (prev_rank["tfidf"] == rank_recomp["tfidf"]) if prev_rank["tfidf"] is not None else None,
    "recomputo_lsa_coincide_con_T2": (prev_rank["lsa"] == rank_recomp["lsa"]) if prev_rank["lsa"] is not None else None,
}
if verif["recomputo_tfidf_coincide_con_T1"] is False:
    print("AVISO: el recómputo TF-IDF difiere de entrada/T1; se usan los rankings oficiales de T1.")
if verif["recomputo_lsa_coincide_con_T2"] is False:
    print("AVISO: el recómputo LSA difiere de entrada/T2; se usan los rankings oficiales de T2.")

# ---------------------------------------------------------------------------
# 4) Términos compartidos (misma tokenización que usa el ranking:
#    analyzer por defecto de TfidfVectorizer)
# ---------------------------------------------------------------------------
analizar = vectorizer.build_analyzer()
tokens_docs = {d: set(analizar(DOCS[d])) for d in DOC_IDS}
tokens_q_set = {q: set(analizar(QUERIES[q][0])) for q in QIDS}
vocab = vectorizer.vocabulary_
idf_vec = vectorizer.idf_
df_termino = {t: sum(1 for d in DOC_IDS if t in tokens_docs[d]) for t in vocab}


def info_termino(t):
    en_v = t in vocab
    return {
        "termino": t,
        "en_vocabulario": en_v,
        "df": int(df_termino[t]) if en_v else 0,
        "idf": float(idf_vec[vocab[t]]) if en_v else None,
    }


def compartidos(q, d):
    """Términos del vocabulario compartidos entre la consulta q y el documento d
    (todo término que aparece en un documento está en el vocabulario), ordenados
    por idf ascendente: los menos discriminativos primero."""
    inter = tokens_q_set[q] & tokens_docs[d]
    terms = [info_termino(t) for t in inter]
    terms.sort(key=lambda x: (x["idf"] if x["idf"] is not None else 99.0, x["termino"]))
    return terms


# ---------------------------------------------------------------------------
# 5) Detección de fallos (relevante fuera del top 3) por método
# ---------------------------------------------------------------------------
rank_relevante = {m: {} for m in ("tfidf", "lsa")}
fallos = {m: [] for m in ("tfidf", "lsa")}
for m in ("tfidf", "lsa"):
    for q in QIDS:
        rk = RANKS[m][q]
        rel = QUERIES[q][1]
        pos = rk.index(rel) + 1
        rank_relevante[m][q] = pos
        if pos > TOP_K:
            comp_rel = compartidos(q, rel)
            superiores = []
            for p, d in enumerate(rk[: pos - 1], start=1):
                terms = compartidos(q, d)
                superiores.append({
                    "pos": p,
                    "doc": d,
                    "similitud": SIMS[m][q][d],
                    "n_terminos_compartidos_con_consulta": len(terms),
                    "terminos_compartidos_con_consulta": [t["termino"] for t in terms],
                    "detalle_terminos": terms,
                })
            fallos[m].append({
                "consulta": q,
                "texto_consulta": QUERIES[q][0],
                "documento_relevante": rel,
                "rank_del_relevante": pos,
                "similitud_del_relevante": SIMS[m][q][rel],
                "terminos_compartidos_con_relevante": [t["termino"] for t in comp_rel],
                "detalle_terminos_con_relevante": comp_rel,
                "terminos_consulta_fuera_de_vocabulario": sorted(t for t in tokens_q_set[q] if t not in vocab),
                "documentos_que_superan_al_relevante": superiores,
                "ranking_completo_de_la_consulta": [
                    {"pos": p + 1, "doc": d, "similitud": SIMS[m][q][d]} for p, d in enumerate(rk)
                ],
            })

# ---------------------------------------------------------------------------
# 6) Tabla de términos compartidos consulta–documento relevante (6 consultas)
# ---------------------------------------------------------------------------
tabla_terminos = []
for q in QIDS:
    rel = QUERIES[q][1]
    comp = compartidos(q, rel)
    en_otros = {}
    for t in sorted(tokens_q_set[q]):
        if t in vocab and t not in tokens_docs[rel]:
            en_otros[t] = [d for d in DOC_IDS if t in tokens_docs[d]]
    fuera = sorted(t for t in tokens_q_set[q] if t not in vocab)
    tabla_terminos.append({
        "consulta": q,
        "texto_consulta": QUERIES[q][0],
        "documento_relevante": rel,
        "rank_tfidf": rank_relevante["tfidf"][q],
        "rank_lsa": rank_relevante["lsa"][q],
        "fallo_tfidf": rank_relevante["tfidf"][q] > TOP_K,
        "fallo_lsa": rank_relevante["lsa"][q] > TOP_K,
        "similitud_tfidf_con_relevante": SIMS["tfidf"][q][rel],
        "similitud_lsa_con_relevante": SIMS["lsa"][q][rel],
        "n_terminos_compartidos": len(comp),
        "terminos_compartidos_con_relevante": [t["termino"] for t in comp],
        "detalle_terminos_compartidos": comp,
        "terminos_de_la_consulta_presentes_en_otros_documentos": en_otros,
        "terminos_de_la_consulta_fuera_de_vocabulario": fuera,
    })

# ---------------------------------------------------------------------------
# 7) Insumos textuales para la explicación de la Parte 3 en el reporte
# ---------------------------------------------------------------------------
def explicar(metodo, f):
    partes = [
        f"[{metodo.upper()}] {f['consulta']}: fallo — el relevante {f['documento_relevante']} quedó en la "
        f"posición {f['rank_del_relevante']} (fuera del top 3)."
    ]
    det = f["detalle_terminos_con_relevante"]
    if det:
        partes.append(
            "La consulta comparte con " + f["documento_relevante"] + " únicamente: "
            + ", ".join(f"{t['termino']} (df={t['df']}/10, idf={t['idf']:.3f})" for t in det)
            + " — cobertura léxica escasa y de baja discriminación."
        )
    else:
        partes.append(f"La consulta no comparte ningún término del vocabulario con {f['documento_relevante']}.")
    if f["terminos_consulta_fuera_de_vocabulario"]:
        partes.append(
            "Términos de la consulta fuera del vocabulario (no aportan al ranking): "
            + ", ".join(f["terminos_consulta_fuera_de_vocabulario"]) + "."
        )
    sup = f["documentos_que_superan_al_relevante"]
    if sup:
        partes.append(
            "Documentos que la superan y sus términos compartidos con la consulta: "
            + "; ".join(
                f"{s['doc']} (pos {s['pos']}, sim={s['similitud']:.3f}): "
                + (", ".join(s["terminos_compartidos_con_consulta"]) if s["terminos_compartidos_con_consulta"] else "sin solape léxico")
                for s in sup
            ) + "."
        )
    if metodo == "lsa":
        sin_solape = [s["doc"] for s in sup if s["n_terminos_compartidos_con_consulta"] == 0]
        if sin_solape:
            partes.append(
                "En LSA, " + ", ".join(sin_solape) + " supera a la consulta sin compartir ningún término: "
                "el espacio latente (k=4) propaga similitud por co-ocurrencia global, no por solape exacto."
            )
    return " ".join(partes)


insumos = {}
for m in ("tfidf", "lsa"):
    for f in fallos[m]:
        insumos[f"{m}_{f['consulta']}"] = explicar(m, f)
for q in QIDS:
    rt, rl = rank_relevante["tfidf"][q], rank_relevante["lsa"][q]
    if rl > rt and rl <= TOP_K:
        insumos[f"degradacion_sin_fallo_{q}_en_lsa"] = (
            f"[LSA] {q}: el relevante {QUERIES[q][1]} baja de la posición {rt} (TF-IDF) a la {rl} (LSA); "
            "sigue dentro del top 3, así que no cuenta como fallo con el criterio del enunciado."
        )
    elif rl < rt:
        insumos[f"mejora_{q}_en_lsa"] = (
            f"[LSA] {q}: el relevante {QUERIES[q][1]} sube de la posición {rt} (TF-IDF) a la {rl} (LSA)."
        )

# ---------------------------------------------------------------------------
# 8) Figuras
# ---------------------------------------------------------------------------
M = np.zeros((len(QIDS), len(DOC_IDS)), dtype=int)
for i, q in enumerate(QIDS):
    for j, d in enumerate(DOC_IDS):
        M[i, j] = len(tokens_q_set[q] & tokens_docs[d])

fig, ax = plt.subplots(figsize=(9.5, 4.8))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(1, int(M.max())))
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS)
ax.set_yticks(range(len(QIDS)))
ax.set_yticklabels(QIDS)
for i in range(len(QIDS)):
    for j in range(len(DOC_IDS)):
        ax.text(j, i, str(int(M[i, j])), ha="center", va="center", fontsize=9,
                color="white" if M[i, j] > 0.6 * M.max() else "black")
for i, q in enumerate(QIDS):
    j = DOC_IDS.index(QUERIES[q][1])
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="crimson", lw=2))
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
ax.set_title("T3 — Número de términos compartidos consulta–documento\n(marco rojo = documento relevante)")
fig.colorbar(im, ax=ax, label="# términos compartidos (vocabulario TF-IDF)")
fig.tight_layout()
plt.savefig("T3_heatmap_terminos_compartidos.png", dpi=120)
plt.close(fig)

r_t = [rank_relevante["tfidf"][q] for q in QIDS]
r_l = [rank_relevante["lsa"][q] for q in QIDS]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
x = np.arange(len(QIDS))
w = 0.38
ax.bar(x - w / 2, r_t, w, label="TF-IDF + coseno (T1)", color="#4C72B0")
ax.bar(x + w / 2, r_l, w, label="LSA k=4 + coseno (T2)", color="#DD8452")
ax.axhline(TOP_K + 0.5, ls="--", color="crimson", lw=1.2)
ax.text(len(QIDS) - 0.5, TOP_K + 0.62, "límite top-3", color="crimson", fontsize=8, ha="right")
for xi in range(len(QIDS)):
    ax.text(xi - w / 2, r_t[xi] + 0.12, str(r_t[xi]), ha="center", fontsize=8)
    ax.text(xi + w / 2, r_l[xi] + 0.12, str(r_l[xi]), ha="center", fontsize=8)
    if r_t[xi] > TOP_K:
        ax.text(xi - w / 2, r_t[xi] + 0.55, "fallo", ha="center", fontsize=7, color="crimson")
    if r_l[xi] > TOP_K:
        ax.text(xi + w / 2, r_l[xi] + 0.55, "fallo", ha="center", fontsize=7, color="crimson")
ax.set_xticks(x)
ax.set_xticklabels(QIDS)
ax.set_ylim(0, max(10.8, max(r_t + r_l) + 1.2))
ax.set_ylabel("Posición del documento relevante (1 = mejor)")
ax.set_title("T3 — Posición del relevante por consulta y método (fallo si queda bajo la línea)")
ax.legend(loc="lower right")
fig.tight_layout()
plt.savefig("T3_rank_del_relevante_por_metodo.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 9) Tablas CSV de apoyo para el reporte
# ---------------------------------------------------------------------------
cols_f = ["metodo", "consulta", "texto_consulta", "documento_relevante", "rank_del_relevante",
          "terminos_compartidos_con_relevante", "documentos_que_superan"]
filas_f = []
for m in ("tfidf", "lsa"):
    for f in fallos[m]:
        filas_f.append({
            "metodo": m,
            "consulta": f["consulta"],
            "texto_consulta": f["texto_consulta"],
            "documento_relevante": f["documento_relevante"],
            "rank_del_relevante": f["rank_del_relevante"],
            "terminos_compartidos_con_relevante": ", ".join(f["terminos_compartidos_con_relevante"]) or "(ninguno)",
            "documentos_que_superan": "; ".join(
                f"{s['pos']}·{s['doc']}[{', '.join(s['terminos_compartidos_con_consulta']) or '—'}]"
                for s in f["documentos_que_superan_al_relevante"]
            ),
        })
pd.DataFrame(filas_f, columns=cols_f).to_csv("T3_tabla_fallos.csv", index=False, encoding="utf-8-sig")

filas_t = []
for row in tabla_terminos:
    filas_t.append({
        "consulta": row["consulta"],
        "documento_relevante": row["documento_relevante"],
        "rank_tfidf": row["rank_tfidf"],
        "rank_lsa": row["rank_lsa"],
        "fallo_tfidf": row["fallo_tfidf"],
        "fallo_lsa": row["fallo_lsa"],
        "n_terminos_compartidos": row["n_terminos_compartidos"],
        "terminos_compartidos_con_relevante": ", ".join(row["terminos_compartidos_con_relevante"]) or "(ninguno)",
        "terminos_consulta_solo_en_otros_docs": "; ".join(
            f"{t}→{','.join(ds)}" for t, ds in row["terminos_de_la_consulta_presentes_en_otros_documentos"].items()
        ),
        "terminos_consulta_fuera_de_vocabulario": ", ".join(row["terminos_de_la_consulta_fuera_de_vocabulario"]) or "(ninguno)",
    })
pd.DataFrame(filas_t).to_csv("T3_tabla_terminos_compartidos.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 10) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resumen_fallos = {}
for m in ("tfidf", "lsa"):
    resumen_fallos[m] = {
        "n_fallos": len(fallos[m]),
        "consultas_fallidas": [f["consulta"] for f in fallos[m]],
        "consultas_sin_fallo_top3": [q for q in QIDS if rank_relevante[m][q] <= TOP_K],
        "q5_entre_los_fallos": any(f["consulta"] == "q5" for f in fallos[m]),
        "hit_at_3_equivalente": (len(QIDS) - len(fallos[m])) / len(QIDS),
    }

rank_del_relevante = {
    q: {
        "tfidf": rank_relevante["tfidf"][q],
        "lsa": rank_relevante["lsa"][q],
        "delta_lsa_menos_tfidf": rank_relevante["lsa"][q] - rank_relevante["tfidf"][q],
        "fallo_tfidf": rank_relevante["tfidf"][q] > TOP_K,
        "fallo_lsa": rank_relevante["lsa"][q] > TOP_K,
    }
    for q in QIDS
}

resultados = {
    "subtarea": "T3",
    "titulo": "Parte 3 — Análisis por consulta: fallos por método (relevante fuera del top 3) y términos compartidos",
    "config": {
        "criterio_de_fallo": "el documento relevante queda en posición >= 4, es decir, fuera del top 3",
        "top_k": TOP_K,
        "metodos": {
            "tfidf": "TF-IDF (TfidfVectorizer por defecto) + similitud coseno — parámetros de T1",
            "lsa": "LSA: TruncatedSVD(n_components=4, random_state=0) + coseno en el espacio latente — parámetros de T2",
        },
        "tokenizacion_de_terminos": "analyzer por defecto de TfidfVectorizer (minúsculas, patrón (?u)\\b\\w\\w+\\b, sin stopwords): la misma tokenización que usa el ranking",
        "tam_vocabulario": int(len(vocab)),
        "n_consultas": len(QIDS),
        "n_documentos": len(DOC_IDS),
        "desempate": "orden estable por id de documento (d01..d10)",
        "fuentes": {
            "rankings_tfidf": verif["fuente_rankings_tfidf"],
            "rankings_lsa": verif["fuente_rankings_lsa"],
        },
    },
    "verificacion_con_previas": verif,
    "resumen_fallos": resumen_fallos,
    "rank_del_relevante_por_metodo": rank_del_relevante,
    "tabla_fallos_por_metodo": fallos,
    "tabla_terminos_compartidos": tabla_terminos,
    "rankings_usados": RANKS,
    "insumos_para_el_reporte": insumos,
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 11) Resumen breve
# ---------------------------------------------------------------------------
print("=" * 72)
print("T3 — Análisis por consulta (fallo = relevante fuera del top 3)")
print("=" * 72)
print(f"Vocabulario TF-IDF: {len(vocab)} términos | Verificación T1: {verif['recomputo_tfidf_coincide_con_T1']} | "
      f"Verificación T2: {verif['recomputo_lsa_coincide_con_T2']}")
for m in ("tfidf", "lsa"):
    fs = fallos[m]
    print(f"\n{m.upper()}: {len(fs)} fallo(s) -> {[f['consulta'] for f in fs]}")
    for f in fs:
        print(f"   {f['consulta']}: relevante {f['documento_relevante']} en pos {f['rank_del_relevante']}; "
              f"comparte con él: {f['terminos_compartidos_con_relevante'] or '(nada)'}")
        for s in f["documentos_que_superan_al_relevante"]:
            print(f"      pos {s['pos']}: {s['doc']} comparte {s['terminos_compartidos_con_consulta'] or '(nada)'}")
print("\nTérminos compartidos consulta–relevante (6 consultas):")
for row in tabla_terminos:
    print(f"   {row['consulta']} vs {row['documento_relevante']}: "
          f"{row['terminos_compartidos_con_relevante'] or '(nada)'} "
          f"[rank TF-IDF {row['rank_tfidf']} | rank LSA {row['rank_lsa']}]")
print("\nArchivos escritos: resultados.json, T3_heatmap_terminos_compartidos.png, "
      "T3_rank_del_relevante_por_metodo.png, T3_tabla_fallos.csv, T3_tabla_terminos_compartidos.csv")
