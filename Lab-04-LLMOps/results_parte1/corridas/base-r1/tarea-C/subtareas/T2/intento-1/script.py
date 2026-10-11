#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Parte 2 de la Tarea C: semántica latente (LSA) sobre la misma matriz TF-IDF de T1.

Pasos:
  1. Reconstruye el corpus (10 documentos), las 6 consultas y los juicios de T1.
  2. Reconstruye la matriz TF-IDF con TfidfVectorizer por defecto (IDF ajustado SOLO
     sobre los 10 documentos; las consultas solo se transforman), igual que en T1.
  3. Proyecta a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA),
     ajustado SOLO sobre la matriz TF-IDF de los documentos; las consultas se
     proyectan con transform al mismo espacio latente.
  4. Rankea los 10 documentos por similitud coseno en el espacio latente y calcula
     Hit@1, Hit@3 y MRR sobre las 6 consultas.
  5. Construye la tabla comparativa TF-IDF vs LSA (2 métodos × 3 métricas) y guarda
     los rankings por consulta de LSA.

Salidas: resultados.json, t2_comparativa_metodos.png, t2_tabla_comparativa.png,
         t2_ranks_por_consulta.png, t2_docs_espacio_latente.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios: idénticos a T1 (datos citados del enunciado)
# ---------------------------------------------------------------------------
DOCS = [
    ("d01", "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión."),
    ("d02", "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización."),
    ("d03", "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo."),
    ("d04", "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento."),
    ("d05", "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación."),
    ("d06", "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado."),
    ("d07", "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos."),
    ("d08", "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales."),
    ("d09", "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad."),
    ("d10", "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria."),
]
QUERIES = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = [d[0] for d in DOCS]
doc_texts = [d[1] for d in DOCS]
q_ids = [q[0] for q in QUERIES]
q_texts = [q[1] for q in QUERIES]
RELEVANTE = {qid: rel for qid, _, rel in QUERIES}
N_DOCS, N_QUERIES = len(DOCS), len(QUERIES)
K_LATENTE, SEED = 4, 0

# ---------------------------------------------------------------------------
# 2) Misma matriz TF-IDF que T1 (parámetros por defecto, IDF sobre los 10 docs)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que en T1
X_docs = vectorizer.fit_transform(doc_texts)      # ajuste SOLO con los documentos
X_queries = vectorizer.transform(q_texts)         # las consultas NUNCA ajustan nada

# ---------------------------------------------------------------------------
# 3) Línea base TF-IDF oficial de T1 (entrada/T1/resultados.json) + verificación
# ---------------------------------------------------------------------------
t1 = None
t1_path = Path("entrada") / "T1" / "resultados.json"
if t1_path.exists():
    try:
        with t1_path.open(encoding="utf-8") as f:
            t1 = json.load(f)
    except Exception:
        t1 = None


def rankear(filas_sim):
    """Orden estable: similitud descendente; empates por id de documento (d01..d10)."""
    rankings = []
    for i in range(filas_sim.shape[0]):
        sims = filas_sim[i]
        orden = sorted(range(N_DOCS), key=lambda j: (-float(sims[j]), j))
        rankings.append(orden)
    return rankings


def rank_del_relevante(orden, doc_rel):
    for pos, j in enumerate(orden, start=1):
        if doc_ids[j] == doc_rel:
            return pos
    raise ValueError("documento relevante no encontrado en el ranking")


def metricas_globales(rankings):
    h1 = h3 = 0
    rr_sum = 0.0
    for qid, orden in zip(q_ids, rankings):
        pos = rank_del_relevante(orden, RELEVANTE[qid])
        h1 += int(pos == 1)
        h3 += int(pos <= 3)
        rr_sum += 1.0 / pos
    return {"hit_at_1": h1 / N_QUERIES, "hit_at_3": h3 / N_QUERIES, "mrr": rr_sum / N_QUERIES}


# TF-IDF recomputada (solo como verificación de que la matriz es la misma de T1)
S_tfidf = cosine_similarity(X_queries, X_docs)
rank_tfidf = rankear(S_tfidf)
met_tfidf_recalc = metricas_globales(rank_tfidf)

if t1 is not None and isinstance(t1.get("metricas_globales"), dict):
    met_tfidf = {k: float(t1["metricas_globales"][k]) for k in ("hit_at_1", "hit_at_3", "mrr")}
    fuente_tfidf = "entrada/T1/resultados.json (línea base oficial de T1)"
else:
    met_tfidf = dict(met_tfidf_recalc)
    fuente_tfidf = "recomputada en T2 con el mismo procedimiento de T1 (T1 no disponible)"

t1_ranks = {}
if t1 is not None:
    for qid, info in (t1.get("metricas_por_consulta") or {}).items():
        if isinstance(info, dict) and info.get("rank_del_relevante") is not None:
            t1_ranks[qid] = int(info["rank_del_relevante"])

vocab_t1 = (t1.get("config") or {}).get("tam_vocabulario") if t1 is not None else None
coincide_vocab = None if vocab_t1 is None else bool(int(vocab_t1) == int(X_docs.shape[1]))
coincide_metricas = (
    None if t1 is None
    else bool(all(np.isclose(met_tfidf_recalc[k], met_tfidf[k], atol=1e-9)
                  for k in ("hit_at_1", "hit_at_3", "mrr")))
)

# ---------------------------------------------------------------------------
# 4) LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=K_LATENTE, random_state=SEED)
L_docs = svd.fit_transform(X_docs)      # ajuste SOLO sobre los 10 documentos
L_queries = svd.transform(X_queries)    # consultas proyectadas al mismo espacio

S_lsa = cosine_similarity(L_queries, L_docs)
rank_lsa = rankear(S_lsa)
met_lsa = metricas_globales(rank_lsa)

# ---------------------------------------------------------------------------
# 5) Rankings por consulta de LSA + tablas
# ---------------------------------------------------------------------------
metricas_por_consulta_lsa = {}
for i, qid in enumerate(q_ids):
    orden = rank_lsa[i]
    rel = RELEVANTE[qid]
    pos = rank_del_relevante(orden, rel)
    metricas_por_consulta_lsa[qid] = {
        "texto_consulta": q_texts[i],
        "documento_relevante": rel,
        "rank_del_relevante": int(pos),
        "reciprocal_rank": 1.0 / pos,
        "hit_at_1": int(pos == 1),
        "hit_at_3": int(pos <= 3),
        "ranking_completo": [
            {"pos": p, "doc": doc_ids[j], "similitud_coseno_lsa": float(S_lsa[i, j])}
            for p, j in enumerate(orden, start=1)
        ],
    }

tabla_comparativa = {
    "descripcion": "Métricas de recuperación sobre las 6 consultas (10 documentos): 2 métodos × 3 métricas",
    "columnas": ["metodo", "hit_at_1", "hit_at_3", "mrr"],
    "filas": [
        {
            "metodo": "TF-IDF + coseno (T1, línea base léxica)",
            "hit_at_1": met_tfidf["hit_at_1"],
            "hit_at_3": met_tfidf["hit_at_3"],
            "mrr": met_tfidf["mrr"],
        },
        {
            "metodo": "LSA: TruncatedSVD(k=4, random_state=0) + coseno",
            "hit_at_1": met_lsa["hit_at_1"],
            "hit_at_3": met_lsa["hit_at_3"],
            "mrr": met_lsa["mrr"],
        },
    ],
    "deltas_lsa_menos_tfidf": {
        "hit_at_1": met_lsa["hit_at_1"] - met_tfidf["hit_at_1"],
        "hit_at_3": met_lsa["hit_at_3"] - met_tfidf["hit_at_3"],
        "mrr": met_lsa["mrr"] - met_tfidf["mrr"],
    },
}

tabla_por_consulta = []
for i, qid in enumerate(q_ids):
    rel = RELEVANTE[qid]
    rk_t = t1_ranks.get(qid, rank_del_relevante(rank_tfidf[i], rel))
    rk_l = rank_del_relevante(rank_lsa[i], rel)
    tabla_por_consulta.append({
        "consulta": qid,
        "texto_consulta": q_texts[i],
        "documento_relevante": rel,
        "rank_tfidf": int(rk_t),
        "rank_lsa": int(rk_l),
        "hit_at_1_tfidf": int(rk_t == 1),
        "hit_at_1_lsa": int(rk_l == 1),
        "hit_at_3_tfidf": int(rk_t <= 3),
        "hit_at_3_lsa": int(rk_l <= 3),
        "reciprocal_rank_tfidf": 1.0 / rk_t,
        "reciprocal_rank_lsa": 1.0 / rk_l,
    })

# ---------------------------------------------------------------------------
# 6) Figuras (PNG, dpi=120)
# ---------------------------------------------------------------------------
# 6a) Barras agrupadas: TF-IDF vs LSA
fig, ax = plt.subplots(figsize=(7.0, 4.2))
nombres = ["Hit@1", "Hit@3", "MRR"]
vals_t = [met_tfidf["hit_at_1"], met_tfidf["hit_at_3"], met_tfidf["mrr"]]
vals_l = [met_lsa["hit_at_1"], met_lsa["hit_at_3"], met_lsa["mrr"]]
x = np.arange(3)
w = 0.36
ax.bar(x - w / 2, vals_t, w, label="TF-IDF (léxico, T1)", color="#4C72B0")
ax.bar(x + w / 2, vals_l, w, label="LSA (TruncatedSVD k=4)", color="#DD8452")
for xi, v in zip(x - w / 2, vals_t):
    ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, vals_l):
    ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(nombres)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Valor (promedio sobre 6 consultas)")
ax.set_title("Tarea C — Parte 2: TF-IDF vs LSA (k=4, random_state=0)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("t2_comparativa_metodos.png", dpi=120)
plt.close(fig)

# 6b) Tabla comparativa renderizada
fig, ax = plt.subplots(figsize=(7.4, 2.4))
ax.axis("off")
cols = ["Método", "Hit@1", "Hit@3", "MRR"]
rows = [
    ["TF-IDF + coseno (T1)", f"{met_tfidf['hit_at_1']:.4f}", f"{met_tfidf['hit_at_3']:.4f}", f"{met_tfidf['mrr']:.4f}"],
    ["LSA (SVD k=4) + coseno", f"{met_lsa['hit_at_1']:.4f}", f"{met_lsa['hit_at_3']:.4f}", f"{met_lsa['mrr']:.4f}"],
]
tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
               colWidths=[0.40, 0.20, 0.20, 0.20])
tab.auto_set_font_size(False)
tab.set_fontsize(11)
tab.scale(1, 1.7)
ax.set_title("Tabla comparativa sobre las 6 consultas (2 métodos × 3 métricas)",
             fontsize=12, pad=14)
fig.tight_layout()
fig.savefig("t2_tabla_comparativa.png", dpi=120)
plt.close(fig)

# 6c) Rank del relevante por consulta (TF-IDF vs LSA)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
y = np.arange(N_QUERIES)
rk_t = [r["rank_tfidf"] for r in tabla_por_consulta]
rk_l = [r["rank_lsa"] for r in tabla_por_consulta]
ax.axvspan(0.3, 3.5, color="green", alpha=0.06, label="zona Hit@3")
ax.axvspan(0.3, 1.5, color="green", alpha=0.12, label="zona Hit@1")
ax.scatter([r - 0.15 for r in rk_t], y, marker="o", s=80, facecolors="none",
           edgecolors="#4C72B0", linewidths=2, label="TF-IDF", zorder=3)
ax.scatter([r + 0.15 for r in rk_l], y, marker="s", s=70, color="#DD8452",
           alpha=0.9, label="LSA (k=4)", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"{q}  (rel: {RELEVANTE[q]})" for q in q_ids])
ax.invert_yaxis()
ax.set_xticks(range(1, 11))
ax.set_xlim(0.3, 10.7)
ax.set_xlabel("Rank del documento relevante (menor = mejor)")
ax.set_title("Rank del relevante por consulta: TF-IDF vs LSA")
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("t2_ranks_por_consulta.png", dpi=120)
plt.close(fig)

# 6d) Documentos y consultas en el espacio latente (componentes 1 y 2)
fig, ax = plt.subplots(figsize=(7.0, 5.4))
ax.scatter(L_docs[:, 0], L_docs[:, 1], color="#4C72B0", s=60, zorder=3, label="documentos")
for k, did in enumerate(doc_ids):
    ax.annotate(did, (L_docs[k, 0], L_docs[k, 1]), textcoords="offset points",
                xytext=(6, 4), fontsize=9)
ax.scatter(L_queries[:, 0], L_queries[:, 1], color="#DD8452", marker="x", s=90,
           linewidths=2, zorder=3, label="consultas")
for k, qid in enumerate(q_ids):
    ax.annotate(qid, (L_queries[k, 0], L_queries[k, 1]), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color="#B04A2A")
ax.axhline(0, color="gray", lw=0.6, alpha=0.6)
ax.axvline(0, color="gray", lw=0.6, alpha=0.6)
ax.set_xlabel("Componente latente 1")
ax.set_ylabel("Componente latente 2")
ax.set_title("Espacio latente LSA (k=4): documentos y consultas")
ax.legend()
fig.tight_layout()
fig.savefig("t2_docs_espacio_latente.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "titulo": "Parte 2 — Semántica latente (LSA, TruncatedSVD k=4) y tabla comparativa con TF-IDF",
    "config": {
        "vectorizador": "sklearn TfidfVectorizer, parámetros por defecto (idéntico a T1)",
        "idf_ajustado_sobre": "los 10 documentos del corpus (consultas solo transform)",
        "lsa": "TruncatedSVD(n_components=4, random_state=0) ajustado sobre la matriz TF-IDF de los 10 documentos; consultas proyectadas con transform",
        "similitud": "coseno en el espacio latente de 4 dimensiones",
        "n_documentos": N_DOCS,
        "n_consultas": N_QUERIES,
        "dimensiones_latentes": K_LATENTE,
        "random_state": SEED,
        "tam_vocabulario": int(X_docs.shape[1]),
        "desempate": "orden estable por id de documento (d01..d10)",
        "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
        "varianza_explicada_total": float(svd.explained_variance_ratio_.sum()),
        "valores_singulares": [float(v) for v in svd.singular_values_],
    },
    "metricas_globales_tfidf": met_tfidf,
    "fuente_metricas_tfidf": fuente_tfidf,
    "metricas_globales_lsa": met_lsa,
    "tabla_comparativa": tabla_comparativa,
    "metricas_por_consulta_lsa": metricas_por_consulta_lsa,
    "tabla_por_consulta": tabla_por_consulta,
    "coordenadas_latentes": {
        "documentos": {doc_ids[k]: [float(v) for v in L_docs[k]] for k in range(N_DOCS)},
        "consultas": {q_ids[k]: [float(v) for v in L_queries[k]] for k in range(N_QUERIES)},
    },
    "verificacion_con_T1": {
        "t1_cargado": t1 is not None,
        "ruta_T1": str(t1_path),
        "archivos_en_entrada_T1": sorted(p.name for p in t1_path.parent.iterdir()) if t1_path.parent.exists() else [],
        "tam_vocabulario_T1": vocab_t1,
        "vocabulario_coincide_con_T1": coincide_vocab,
        "metricas_tfidf_recomputadas_en_T2": met_tfidf_recalc,
        "metricas_tfidf_coinciden_con_T1": coincide_metricas,
    },
    "figuras": [
        "t2_comparativa_metodos.png",
        "t2_tabla_comparativa.png",
        "t2_ranks_por_consulta.png",
        "t2_docs_espacio_latente.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 66)
print("T2 — LSA (TruncatedSVD k=4, random_state=0) sobre la matriz TF-IDF de T1")
print(f"Matriz TF-IDF: {X_docs.shape[0]} docs × {X_docs.shape[1]} términos | "
      f"varianza explicada por las 4 componentes: {svd.explained_variance_ratio_.sum():.4f}")
print("-" * 66)
print("Tabla comparativa (6 consultas):")
print(f"  {'Método':<34}{'Hit@1':>9}{'Hit@3':>9}{'MRR':>9}")
print(f"  {'TF-IDF + coseno (T1)':<34}{met_tfidf['hit_at_1']:>9.4f}{met_tfidf['hit_at_3']:>9.4f}{met_tfidf['mrr']:>9.4f}")
print(f"  {'LSA (SVD k=4) + coseno':<34}{met_lsa['hit_at_1']:>9.4f}{met_lsa['hit_at_3']:>9.4f}{met_lsa['mrr']:>9.4f}")
print("-" * 66)
print("Rank del relevante por consulta (TF-IDF → LSA):")
for r in tabla_por_consulta:
    print(f"  {r['consulta']}: {r['rank_tfidf']} → {r['rank_lsa']}  (relevante: {r['documento_relevante']})")
print("-" * 66)
print("Verificación T1:", json.dumps(resultados["verificacion_con_T1"], ensure_ascii=False))
print("Figuras: t2_comparativa_metodos.png, t2_tabla_comparativa.png, "
      "t2_ranks_por_consulta.png, t2_docs_espacio_latente.png")
print("Resultados escritos en resultados.json")
