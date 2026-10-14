# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica (Tarea C · MMIA 6013)
Corpus de 10 documentos + 6 consultas con juicios de relevancia (datos del enunciado).
Vectorización con TfidfVectorizer (parámetros por defecto) + similitud coseno.
Ranking completo de los 10 documentos por consulta y métricas Hit@1, Hit@3 y MRR.
Salidas: resultados.json, t1_similitud_coseno_heatmap.png, t1_rank_relevante_por_consulta.png
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Corpus: 10 documentos cortos (textuales del enunciado, Parte 1)
# ------------------------------------------------------------------
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
DOC_IDS = list(DOCS.keys())
DOC_TEXTS = [DOCS[d] for d in DOC_IDS]

# ------------------------------------------------------------------
# 2) Consultas y juicios de relevancia (1 documento relevante por consulta)
# ------------------------------------------------------------------
QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
Q_IDS = list(QUERIES.keys())
Q_TEXTS = [QUERIES[q][0] for q in Q_IDS]

# ------------------------------------------------------------------
# 3) Vectorización TF-IDF (parámetros por defecto)
#    El vectorizador se ajusta SOLO con el corpus de 10 documentos
#    (práctica estándar de IR); las consultas se transforman con ese vocabulario.
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(DOC_TEXTS)   # (10 x V)
X_queries = vectorizer.transform(Q_TEXTS)      # (6 x V)

# ------------------------------------------------------------------
# 4) Similitud coseno consulta-documento
# ------------------------------------------------------------------
S = cosine_similarity(X_queries, X_docs)       # (6 x 10)

# ------------------------------------------------------------------
# 5) Ranking completo por consulta y métricas Hit@1, Hit@3, MRR
# ------------------------------------------------------------------
rankings = {}
sims = {}
ranking_detallado = {}
rank_relevante = {}
metricas_q = {}

for i, q in enumerate(Q_IDS):
    order = np.argsort(-S[i], kind="stable")   # descendente; empates -> orden del corpus
    ranked_ids = [DOC_IDS[j] for j in order]
    rankings[q] = ranked_ids
    sims[q] = {DOC_IDS[j]: float(S[i, j]) for j in range(len(DOC_IDS))}
    ranking_detallado[q] = [
        {"pos": p + 1, "doc": ranked_ids[p], "sim": float(S[i, j])}
        for p, j in enumerate(order)
    ]
    rel = QUERIES[q][1]
    r = ranked_ids.index(rel) + 1
    rank_relevante[q] = r
    metricas_q[q] = {
        "relevante": rel,
        "rank": r,
        "hit@1": 1 if r == 1 else 0,
        "hit@3": 1 if r <= 3 else 0,
        "reciprocal_rank": 1.0 / r,
    }

hit1_count = int(sum(metricas_q[q]["hit@1"] for q in Q_IDS))
hit3_count = int(sum(metricas_q[q]["hit@3"] for q in Q_IDS))
hit1 = float(np.mean([metricas_q[q]["hit@1"] for q in Q_IDS]))
hit3 = float(np.mean([metricas_q[q]["hit@3"] for q in Q_IDS]))
mrr = float(np.mean([metricas_q[q]["reciprocal_rank"] for q in Q_IDS]))

# ------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "titulo": "Línea base léxica: TF-IDF (defecto) + similitud coseno",
    "config": {
        "vectorizador": "TfidfVectorizer (parámetros por defecto)",
        "vocabulario_ajustado_sobre": "corpus de 10 documentos",
        "tam_vocabulario": int(len(vectorizer.vocabulary_)),
        "similitud": "coseno",
        "n_documentos": len(DOC_IDS),
        "n_consultas": len(Q_IDS),
    },
    "corpus": DOCS,
    "consultas": {q: {"texto": QUERIES[q][0], "relevante": QUERIES[q][1]} for q in Q_IDS},
    "similitud_coseno": sims,
    "ranking_por_consulta": rankings,
    "ranking_detallado": ranking_detallado,
    "rank_del_relevante_por_consulta": rank_relevante,
    "metricas_por_consulta": metricas_q,
    "hit_at_1": hit1,
    "hit_at_1_aciertos": hit1_count,
    "hit_at_3": hit3,
    "hit_at_3_aciertos": hit3_count,
    "mrr": mrr,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 7) Figuras
# ------------------------------------------------------------------
# 7a) Mapa de calor de la matriz de similitud coseno (6 consultas x 10 documentos)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0)
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS)
ax.set_yticks(range(len(Q_IDS)))
ax.set_yticklabels(Q_IDS)
for i in range(len(Q_IDS)):
    for j in range(len(DOC_IDS)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
fig.colorbar(im, ax=ax, label="similitud coseno")
ax.set_title("T1 · TF-IDF (defecto): similitud coseno consulta-documento")
fig.tight_layout()
fig.savefig("t1_similitud_coseno_heatmap.png", dpi=120)
plt.close(fig)

# 7b) Rank del documento relevante por consulta
ranks = [rank_relevante[q] for q in Q_IDS]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(Q_IDS, ranks, color=["tab:green" if r <= 3 else "tab:red" for r in ranks])
ax.axhline(3, ls="--", c="gray", label="umbral Hit@3")
ax.set_ylim(0, 10.5)
ax.set_ylabel("rank del documento relevante")
ax.set_title(f"T1 línea base TF-IDF · Hit@1={hit1:.3f} · Hit@3={hit3:.3f} · MRR={mrr:.3f}")
for i, r in enumerate(ranks):
    ax.text(i, r + 0.15, str(r), ha="center", fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig("t1_rank_relevante_por_consulta.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 8) Resumen
# ------------------------------------------------------------------
print("=" * 72)
print("T1 · Línea base léxica: TF-IDF (parámetros por defecto) + similitud coseno")
print("=" * 72)
print(f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos (ajustado solo con el corpus)")
for q in Q_IDS:
    rel, r = QUERIES[q][1], rank_relevante[q]
    print(f"{q}: «{QUERIES[q][0]}»  [relevante: {rel}]")
    print(f"   ranking: {' > '.join(rankings[q])}")
    print(f"   rank del relevante = {r} | Hit@1={metricas_q[q]['hit@1']} "
          f"Hit@3={metricas_q[q]['hit@3']} RR={metricas_q[q]['reciprocal_rank']:.4f}")
print("-" * 72)
print(f"Agregado sobre 6 consultas:  Hit@1 = {hit1:.6f} ({hit1_count}/6)   "
      f"Hit@3 = {hit3:.6f} ({hit3_count}/6)   MRR = {mrr:.6f}")
print("Salidas: resultados.json | t1_similitud_coseno_heatmap.png | t1_rank_relevante_por_consulta.png")
