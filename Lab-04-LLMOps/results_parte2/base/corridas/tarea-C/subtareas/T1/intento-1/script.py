# ============================================================================
# T1 — Línea base léxica: TF-IDF (TfidfVectorizer por defecto) + coseno
# Corpus: 10 documentos (d01–d10) · 6 consultas (q1–q6) con 1 relevante c/u
# Métricas: Hit@1, Hit@3, MRR sobre las 6 consultas + ranking completo
# Salidas: resultados.json, rankings_tfidf.csv, metricas_tfidf_por_consulta.csv,
#          corpus.csv, consultas.csv, t1_linea_base_tfidf.png
# ============================================================================
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1) Corpus y consultas (textuales del enunciado, Tarea C — Parte 1)
# ----------------------------------------------------------------------------
documentos = {
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

consultas = {
    "q1": {"texto": "¿Qué función de ranking léxica pondera la frecuencia de términos?", "relevante": "d04"},
    "q2": {"texto": "¿Cómo se añaden fragmentos recuperados al prompt?", "relevante": "d03"},
    "q3": {"texto": "¿Qué técnica entrena matrices de bajo rango?", "relevante": "d08"},
    "q4": {"texto": "¿Cómo se compara la orientación de dos vectores de texto?", "relevante": "d05"},
    "q5": {"texto": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "relevante": "d09"},
    "q6": {"texto": "¿Qué arquitectura combina autoatención con capas feed-forward?", "relevante": "d02"},
}

doc_ids = sorted(documentos.keys())          # d01..d10 (orden fijo)
doc_texts = [documentos[d] for d in doc_ids]
q_ids = list(consultas.keys())               # q1..q6
q_texts = [consultas[q]["texto"] for q in q_ids]

# ----------------------------------------------------------------------------
# 2) Vectorización TF-IDF con parámetros por defecto.
#    El IDF se ajusta SOLO con los 10 documentos (práctica IR estándar);
#    las consultas se proyectan al vocabulario del corpus con transform().
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()               # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)
X_q = vectorizer.transform(q_texts)

# ----------------------------------------------------------------------------
# 3) Similitud coseno (6 x 10) y ranking por consulta
# ----------------------------------------------------------------------------
S = cosine_similarity(X_q, X_docs)           # S[i, j] = cos(q_i, d_j)

rankings = {}
por_consulta = {}
rr_list, h1_list, h3_list = [], [], []

for i, q in enumerate(q_ids):
    orden = np.argsort(-S[i], kind="stable")             # descendente; desempate estable por id
    ranking = [doc_ids[j] for j in orden]
    scores = [float(S[i, j]) for j in orden]
    rel = consultas[q]["relevante"]
    pos = ranking.index(rel) + 1                          # rank 1-based del relevante
    rr_i = 1.0 / pos
    h1_i = int(pos == 1)
    h3_i = int(pos <= 3)
    rr_list.append(rr_i); h1_list.append(h1_i); h3_list.append(h3_i)
    rankings[q] = ranking
    por_consulta[q] = {
        "texto_consulta": consultas[q]["texto"],
        "documento_relevante": rel,
        "rank_del_relevante": int(pos),
        "reciprocal_rank": float(rr_i),
        "hit_at_1": int(h1_i),
        "hit_at_3": int(h3_i),
        "ranking_completo": [
            {"pos": p, "doc": d, "similitud_coseno": s}
            for p, (d, s) in enumerate(zip(ranking, scores), start=1)
        ],
    }

hit_at_1 = float(np.mean(h1_list))
hit_at_3 = float(np.mean(h3_list))
mrr = float(np.mean(rr_list))

# ----------------------------------------------------------------------------
# 4) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "titulo": "Línea base léxica: TF-IDF + similitud coseno",
    "config": {
        "vectorizador": "sklearn TfidfVectorizer, parámetros por defecto",
        "idf_ajustado_sobre": "los 10 documentos del corpus (consultas solo transform)",
        "similitud": "coseno (sklearn.metrics.pairwise.cosine_similarity)",
        "n_documentos": len(doc_ids),
        "n_consultas": len(q_ids),
        "tam_vocabulario": int(len(vectorizer.vocabulary_)),
        "desempate": "orden estable por id de documento (d01..d10)",
    },
    "metricas_globales": {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": mrr,
    },
    "metricas_por_consulta": por_consulta,
    "rankings_por_consulta": rankings,   # q -> [d??, ...] ordenados por similitud desc.
    "matriz_similitud_coseno": {
        "filas_consultas": q_ids,
        "columnas_documentos": doc_ids,
        "valores": [[float(v) for v in fila] for fila in S],
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 5) CSVs de apoyo para subtareas siguientes
# ----------------------------------------------------------------------------
pd.DataFrame(
    [{"id": d, "texto": t} for d, t in documentos.items()]
).to_csv("corpus.csv", index=False, encoding="utf-8")

pd.DataFrame(
    [{"id": q, "consulta": v["texto"], "relevante": v["relevante"]} for q, v in consultas.items()]
).to_csv("consultas.csv", index=False, encoding="utf-8")

filas = []
for i, q in enumerate(q_ids):
    orden = np.argsort(-S[i], kind="stable")
    for pos, j in enumerate(orden, start=1):
        filas.append({
            "consulta": q,
            "posicion": pos,
            "documento": doc_ids[j],
            "similitud_coseno": float(S[i, j]),
            "es_relevante": int(doc_ids[j] == consultas[q]["relevante"]),
        })
pd.DataFrame(filas).to_csv("rankings_tfidf.csv", index=False, encoding="utf-8")

tabla = pd.DataFrame([{
    "consulta": q,
    "relevante": consultas[q]["relevante"],
    "rank_relevante": por_consulta[q]["rank_del_relevante"],
    "hit@1": por_consulta[q]["hit_at_1"],
    "hit@3": por_consulta[q]["hit_at_3"],
    "reciprocal_rank": por_consulta[q]["reciprocal_rank"],
} for q in q_ids])
tabla.to_csv("metricas_tfidf_por_consulta.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------------
# 6) Figura: mapa de calor de similitudes + 1/rank por consulta
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
im = ax.imshow(S, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids, rotation=90, fontsize=8)
ax.set_yticks(range(len(q_ids)))
ax.set_yticklabels(q_ids, fontsize=8)
for i in range(len(q_ids)):
    for j in range(len(doc_ids)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6.5)
ax.set_title("Similitud coseno TF-IDF (consulta × documento)")
fig.colorbar(im, ax=ax, shrink=0.85)

ax2 = axes[1]
colores = ["#2ca02c" if r == 1.0 else ("#ff7f0e" if r >= 1.0 / 3 else "#d62728") for r in rr_list]
ax2.bar(q_ids, rr_list, color=colores)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("1 / rank del documento relevante")
ax2.set_title(f"Línea base TF-IDF — Hit@1={hit_at_1:.3f} | Hit@3={hit_at_3:.3f} | MRR={mrr:.3f}")
for k, r in enumerate(rr_list):
    ax2.text(k, r + 0.02, f"{r:.3f}", ha="center", fontsize=8)
ax2.axhline(mrr, color="gray", linestyle="--", linewidth=1, label=f"MRR = {mrr:.3f}")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("t1_linea_base_tfidf.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 7) Resumen
# ----------------------------------------------------------------------------
print("=" * 72)
print("T1 — Línea base TF-IDF (TfidfVectorizer por defecto) + similitud coseno")
print("=" * 72)
print(f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos (10 docs, 6 consultas)")
print(tabla.to_string(index=False))
print("-" * 72)
print(f"Hit@1 = {hit_at_1:.6f}  ({int(sum(h1_list))}/6 consultas)")
print(f"Hit@3 = {hit_at_3:.6f}  ({int(sum(h3_list))}/6 consultas)")
print(f"MRR   = {mrr:.6f}")
print("-" * 72)
for q in q_ids:
    print(f"{q}: {' > '.join(rankings[q])}  | relevante {consultas[q]['relevante']} en pos {por_consulta[q]['rank_del_relevante']}")
print("-" * 72)
print("Archivos escritos: resultados.json, rankings_tfidf.csv, "
      "metricas_tfidf_por_consulta.csv, corpus.csv, consultas.csv, t1_linea_base_tfidf.png")
