# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica (Tarea C, MMIA 6013)
============================================
Carga los 10 documentos y las 6 consultas (con sus juicios de relevancia) del enunciado,
vectoriza con TfidfVectorizer (parámetros por defecto), calcula similitud coseno
consulta-documento, ordena los 10 documentos por consulta y calcula Hit@1, Hit@3 y MRR
sobre las 6 consultas.

Salidas:
  - resultados.json              (contrato: rankings completos + métricas)
  - t1_similitud_coseno.png      (mapa de calor de similitudes)
  - t1_metricas_baseline.png     (tabla de métricas de la línea base)
  - t1_documentos.csv, t1_consultas.csv  (datos crudos para subtareas posteriores)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1. Corpus y consultas (datos completos, tomados del enunciado de la Tarea C)
# ----------------------------------------------------------------------
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
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(documentos.keys())
doc_texts = [documentos[d] for d in doc_ids]
query_ids = list(consultas.keys())
query_texts = [consultas[q][0] for q in query_ids]

# ----------------------------------------------------------------------
# 2. Vectorización TF-IDF (parámetros por defecto de scikit-learn)
#    El vocabulario/IDF se aprende SOLO del corpus de documentos;
#    las consultas se transforman con ese mismo vectorizador.
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)

# ----------------------------------------------------------------------
# 3. Similitud coseno consulta-documento (matriz 6 x 10)
# ----------------------------------------------------------------------
S = cosine_similarity(X_queries, X_docs)

# ----------------------------------------------------------------------
# 4. Ranking completo de los 10 documentos para cada consulta
# ----------------------------------------------------------------------
rankings = {}
por_consulta = {}
for i, qid in enumerate(query_ids):
    orden = np.argsort(-S[i], kind="stable")  # descendente por similitud
    ranking = [doc_ids[j] for j in orden]
    rankings[qid] = ranking
    relevante = consultas[qid][1]
    rank = ranking.index(relevante) + 1  # posición 1-based del documento relevante
    por_consulta[qid] = {
        "consulta": consultas[qid][0],
        "relevante": relevante,
        "ranking": ranking,
        "rank_relevante": int(rank),
        "hit@1": int(rank == 1),
        "hit@3": int(rank <= 3),
        "reciprocal_rank": float(1.0 / rank),
        "similitud_relevante": float(S[i, doc_ids.index(relevante)]),
        "similitudes_por_documento": {doc_ids[j]: float(S[i, j]) for j in range(len(doc_ids))},
    }

# ----------------------------------------------------------------------
# 5. Métricas agregadas sobre las 6 consultas
# ----------------------------------------------------------------------
hit1 = float(np.mean([por_consulta[q]["hit@1"] for q in query_ids]))
hit3 = float(np.mean([por_consulta[q]["hit@3"] for q in query_ids]))
mrr = float(np.mean([por_consulta[q]["reciprocal_rank"] for q in query_ids]))

# ----------------------------------------------------------------------
# 6. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "descripcion": "Línea base léxica: TfidfVectorizer (parámetros por defecto) + similitud coseno",
    "n_documentos": len(doc_ids),
    "n_consultas": len(query_ids),
    "tam_vocabulario_tfidf": int(len(vectorizer.vocabulary_)),
    "metricas_linea_base": {
        "Hit@1": hit1,
        "Hit@3": hit3,
        "MRR": mrr,
    },
    "metricas_por_consulta": por_consulta,
    "rankings_por_consulta": rankings,
    "matriz_similitud_coseno": {
        "filas_consultas": query_ids,
        "columnas_documentos": doc_ids,
        "valores": [[float(S[i, j]) for j in range(len(doc_ids))] for i in range(len(query_ids))],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Datos crudos reutilizables para subtareas posteriores
# ----------------------------------------------------------------------
pd.DataFrame({"id": doc_ids, "texto": doc_texts}).to_csv("t1_documentos.csv", index=False, encoding="utf-8")
pd.DataFrame(
    [{"id": q, "consulta": consultas[q][0], "relevante": consultas[q][1]} for q in query_ids]
).to_csv("t1_consultas.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# 8. Figuras PNG
# ----------------------------------------------------------------------
# 8a. Mapa de calor de la similitud coseno consulta-documento
fig, ax = plt.subplots(figsize=(9.5, 5))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(S.max(), 0.1))
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids)
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids)
for i in range(len(query_ids)):
    for j in range(len(doc_ids)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
ax.set_title("T1 — Similitud coseno consulta-documento (TF-IDF, parámetros por defecto)")
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
fig.colorbar(im, ax=ax, label="similitud coseno")
fig.tight_layout()
fig.savefig("t1_similitud_coseno.png", dpi=120)
plt.close(fig)

# 8b. Tabla de métricas de la línea base
fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.axis("off")
tabla = ax.table(
    cellText=[[f"{hit1:.6f}"], [f"{hit3:.6f}"], [f"{mrr:.6f}"]],
    rowLabels=["Hit@1", "Hit@3", "MRR"],
    colLabels=["Línea base TF-IDF"],
    loc="center",
    cellLoc="center",
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1, 1.8)
ax.set_title("T1 — Métricas de la línea base léxica (6 consultas, 10 documentos)")
fig.tight_layout()
fig.savefig("t1_metricas_baseline.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 9. Resumen breve por consola
# ----------------------------------------------------------------------
print("=== T1: Línea base léxica (TF-IDF + similitud coseno) ===")
print(f"Documentos: {len(doc_ids)} | Consultas: {len(query_ids)} | Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
for qid in query_ids:
    pq = por_consulta[qid]
    print(f"{qid} (relevante={pq['relevante']}): rank={pq['rank_relevante']} | "
          f"ranking: {' > '.join(pq['ranking'])}")
print("-" * 60)
print(f"Hit@1 = {hit1:.6f}")
print(f"Hit@3 = {hit3:.6f}")
print(f"MRR   = {mrr:.6f}")
print("Archivos escritos: resultados.json, t1_similitud_coseno.png, "
      "t1_metricas_baseline.png, t1_documentos.csv, t1_consultas.csv")
