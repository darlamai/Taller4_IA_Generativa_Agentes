#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica (Tarea C, MMIA 6013).
Corpus de 10 documentos + 6 consultas (datos tomados íntegramente del enunciado).
Vectorización con TfidfVectorizer de scikit-learn (parámetros por defecto),
similitud coseno consulta-documento, ranking completo de los 10 documentos por
consulta y evaluación de Hit@1, Hit@3 y MRR sobre las 6 consultas.

Salidas:
  - resultados.json              (contrato: rankings completos + métricas)
  - t1_heatmap_similitud.png     (mapa de calor de similitudes)
  - t1_rank_relevante.png        (rank del documento relevante por consulta)
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

RUTA_RESULTADOS = Path("resultados.json")

# ---------------------------------------------------------------------------
# 1. Corpus y consultas, tal como aparecen en el enunciado
# ---------------------------------------------------------------------------
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
DOC_TEXTOS = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
]

CONSULTAS = [
    {"id": "q1", "texto": "¿Qué función de ranking léxica pondera la frecuencia de términos?", "relevante": "d04"},
    {"id": "q2", "texto": "¿Cómo se añaden fragmentos recuperados al prompt?", "relevante": "d03"},
    {"id": "q3", "texto": "¿Qué técnica entrena matrices de bajo rango?", "relevante": "d08"},
    {"id": "q4", "texto": "¿Cómo se compara la orientación de dos vectores de texto?", "relevante": "d05"},
    {"id": "q5", "texto": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "relevante": "d09"},
    {"id": "q6", "texto": "¿Qué arquitectura combina autoatención con capas feed-forward?", "relevante": "d02"},
]

# ---------------------------------------------------------------------------
# 2. TF-IDF con parámetros por defecto + similitud coseno
#    El vectorizador se ajusta SOLO con los documentos (índice); las consultas
#    únicamente se transforman con ese vocabulario (no se ajusta nada con ellas).
# ---------------------------------------------------------------------------
vectorizador = TfidfVectorizer()  # parámetros por defecto, como exige el enunciado
X_docs = vectorizador.fit_transform(DOC_TEXTOS)
X_cons = vectorizador.transform([c["texto"] for c in CONSULTAS])

S = cosine_similarity(X_cons, X_docs)  # forma (6 consultas, 10 documentos)

# ---------------------------------------------------------------------------
# 3. Ranking completo por consulta y métricas Hit@1, Hit@3, MRR
# ---------------------------------------------------------------------------
por_consulta = []
reciprocal_ranks, hits1, hits3 = [], [], []

for i, c in enumerate(CONSULTAS):
    orden = np.argsort(-S[i], kind="stable")  # descendente; desempates por orden d01..d10
    ranking = [DOC_IDS[j] for j in orden]
    rank_rel = ranking.index(c["relevante"]) + 1
    rr = 1.0 / rank_rel
    reciprocal_ranks.append(rr)
    hits1.append(1 if rank_rel == 1 else 0)
    hits3.append(1 if rank_rel <= 3 else 0)
    por_consulta.append({
        "consulta": c["id"],
        "texto_consulta": c["texto"],
        "documento_relevante": c["relevante"],
        "ranking_completo_10_docs": ranking,
        "similitudes_por_documento": {DOC_IDS[j]: float(S[i, j]) for j in range(len(DOC_IDS))},
        "rank_del_relevante": int(rank_rel),
        "reciprocal_rank": float(rr),
        "hit@1": int(rank_rel == 1),
        "hit@3": int(rank_rel <= 3),
    })

metricas = {
    "Hit@1": float(np.mean(hits1)),
    "Hit@3": float(np.mean(hits3)),
    "MRR": float(np.mean(reciprocal_ranks)),
}

# ---------------------------------------------------------------------------
# 4. resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "descripcion": "Línea base léxica: TF-IDF (TfidfVectorizer por defecto) + similitud coseno",
    "n_documentos": len(DOC_IDS),
    "n_consultas": len(CONSULTAS),
    "tam_vocabulario_tfidf": int(len(vectorizador.vocabulary_)),
    "metricas_agregadas": metricas,
    "por_consulta": por_consulta,
    "matriz_similitud_coseno": {
        "filas_consultas": [c["id"] for c in CONSULTAS],
        "columnas_documentos": DOC_IDS,
        "valores": [[float(v) for v in fila] for fila in S],
    },
}
RUTA_RESULTADOS.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# 5. Figuras
# ---------------------------------------------------------------------------
# Figura 1: mapa de calor de similitudes
fig, ax = plt.subplots(figsize=(9.5, 4.6))
im = ax.imshow(S, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(DOC_IDS)), DOC_IDS)
ax.set_yticks(range(len(CONSULTAS)), [c["id"] for c in CONSULTAS])
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
fig.colorbar(im, ax=ax, label="similitud coseno")
ax.set_title("T1: similitud coseno consulta–documento (TF-IDF, parámetros por defecto)")
ax.set_xlabel("documentos")
ax.set_ylabel("consultas")
fig.tight_layout()
fig.savefig("t1_heatmap_similitud.png", dpi=120)
plt.close(fig)

# Figura 2: rank del documento relevante por consulta
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ids_q = [pc["consulta"] for pc in por_consulta]
ranks = [pc["rank_del_relevante"] for pc in por_consulta]
colores = ["#2a9d8f" if r <= 3 else "#e76f51" for r in ranks]
ax.bar(ids_q, ranks, color=colores)
ax.axhline(1, color="gray", lw=0.8, ls=":")
ax.axhline(3, color="gray", lw=0.8, ls="--", label="umbral Hit@3")
ax.set_ylim(0, 10)
ax.invert_yaxis()
ax.set_xlabel("consulta")
ax.set_ylabel("rank del documento relevante")
ax.set_title(f"T1: TF-IDF+coseno — Hit@1={metricas['Hit@1']:.3f}, "
             f"Hit@3={metricas['Hit@3']:.3f}, MRR={metricas['MRR']:.3f}")
for k, r in enumerate(ranks):
    ax.text(k, r + 0.15, str(r), ha="center", va="bottom", fontsize=9)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("t1_rank_relevante.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Resumen por consola
# ---------------------------------------------------------------------------
tabla = pd.DataFrame({
    "consulta": ids_q,
    "relevante": [pc["documento_relevante"] for pc in por_consulta],
    "rank_relevante": ranks,
    "RR": [f"{pc['reciprocal_rank']:.4f}" for pc in por_consulta],
    "top3": [" > ".join(pc["ranking_completo_10_docs"][:3]) for pc in por_consulta],
})
print("=" * 78)
print("T1 — Línea base léxica: TF-IDF (por defecto) + similitud coseno")
print("=" * 78)
print(tabla.to_string(index=False))
print("-" * 78)
print(f"Hit@1 = {metricas['Hit@1']:.6f}  ({int(sum(hits1))}/6)")
print(f"Hit@3 = {metricas['Hit@3']:.6f}  ({int(sum(hits3))}/6)")
print(f"MRR   = {metricas['MRR']:.6f}")
print(f"Vocabulario TF-IDF: {len(vectorizador.vocabulary_)} términos")
print(f"Resultados escritos en: {RUTA_RESULTADOS.resolve()}")
print("Figuras: t1_heatmap_similitud.png, t1_rank_relevante.png")
