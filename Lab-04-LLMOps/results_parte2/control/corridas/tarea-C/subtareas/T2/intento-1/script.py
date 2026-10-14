# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la misma matriz TF-IDF de la Parte 1.

Pasos:
1) Reconstruye el corpus (10 documentos), las 6 consultas y los juicios de
   relevancia exactamente como aparecen en el enunciado.
2) Reproduce la matriz TF-IDF de la Parte 1 (TfidfVectorizer, parámetros por
   defecto) y verifica contra los resultados oficiales de T1
   (entrada/T1/resultados.json).
3) Proyecta documentos y consultas a 4 dimensiones con
   TruncatedSVD(n_components=4, random_state=0) (LSA). El SVD se ajusta SOLO
   con los documentos; las consultas se proyectan con la misma transformación.
4) Repite el ranking por similitud coseno en el espacio latente.
5) Construye una tabla comparativa única con Hit@1, Hit@3 y MRR de los dos
   métodos; guarda todo en resultados.json más dos figuras PNG.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (idénticos al enunciado)
# ---------------------------------------------------------------------------
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
DOCS = [
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

QUERY_IDS = ["q1", "q2", "q3", "q4", "q5", "q6"]
QUERIES = [
    "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "¿Cómo se añaden fragmentos recuperados al prompt?",
    "¿Qué técnica entrena matrices de bajo rango?",
    "¿Cómo se compara la orientación de dos vectores de texto?",
    "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "¿Qué arquitectura combina autoatención con capas feed-forward?",
]
RELEVANTE = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}

# ---------------------------------------------------------------------------
# 2) Misma matriz TF-IDF que la Parte 1 (parámetros por defecto)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que en la Parte 1
X_docs = vectorizer.fit_transform(DOCS)    # 10 x |V| (solo documentos)
X_queries = vectorizer.transform(QUERIES)  # 6 x |V|

S_tfidf = cosine_similarity(X_queries, X_docs)  # 6 x 10

# ---------------------------------------------------------------------------
# 3) LSA: TruncatedSVD(n_components=4, random_state=0), ajustado solo con docs
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
svd.fit(X_docs)
X_docs_lsa = svd.transform(X_docs)        # 10 x 4 (misma proyección para docs)
X_queries_lsa = svd.transform(X_queries)  # 6 x 4 (y para consultas)

S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)  # 6 x 10

# ---------------------------------------------------------------------------
# 4) Ranking por consulta y métricas Hit@1, Hit@3, MRR
# ---------------------------------------------------------------------------
def evaluar_matriz(S):
    """Ranking descendente por similitud (empates: orden d01..d10) y métricas."""
    por_consulta = []
    for i, qid in enumerate(QUERY_IDS):
        sims = S[i]
        orden = np.argsort(-sims, kind="stable")
        ranking = [DOC_IDS[j] for j in orden]
        rel = RELEVANTE[qid]
        rank = ranking.index(rel) + 1
        por_consulta.append({
            "consulta": qid,
            "texto_consulta": QUERIES[i],
            "documento_relevante": rel,
            "ranking_completo_10_docs": ranking,
            "similitudes_por_documento": {DOC_IDS[j]: float(sims[j]) for j in range(len(DOC_IDS))},
            "rank_del_relevante": int(rank),
            "reciprocal_rank": float(1.0 / rank),
            "hit@1": int(rank == 1),
            "hit@3": int(rank <= 3),
        })
    metricas = {
        "Hit@1": float(np.mean([c["hit@1"] for c in por_consulta])),
        "Hit@3": float(np.mean([c["hit@3"] for c in por_consulta])),
        "MRR": float(np.mean([c["reciprocal_rank"] for c in por_consulta])),
    }
    return por_consulta, metricas


lsa_por_consulta, metricas_lsa = evaluar_matriz(S_lsa)
tfidf_local_por_consulta, metricas_tfidf_local = evaluar_matriz(S_tfidf)

# Métricas TF-IDF oficiales de la Parte 1 (entrada/T1/resultados.json), con
# verificación contra el recálculo local (misma matriz, mismo criterio de ranking).
metricas_tfidf = dict(metricas_tfidf_local)
ranks_tfidf = {c["consulta"]: int(c["rank_del_relevante"]) for c in tfidf_local_por_consulta}
fuente_tfidf = "recalculado localmente con los parámetros de la Parte 1"
t1_path = Path("entrada") / "T1" / "resultados.json"
if t1_path.exists():
    try:
        with open(t1_path, encoding="utf-8") as f:
            t1 = json.load(f)
        metricas_tfidf = {k: float(v) for k, v in t1["metricas_agregadas"].items()}
        ranks_tfidf = {c["consulta"]: int(c["rank_del_relevante"]) for c in t1["por_consulta"]}
        fuente_tfidf = "entrada/T1/resultados.json (resultados oficiales de la Parte 1)"
    except Exception as exc:  # archivo ilegible/truncado -> fallback local
        print(f"Aviso: no se pudo leer {t1_path} ({exc}); se usan métricas TF-IDF recalculadas.")

coincide_parte1 = all(
    abs(metricas_tfidf[m] - metricas_tfidf_local[m]) < 1e-12
    for m in ("Hit@1", "Hit@3", "MRR")
)

# ---------------------------------------------------------------------------
# 5) Tabla comparativa única (los dos métodos x las tres métricas)
# ---------------------------------------------------------------------------
tabla_records = [
    {"Metodo": "TF-IDF (línea base léxica, Parte 1)",
     "Hit@1": metricas_tfidf["Hit@1"], "Hit@3": metricas_tfidf["Hit@3"], "MRR": metricas_tfidf["MRR"]},
    {"Metodo": "LSA (TruncatedSVD, 4 dimensiones)",
     "Hit@1": metricas_lsa["Hit@1"], "Hit@3": metricas_lsa["Hit@3"], "MRR": metricas_lsa["MRR"]},
]
tabla = pd.DataFrame(tabla_records)

comparacion_por_consulta = []
for c in lsa_por_consulta:
    qid = c["consulta"]
    rt, rl = int(ranks_tfidf[qid]), int(c["rank_del_relevante"])
    comparacion_por_consulta.append({
        "consulta": qid,
        "documento_relevante": c["documento_relevante"],
        "rank_tfidf": rt,
        "rank_lsa": rl,
        "diferencia_rank_tfidf_menos_lsa": rt - rl,
        "reciprocal_rank_tfidf": float(1.0 / rt),
        "reciprocal_rank_lsa": float(1.0 / rl),
    })

# ---------------------------------------------------------------------------
# 6) Figuras PNG
# ---------------------------------------------------------------------------
# 6a) Barras agrupadas: las tres métricas para los dos métodos
fig, ax = plt.subplots(figsize=(7.0, 4.5))
nombres_metricas = ["Hit@1", "Hit@3", "MRR"]
x = np.arange(len(nombres_metricas))
ancho = 0.35
b1 = ax.bar(x - ancho / 2, [metricas_tfidf[m] for m in nombres_metricas], ancho,
            label="TF-IDF (léxico)", color="#4C72B0")
b2 = ax.bar(x + ancho / 2, [metricas_lsa[m] for m in nombres_metricas], ancho,
            label="LSA (TruncatedSVD, 4 dim)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(nombres_metricas)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Valor (promedio sobre 6 consultas)")
ax.set_title("Recuperación sobre 10 documentos: TF-IDF vs. LSA (k=4)")
ax.legend(loc="upper center", ncol=2, frameon=True)
fig.tight_layout()
fig.savefig("comparacion_metricas_tfidf_lsa.png", dpi=120)
plt.close(fig)

# 6b) Mapa de calor de similitudes coseno en el espacio latente
fig, ax = plt.subplots(figsize=(8.5, 4.8))
im = ax.imshow(S_lsa, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS, rotation=45)
ax.set_yticks(range(len(QUERY_IDS)))
ax.set_yticklabels(QUERY_IDS)
vmax = float(S_lsa.max()) if float(S_lsa.max()) > 0 else 1.0
for i in range(len(QUERY_IDS)):
    for j in range(len(DOC_IDS)):
        val = float(S_lsa[i, j])
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if val < 0.55 * vmax else "black",
                fontweight="bold" if DOC_IDS[j] == RELEVANTE[QUERY_IDS[i]] else "normal")
ax.set_title("Similitud coseno consulta-documento en el espacio latente LSA (4 dim)\n(negrita = documento relevante del juicio)")
fig.colorbar(im, ax=ax, label="Similitud coseno")
fig.tight_layout()
fig.savefig("mapa_calor_similitud_lsa.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": ("Semántica latente: LSA con TruncatedSVD(n_components=4, random_state=0) "
                    "sobre la misma matriz TF-IDF de la Parte 1; ranking por similitud coseno "
                    "en el espacio latente y tabla comparativa única de Hit@1, Hit@3 y MRR."),
    "configuracion": {
        "tfidf": "TfidfVectorizer con parámetros por defecto (idéntico a la Parte 1)",
        "lsa": "TruncatedSVD(n_components=4, random_state=0), ajustado solo con los 10 documentos",
        "proyeccion_consultas": "mismo TfidfVectorizer + svd.transform (misma proyección que los documentos)",
        "similitud": "coseno en el espacio latente de 4 dimensiones",
        "n_documentos": 10,
        "n_consultas": 6,
        "tam_vocabulario_tfidf": int(len(vectorizer.vocabulary_)),
    },
    "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
    "varianza_explicada_acumulada_4_componentes": float(np.sum(svd.explained_variance_ratio_)),
    "metricas_tfidf_parte1": metricas_tfidf,
    "fuente_metricas_tfidf": fuente_tfidf,
    "verificacion_parte1": {
        "metricas_tfidf_recalculadas_localmente": metricas_tfidf_local,
        "coincide_con_resultados_T1": bool(coincide_parte1),
    },
    "metricas_lsa": metricas_lsa,
    "tabla_comparativa": {
        "columnas": ["Metodo", "Hit@1", "Hit@3", "MRR"],
        "filas": tabla_records,
    },
    "lsa_por_consulta": lsa_por_consulta,
    "comparacion_por_consulta": comparacion_por_consulta,
    "matriz_similitud_coseno_lsa": {
        "filas_consultas": QUERY_IDS,
        "columnas_documentos": DOC_IDS,
        "valores": [[float(v) for v in fila] for fila in S_lsa],
    },
    "figuras": ["comparacion_metricas_tfidf_lsa.png", "mapa_calor_similitud_lsa.png"],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 72)
print("T2 — Semántica latente (LSA, TruncatedSVD k=4) vs. línea base TF-IDF")
print("=" * 72)
print(f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos | "
      f"Varianza explicada (4 comp.): {float(np.sum(svd.explained_variance_ratio_)):.4f}")
print(f"Métricas TF-IDF tomadas de: {fuente_tfidf} | coinciden con recálculo local: {coincide_parte1}")
print("\nTabla comparativa (6 consultas):")
print(tabla.to_string(index=False))
print("\nRankings LSA por consulta ([...] = documento relevante):")
for c in lsa_por_consulta:
    ranking_str = " > ".join(
        f"[{d}]" if d == c["documento_relevante"] else d
        for d in c["ranking_completo_10_docs"]
    )
    print(f"  {c['consulta']} (rel={c['documento_relevante']}, rank={c['rank_del_relevante']}): {ranking_str}")
print("\nFiguras: comparacion_metricas_tfidf_lsa.png, mapa_calor_similitud_lsa.png")
print("Resultados escritos en resultados.json")
