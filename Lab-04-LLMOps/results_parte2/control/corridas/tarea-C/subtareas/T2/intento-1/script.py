# -*- coding: utf-8 -*-
"""
T2 — Parte 2: Semántica latente (LSA) sobre la línea base TF-IDF de la Parte 1.

Reutiliza el corpus de 10 documentos, las 6 consultas y los juicios de relevancia
de la Parte 1 (T1). Reproduce la MISMA matriz TF-IDF (TfidfVectorizer con
parámetros por defecto, ajustado solo con los documentos), la proyecta a 4
dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA), recalcula la
similitud coseno, reordena los 10 documentos por consulta y repite la evaluación
(Hit@1, Hit@3, MRR sobre las 6 consultas).

Salidas (carpeta actual):
  - resultados.json                     (contrato: tabla única 2 métodos × 3 métricas,
                                         rankings LSA por consulta, verificación con T1)
  - t2_tabla_comparativa.csv            (tabla única en CSV)
  - t2_rank_por_consulta.csv            (rank del relevante por consulta y método)
  - t2_metricas_comparacion.png         (barras agrupadas de las 3 métricas)
  - t2_rank_relevante_por_consulta.png  (rank del relevante por consulta, 2 métodos)
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

# ----------------------------------------------------------------------------
# 1) Corpus, consultas y juicios: EXACTAMENTE los de la Parte 1 (T1)
# ----------------------------------------------------------------------------
DOCUMENTOS = [
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

CONSULTAS = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = [d[0] for d in DOCUMENTOS]
doc_textos = [d[1] for d in DOCUMENTOS]
q_ids = [q[0] for q in CONSULTAS]
q_textos = [q[1] for q in CONSULTAS]
q_relevante = {q[0]: q[2] for q in CONSULTAS}
N_DOC, N_Q = len(doc_ids), len(q_ids)
METRICAS = ["Hit@1", "Hit@3", "MRR"]

# ----------------------------------------------------------------------------
# 2) Misma matriz TF-IDF que T1: parámetros por defecto, ajuste SOLO con docs
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que la Parte 1
X_docs_tfidf = vectorizer.fit_transform(doc_textos)
X_queries_tfidf = vectorizer.transform(q_textos)  # las consultas solo se proyectan
vocab_size = len(vectorizer.vocabulary_)

# ----------------------------------------------------------------------------
# 3) LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs_tfidf)
X_queries_lsa = svd.transform(X_queries_tfidf)  # folding-in de las consultas
varianza = svd.explained_variance_ratio_

# ----------------------------------------------------------------------------
# 4) Similitud coseno y rankings en los dos espacios
# ----------------------------------------------------------------------------
sim_tfidf = cosine_similarity(X_queries_tfidf, X_docs_tfidf)
sim_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)


def evaluar(matriz_sim):
    """Ordena los 10 documentos por consulta (similitud descendente, desempate
    estable en orden d01..d10) y calcula Hit@1, Hit@3 y MRR."""
    detalle = {}
    for i, qid in enumerate(q_ids):
        sims = matriz_sim[i]
        orden = np.argsort(-sims, kind="stable")
        ranking = [doc_ids[j] for j in orden]
        relevante = q_relevante[qid]
        rank = ranking.index(relevante) + 1
        detalle[qid] = {
            "consulta": q_textos[i],
            "relevante": relevante,
            "ranking": ranking,
            "rank_relevante": int(rank),
            "hit@1": int(rank == 1),
            "hit@3": int(rank <= 3),
            "reciprocal_rank": float(1.0 / rank),
            "similitud_relevante": float(sims[doc_ids.index(relevante)]),
            "similitudes_por_documento": {doc_ids[j]: float(sims[j]) for j in range(N_DOC)},
        }
    metricas = {
        "Hit@1": float(np.mean([detalle[q]["hit@1"] for q in q_ids])),
        "Hit@3": float(np.mean([detalle[q]["hit@3"] for q in q_ids])),
        "MRR": float(np.mean([detalle[q]["reciprocal_rank"] for q in q_ids])),
    }
    return metricas, detalle


metricas_tfidf, detalle_tfidf = evaluar(sim_tfidf)
metricas_lsa, detalle_lsa = evaluar(sim_lsa)

# ----------------------------------------------------------------------------
# 5) Tabla única: 2 métodos × 3 métricas (+ rank del relevante por consulta)
# ----------------------------------------------------------------------------
filas_tabla = [
    {"metodo": "TF-IDF (línea base, Parte 1)", **metricas_tfidf},
    {"metodo": "LSA (TruncatedSVD k=4, random_state=0)", **metricas_lsa},
]
tabla = pd.DataFrame(filas_tabla)
tabla.to_csv("t2_tabla_comparativa.csv", index=False, encoding="utf-8")

filas_rank = []
for q in q_ids:
    rt = detalle_tfidf[q]["rank_relevante"]
    rl = detalle_lsa[q]["rank_relevante"]
    filas_rank.append({
        "consulta": q,
        "relevante": q_relevante[q],
        "rank_tfidf": rt,
        "rank_lsa": rl,
        "delta_rank_lsa_menos_tfidf": rl - rt,
        "hit1_tfidf": detalle_tfidf[q]["hit@1"],
        "hit1_lsa": detalle_lsa[q]["hit@1"],
        "hit3_tfidf": detalle_tfidf[q]["hit@3"],
        "hit3_lsa": detalle_lsa[q]["hit@3"],
    })
tabla_ranks = pd.DataFrame(filas_rank)
tabla_ranks.to_csv("t2_rank_por_consulta.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------------
# 6) Verificación: este script reproduce la matriz/rankings TF-IDF de T1
# ----------------------------------------------------------------------------
verif = {
    "archivo_t1": "entrada/T1/resultados.json",
    "encontrado": False,
    "nota": "Comprueba que la matriz TF-IDF reutilizada es idéntica a la de la Parte 1.",
}
ruta_t1 = Path("entrada/T1/resultados.json")
if ruta_t1.exists():
    try:
        t1 = json.loads(ruta_t1.read_text(encoding="utf-8"))
        verif["encontrado"] = True
        mb = t1.get("metricas_linea_base", {})
        if mb:
            difs = {m: abs(float(mb[m]) - metricas_tfidf[m]) for m in METRICAS if m in mb}
            verif["max_dif_metricas_vs_T1"] = max(difs.values()) if difs else None
            verif["coincide_metricas"] = bool(difs) and all(v < 1e-9 for v in difs.values())
        rk_t1 = t1.get("rankings_por_consulta", {})
        if rk_t1:
            verif["coincide_rankings"] = all(
                list(rk_t1.get(q, [])) == detalle_tfidf[q]["ranking"] for q in q_ids
            )
        if "tam_vocabulario_tfidf" in t1:
            verif["vocabulario_T1"] = int(t1["tam_vocabulario_tfidf"])
            verif["vocabulario_reproducido"] = int(vocab_size)
            verif["coincide_vocabulario"] = bool(int(t1["tam_vocabulario_tfidf"]) == vocab_size)
    except Exception as exc:
        verif["error_lectura"] = f"{type(exc).__name__}: {exc}"

# ----------------------------------------------------------------------------
# 7) Figuras
# ----------------------------------------------------------------------------
# Figura 1: comparación de las 3 métricas entre métodos
fig, ax = plt.subplots(figsize=(7.2, 4.5))
x = np.arange(len(METRICAS))
w = 0.36
b1 = ax.bar(x - w / 2, [metricas_tfidf[m] for m in METRICAS], w,
            label="TF-IDF (línea base)", color="#4C72B0")
b2 = ax.bar(x + w / 2, [metricas_lsa[m] for m in METRICAS], w,
            label="LSA (TruncatedSVD, k=4)", color="#DD8452")
for barras in (b1, b2):
    try:
        ax.bar_label(barras, fmt="%.3f", fontsize=8, padding=2)
    except Exception:
        pass
ax.set_xticks(x)
ax.set_xticklabels(METRICAS)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Valor (promedio sobre 6 consultas)")
ax.set_title("Tarea C — Parte 2: recuperación TF-IDF vs LSA (k=4)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("t2_metricas_comparacion.png", dpi=120)
plt.close(fig)

# Figura 2: rank del documento relevante por consulta (análisis de fallos)
fig, ax = plt.subplots(figsize=(8.2, 4.5))
x = np.arange(N_Q)
ranks_t = [detalle_tfidf[q]["rank_relevante"] for q in q_ids]
ranks_l = [detalle_lsa[q]["rank_relevante"] for q in q_ids]
b1 = ax.bar(x - w / 2, ranks_t, w, label="TF-IDF", color="#4C72B0")
b2 = ax.bar(x + w / 2, ranks_l, w, label="LSA (k=4)", color="#DD8452")
for barras in (b1, b2):
    try:
        ax.bar_label(barras, fontsize=8, padding=2)
    except Exception:
        pass
ax.axhline(1, color="green", ls="--", lw=1, alpha=0.8, label="Umbral Hit@1 (rank=1)")
ax.axhline(3, color="darkorange", ls="--", lw=1, alpha=0.8, label="Umbral Hit@3 (rank=3)")
ax.set_xticks(x)
ax.set_xticklabels(q_ids)
ax.set_ylim(0, 11.5)
ax.set_ylabel("Rank del documento relevante")
ax.set_title("Rank del documento relevante por consulta: TF-IDF vs LSA (k=4)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("t2_rank_relevante_por_consulta.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Parte 2 — Semántica latente: la misma matriz TF-IDF de la Parte 1 se proyecta a 4 "
        "dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA); se recalcula la "
        "similitud coseno, se reordenan los 10 documentos por cada una de las 6 consultas y se "
        "repite la evaluación (Hit@1, Hit@3, MRR)."
    ),
    "n_documentos": N_DOC,
    "n_consultas": N_Q,
    "tam_vocabulario_tfidf": int(vocab_size),
    "configuracion": {
        "tfidf": "TfidfVectorizer con parámetros por defecto, ajustado SOLO con los 10 documentos (igual que T1)",
        "lsa": {
            "metodo": "TruncatedSVD",
            "n_components": 4,
            "random_state": 0,
            "algorithm": "randomized",
            "ajustado_sobre": (
                f"matriz TF-IDF {N_DOC}x{vocab_size} de los documentos; "
                "consultas proyectadas con svd.transform (folding-in)"
            ),
        },
        "similitud": "coseno (sklearn.metrics.pairwise.cosine_similarity)",
        "nota_ajuste": "Las consultas no intervienen en el ajuste del vectorizador ni del SVD.",
    },
    "varianza_explicada_por_componente": [float(v) for v in varianza],
    "varianza_explicada_acumulada": float(varianza.sum()),
    "metricas_tfidf": metricas_tfidf,
    "metricas_lsa": metricas_lsa,
    "diferencia_lsa_menos_tfidf": {m: metricas_lsa[m] - metricas_tfidf[m] for m in METRICAS},
    "tabla_metodos": filas_tabla,
    "tabla_rank_relevante_por_consulta": filas_rank,
    "rankings_lsa_por_consulta": {q: detalle_lsa[q]["ranking"] for q in q_ids},
    "rankings_tfidf_por_consulta": {q: detalle_tfidf[q]["ranking"] for q in q_ids},
    "detalle_lsa_por_consulta": detalle_lsa,
    "detalle_tfidf_por_consulta": detalle_tfidf,
    "verificacion_con_T1": verif,
    "archivos_generados": [
        "resultados.json",
        "t2_tabla_comparativa.csv",
        "t2_rank_por_consulta.csv",
        "t2_metricas_comparacion.png",
        "t2_rank_relevante_por_consulta.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 9) Resumen
# ----------------------------------------------------------------------------
print("=" * 72)
print("T2 — Semántica latente (LSA, TruncatedSVD k=4) vs línea base TF-IDF")
print("=" * 72)
print(f"Documentos: {N_DOC} | Consultas: {N_Q} | Vocabulario TF-IDF: {vocab_size} términos")
print(f"Varianza explicada por los 4 componentes LSA: {varianza.sum():.4f} "
      f"({', '.join(f'{v:.3f}' for v in varianza)})")
print("\nTABLA ÚNICA (2 métodos × 3 métricas, 6 consultas):")
print(tabla.to_string(index=False, float_format="{:.4f}".format))
print("\nRank del documento relevante por consulta (menor es mejor):")
print(tabla_ranks.to_string(index=False))
print("\nRankings LSA por consulta:")
for q in q_ids:
    r = detalle_lsa[q]
    print(f"  {q} (relevante={r['relevante']}, rank={r['rank_relevante']}): "
          + " > ".join(r["ranking"]))
if verif.get("encontrado"):
    print(f"\nVerificación con T1: métricas coinciden={verif.get('coincide_metricas')}, "
          f"rankings coinciden={verif.get('coincide_rankings')}, "
          f"vocabulario coincide={verif.get('coincide_vocabulario')}")
else:
    print("\nVerificación con T1: entrada/T1/resultados.json no encontrado (se continúa sin cruzar).")
print("\nArchivos generados:")
for a in resultados["archivos_generados"]:
    print(f"  - {a}")
