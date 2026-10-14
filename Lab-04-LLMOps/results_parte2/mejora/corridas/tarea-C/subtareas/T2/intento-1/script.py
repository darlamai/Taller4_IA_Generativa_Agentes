#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la matriz TF-IDF de la Parte 1 (T1).

Pasos:
  1. Carga de T1 (entrada/T1/resultados.json) el corpus de 10 documentos, las
     6 consultas y sus juicios de relevancia (respaldo literal del enunciado).
  2. Reconstruye la matriz TF-IDF exactamente como en T1:
     TfidfVectorizer() con parámetros por defecto, ajustado SOLO sobre los
     10 documentos del corpus.
  3. Proyecta a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0)
     (LSA). Las consultas se proyectan con el mismo vectorizador y la misma
     SVD (solo transform; nada se reajusta con las consultas).
  4. Recalcula la similitud coseno en el espacio latente, ordena los 10
     documentos por consulta y evalúa Hit@1, Hit@3 y MRR.
  5. Genera una tabla única TF-IDF vs LSA (JSON/CSV/PNG), los rankings por
     consulta de LSA y figuras comparativas.

Salidas (carpeta actual):
  resultados.json, tabla_comparativa.csv, tabla_comparativa.png,
  mapa_calor_tfidf_vs_lsa.png, metricas_tfidf_vs_lsa.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios: idénticos a T1
# ---------------------------------------------------------------------------
CORPUS_ENUNCIADO = {
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

CONSULTAS_ENUNCIADO = {
    "q1": {"texto": "¿Qué función de ranking léxica pondera la frecuencia de términos?", "relevante": "d04"},
    "q2": {"texto": "¿Cómo se añaden fragmentos recuperados al prompt?", "relevante": "d03"},
    "q3": {"texto": "¿Qué técnica entrena matrices de bajo rango?", "relevante": "d08"},
    "q4": {"texto": "¿Cómo se compara la orientación de dos vectores de texto?", "relevante": "d05"},
    "q5": {"texto": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "relevante": "d09"},
    "q6": {"texto": "¿Qué arquitectura combina autoatención con capas feed-forward?", "relevante": "d02"},
}

RUTA_T1 = Path("entrada") / "T1" / "resultados.json"

t1 = None
fuente = "enunciado (respaldo)"
corpus, consultas = dict(CORPUS_ENUNCIADO), dict(CONSULTAS_ENUNCIADO)
if RUTA_T1.exists():
    try:
        with open(RUTA_T1, encoding="utf-8") as f:
            t1 = json.load(f)
        corpus = t1.get("corpus", corpus)
        consultas = t1.get("consultas", consultas)
        fuente = str(RUTA_T1)
    except Exception:
        t1 = None
        corpus, consultas = dict(CORPUS_ENUNCIADO), dict(CONSULTAS_ENUNCIADO)
        fuente = "enunciado (respaldo; lectura de T1 falló)"

doc_ids = sorted(corpus.keys())
docs = [corpus[d] for d in doc_ids]
q_ids = sorted(consultas.keys())
q_texts = [consultas[q]["texto"] for q in q_ids]
relevante = {q: consultas[q]["relevante"] for q in q_ids}

# ---------------------------------------------------------------------------
# 2) Matriz TF-IDF (idéntica a T1: ajuste SOLO sobre el corpus)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()           # parámetros por defecto, igual que T1
X_docs = vectorizer.fit_transform(docs)  # 10 x V
X_cons = vectorizer.transform(q_texts)   # las consultas nunca participan del ajuste

# ---------------------------------------------------------------------------
# 3) LSA: TruncatedSVD a 4 dimensiones (random_state=0)
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)   # proyección de los documentos
X_cons_lsa = svd.transform(X_cons)       # mismas componentes para las consultas

S_tfidf = cosine_similarity(X_cons, X_docs)          # 6 x 10 (línea base)
S_lsa = cosine_similarity(X_cons_lsa, X_docs_lsa)    # 6 x 10 (espacio latente)

# ---------------------------------------------------------------------------
# 4) Rankings y métricas (empates: orden original, como en T1)
# ---------------------------------------------------------------------------
def ranking_de_fila(sim_fila):
    orden = np.argsort(-np.asarray(sim_fila), kind="stable")
    return [doc_ids[i] for i in orden]

rankings_tfidf = {q: ranking_de_fila(S_tfidf[i]) for i, q in enumerate(q_ids)}
rankings_lsa = {q: ranking_de_fila(S_lsa[i]) for i, q in enumerate(q_ids)}

METRICAS = ("Hit@1", "Hit@3", "MRR")

def metricas(rankings):
    h1 = h3 = 0.0
    rr = []
    for q in q_ids:
        r = rankings[q].index(relevante[q]) + 1
        h1 += float(r == 1)
        h3 += float(r <= 3)
        rr.append(1.0 / r)
    n = len(q_ids)
    return {"Hit@1": h1 / n, "Hit@3": h3 / n, "MRR": float(np.mean(rr))}

m_tfidf = metricas(rankings_tfidf)
m_lsa = metricas(rankings_lsa)

# --- verificación contra T1 (rankings y, si existen, métricas guardadas) ---
def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

metricas_t1_guardadas = None
if isinstance(t1, dict):
    candidatos = [t1]
    for v in t1.values():
        if isinstance(v, dict):
            candidatos.append(v)
            for w in v.values():
                if isinstance(w, dict):
                    candidatos.append(w)
    for c in candidatos:
        out = {}
        for k, v in c.items():
            nk = _norm(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if "hit" in nk and nk.endswith("1"):
                out["Hit@1"] = float(v)
            elif "hit" in nk and nk.endswith("3"):
                out["Hit@3"] = float(v)
            elif "mrr" in nk:
                out["MRR"] = float(v)
        if len(out) == 3:
            metricas_t1_guardadas = out
            break

coincide_rankings_t1 = None
if isinstance(t1, dict) and isinstance(t1.get("ranking_por_consulta"), dict):
    coincide_rankings_t1 = all(
        list(t1["ranking_por_consulta"].get(q, [])) == rankings_tfidf[q] for q in q_ids
    )

# ---------------------------------------------------------------------------
# 5) Tabla única TF-IDF vs LSA
# ---------------------------------------------------------------------------
filas_tabla = [
    {"metodo": "TF-IDF (línea base)", "Hit@1": m_tfidf["Hit@1"], "Hit@3": m_tfidf["Hit@3"], "MRR": m_tfidf["MRR"]},
    {"metodo": "LSA (TruncatedSVD, 4D)", "Hit@1": m_lsa["Hit@1"], "Hit@3": m_lsa["Hit@3"], "MRR": m_lsa["MRR"]},
]
tabla = pd.DataFrame(filas_tabla).set_index("metodo")[list(METRICAS)]
tabla.to_csv("tabla_comparativa.csv", encoding="utf-8")

lineas_md = ["| Método | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|"]
for m in tabla.index:
    lineas_md.append("| " + str(m) + " | " + " | ".join(f"{tabla.loc[m, c]:.4f}" for c in METRICAS) + " |")
tabla_md = "\n".join(lineas_md)

detalle = {}
for i, q in enumerate(q_ids):
    rel = relevante[q]
    detalle[q] = {
        "consulta": q_texts[i],
        "relevante": rel,
        "rank_tfidf": rankings_tfidf[q].index(rel) + 1,
        "rank_lsa": rankings_lsa[q].index(rel) + 1,
        "ranking_tfidf": rankings_tfidf[q],
        "ranking_lsa": rankings_lsa[q],
        "sim_lsa_doc_relevante": float(S_lsa[i, doc_ids.index(rel)]),
    }

sim_lsa_json = {q: {d: float(S_lsa[i, j]) for j, d in enumerate(doc_ids)} for i, q in enumerate(q_ids)}

# ---------------------------------------------------------------------------
# 6) Figuras
# ---------------------------------------------------------------------------
# 6a) Tabla comparativa como PNG
fig, ax = plt.subplots(figsize=(7.8, 2.9))
ax.axis("off")
celdas = [[f"{tabla.loc[m, c]:.4f}" for c in METRICAS] for m in tabla.index]
tbl = ax.table(cellText=celdas, rowLabels=list(tabla.index), colLabels=list(METRICAS),
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
for (fila, col), celda in tbl.get_celld().items():
    if fila == 0 or col == -1:
        celda.set_facecolor("#DCE6F1")
        celda.set_text_props(weight="bold")
ax.set_title("Tarea C · Parte 2 — Hit@1, Hit@3 y MRR: TF-IDF vs LSA (6 consultas)",
             fontsize=11, pad=14)
plt.savefig("tabla_comparativa.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 6b) Mapas de calor TF-IDF vs LSA (misma escala de color)
def panel_heatmap(ax, S, titulo, vmin, vmax):
    im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(doc_ids)))
    ax.set_xticklabels(doc_ids, fontsize=8)
    ax.set_yticks(range(len(q_ids)))
    ax.set_yticklabels(q_ids, fontsize=8)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            frac = (S[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if frac > 0.55 else "white")
    for i, q in enumerate(q_ids):
        j = doc_ids.index(relevante[q])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.8))
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("documentos")
    ax.set_ylabel("consultas")
    return im

vmin = float(min(S_tfidf.min(), S_lsa.min()))
vmax = float(max(S_tfidf.max(), S_lsa.max()))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
panel_heatmap(axes[0], S_tfidf, "TF-IDF (espacio original) — línea base Parte 1", vmin, vmax)
im2 = panel_heatmap(axes[1], S_lsa, "LSA — TruncatedSVD 4 componentes (random_state=0)", vmin, vmax)
fig.colorbar(im2, ax=axes, shrink=0.85, label="similitud coseno")
fig.suptitle("Tarea C — similitud coseno consulta×documento (recuadro rojo = relevante)", fontsize=11)
plt.savefig("mapa_calor_tfidf_vs_lsa.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 6c) Barras comparativas de las tres métricas
fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(METRICAS))
w = 0.36
b1 = ax.bar(x - w / 2, [m_tfidf[c] for c in METRICAS], w, label="TF-IDF (línea base)", color="#4C72B0")
b2 = ax.bar(x + w / 2, [m_lsa[c] for c in METRICAS], w, label="LSA (TruncatedSVD, 4D)", color="#DD8452")
ax.bar_label(b1, fmt="%.3f", fontsize=9)
ax.bar_label(b2, fmt="%.3f", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(METRICAS)
ax.set_ylim(0, 1.15)
ax.set_ylabel("valor promedio (6 consultas)")
ax.set_title("Tarea C — recuperación: TF-IDF vs LSA (4 dimensiones)")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.savefig("metricas_tfidf_vs_lsa.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "titulo": "Semántica latente: LSA (TruncatedSVD, 4D) sobre la matriz TF-IDF de T1",
    "config": {
        "vectorizador": "TfidfVectorizer (parámetros por defecto), idéntico a T1",
        "ajuste_vectorizador": "solo sobre los 10 documentos del corpus",
        "svd": "TruncatedSVD(n_components=4, random_state=0)",
        "n_componentes": 4,
        "random_state": 0,
        "ajuste_svd": "matriz TF-IDF de los 10 documentos (las consultas solo se proyectan)",
        "proyeccion_consultas": "vectorizer.transform -> svd.transform (sin reajuste)",
        "similitud": "coseno en el espacio latente de 4 dimensiones",
        "n_documentos": len(doc_ids),
        "n_consultas": len(q_ids),
        "tam_vocabulario": int(len(vectorizer.vocabulary_)),
        "fuente_corpus_y_juicios": fuente,
    },
    "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
    "varianza_explicada_total_4d": float(svd.explained_variance_ratio_.sum()),
    "tabla_comparativa": filas_tabla,
    "tabla_comparativa_markdown": tabla_md,
    "metricas": {"tfidf": m_tfidf, "lsa": m_lsa},
    "delta_lsa_menos_tfidf": {c: float(m_lsa[c] - m_tfidf[c]) for c in METRICAS},
    "rankings_tfidf_por_consulta": rankings_tfidf,
    "rankings_lsa_por_consulta": rankings_lsa,
    "detalle_por_consulta": detalle,
    "similitud_coseno_lsa": sim_lsa_json,
    "verificacion_con_T1": {
        "t1_disponible": t1 is not None,
        "rankings_tfidf_recalculados_iguales_a_T1": coincide_rankings_t1,
        "metricas_t1_guardadas": metricas_t1_guardadas,
        "metricas_tfidf_recalculadas": m_tfidf,
    },
    "figuras": [
        "tabla_comparativa.png",
        "mapa_calor_tfidf_vs_lsa.png",
        "metricas_tfidf_vs_lsa.png",
    ],
    "archivos_generados": [
        "resultados.json",
        "tabla_comparativa.csv",
        "tabla_comparativa.png",
        "mapa_calor_tfidf_vs_lsa.png",
        "metricas_tfidf_vs_lsa.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 66)
print("T2 — LSA (TruncatedSVD 4D, random_state=0) sobre la matriz TF-IDF de T1")
print("=" * 66)
print(f"Documentos: {len(doc_ids)} | Consultas: {len(q_ids)} | "
      f"Vocabulario: {len(vectorizer.vocabulary_)}")
print(f"Varianza explicada por las 4 componentes: "
      f"{[round(float(v), 4) for v in svd.explained_variance_ratio_]} "
      f"(total {float(svd.explained_variance_ratio_.sum()):.4f})")
print("\nTabla comparativa (promedio sobre 6 consultas):")
print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
print("\nPosición (1-based) del documento relevante por consulta:")
for q in q_ids:
    print(f"  {q}: TF-IDF rank={detalle[q]['rank_tfidf']:2d} | "
          f"LSA rank={detalle[q]['rank_lsa']:2d} (relevante={relevante[q]})")
if coincide_rankings_t1 is not None:
    print(f"\nVerificación: rankings TF-IDF recalculados idénticos a T1 -> "
          f"{coincide_rankings_t1}")
print("\nArchivos generados:")
for a in resultados["archivos_generados"]:
    print("  -", a)
