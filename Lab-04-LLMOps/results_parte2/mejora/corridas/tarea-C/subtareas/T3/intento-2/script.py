# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta:
  (1) detectar en qué consultas falla cada método (documento relevante fuera
      del top 3), comparando los rankings de T1 (TF-IDF) y T2 (LSA 4D), y
  (2) calcular qué términos comparten cada consulta y su documento relevante
      (intersección de vocabulario y pesos TF-IDF) para sustentar el porqué
      de los fallos (en especial q5).

Entradas : entrada/T1/resultados.json y entrada/T2/resultados.json
Salidas  : resultados.json, T3_fallos_y_terminos.png, T3_q5_terminos.png
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
# 1) Corpus y consultas (fuente primaria: salida de T1; respaldo: enunciado)
# ---------------------------------------------------------------------------
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
Q_IDS = [f"q{i}" for i in range(1, 7)]

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


def cargar_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


t1 = cargar_json(Path("entrada") / "T1" / "resultados.json")
t2 = cargar_json(Path("entrada") / "T2" / "resultados.json")

corpus = {d: ((t1.get("corpus") or {}).get(d) or CORPUS_ENUNCIADO[d]) for d in DOC_IDS}
consultas = {}
for q in Q_IDS:
    base = (t1.get("consultas") or {}).get(q) or CONSULTAS_ENUNCIADO[q]
    consultas[q] = {
        "texto": base.get("texto", CONSULTAS_ENUNCIADO[q]["texto"]),
        "relevante": base.get("relevante", CONSULTAS_ENUNCIADO[q]["relevante"]),
    }

# ---------------------------------------------------------------------------
# 2) Reproducción exacta de T1 (TF-IDF) y T2 (LSA 4D); ajuste SOLO con los docs
# ---------------------------------------------------------------------------
vec = TfidfVectorizer()  # parámetros por defecto, idéntico a T1
X_docs = vec.fit_transform([corpus[d] for d in DOC_IDS])      # ajuste: solo 10 documentos
X_q = vec.transform([consultas[q]["texto"] for q in Q_IDS])   # consultas solo se transforman
S_tfidf = cosine_similarity(X_q, X_docs)

svd = TruncatedSVD(n_components=4, random_state=0)
D_lsa = svd.fit_transform(X_docs)
Q_lsa = svd.transform(X_q)
S_lsa = cosine_similarity(Q_lsa, D_lsa)
var4d = float(svd.explained_variance_ratio_.sum())

# ---------------------------------------------------------------------------
# 3) Rankings: autoritativos = los guardados por T1/T2; recálculo = verificación
# ---------------------------------------------------------------------------
def ranking_de(fila):
    orden = np.argsort(-np.asarray(fila), kind="stable")
    return [DOC_IDS[i] for i in orden]


def rango(ranking, doc):
    return int(ranking.index(doc) + 1)


rank_rec_tfidf = {q: ranking_de(S_tfidf[i]) for i, q in enumerate(Q_IDS)}
rank_rec_lsa = {q: ranking_de(S_lsa[i]) for i, q in enumerate(Q_IDS)}

rank_t1_file = t1.get("ranking_por_consulta") or t2.get("rankings_tfidf_por_consulta") or {}
rank_lsa_file = t2.get("rankings_lsa_por_consulta") or {}

rank_tfidf = {q: (rank_t1_file.get(q) or rank_rec_tfidf[q]) for q in Q_IDS}
rank_lsa = {q: (rank_lsa_file.get(q) or rank_rec_lsa[q]) for q in Q_IDS}

# ---------------------------------------------------------------------------
# 4) Tabla de fallos por método y por consulta (criterio: relevante fuera del top 3)
# ---------------------------------------------------------------------------
def metricas_de_ranks(ranks):
    r = np.asarray(ranks, dtype=float)
    return {
        "Hit@1": float((r == 1).mean()),
        "Hit@3": float((r <= 3).mean()),
        "MRR": float((1.0 / r).mean()),
    }


filas = []
for i, q in enumerate(Q_IDS):
    rel = consultas[q]["relevante"]
    j = DOC_IDS.index(rel)
    r_t = rango(rank_tfidf[q], rel)
    r_l = rango(rank_lsa[q], rel)
    filas.append({
        "consulta": q,
        "texto_consulta": consultas[q]["texto"],
        "relevante": rel,
        "rank_tfidf": r_t,
        "fallo_tfidf": bool(r_t > 3),
        "rank_lsa": r_l,
        "fallo_lsa": bool(r_l > 3),
        "cambio_rank_lsa_menos_tfidf": r_l - r_t,
        "sim_tfidf_relevante": float(S_tfidf[i, j]),
        "sim_lsa_relevante": float(S_lsa[i, j]),
    })
fila_por_q = {f["consulta"]: f for f in filas}

ranks_t = [f["rank_tfidf"] for f in filas]
ranks_l = [f["rank_lsa"] for f in filas]
met_tfidf = metricas_de_ranks(ranks_t)
met_lsa = metricas_de_ranks(ranks_l)

fallos_tfidf = [f["consulta"] for f in filas if f["fallo_tfidf"]]
fallos_lsa = [f["consulta"] for f in filas if f["fallo_lsa"]]

# ---------------------------------------------------------------------------
# 5) Términos compartidos consulta–documento relevante (vocabulario + pesos)
# ---------------------------------------------------------------------------
feat = vec.get_feature_names_out()
vocab = vec.vocabulary_
idf_vec = vec.idf_
Qd = X_q.toarray()
Dd = X_docs.toarray()
df_counts = np.asarray((X_docs > 0).sum(axis=0)).ravel()
analyzer = vec.build_analyzer()


def no_ceros(vector_denso):
    idx = np.nonzero(vector_denso)[0]
    return {feat[k]: float(vector_denso[k]) for k in idx}


terminos_compartidos = {}
for i, q in enumerate(Q_IDS):
    rel = consultas[q]["relevante"]
    j = DOC_IDS.index(rel)
    q_nz = no_ceros(Qd[i])
    d_nz = no_ceros(Dd[j])
    compartidos = sorted(set(q_nz) & set(d_nz), key=lambda t: (-(q_nz[t] * d_nz[t]), t))
    detalle = [{
        "termino": t,
        "peso_consulta": q_nz[t],
        "peso_documento": d_nz[t],
        "producto": q_nz[t] * d_nz[t],
        "idf": float(idf_vec[vocab[t]]),
        "df": int(df_counts[vocab[t]]),
    } for t in compartidos]
    idx_sh = [vocab[t] for t in compartidos]
    q_sq = Qd[i] ** 2
    d_sq = Dd[j] ** 2
    tok_q = set(analyzer(consultas[q]["texto"]))
    tok_d = set(analyzer(corpus[rel]))
    terminos_compartidos[q] = {
        "relevante": rel,
        "n_terminos_compartidos": len(compartidos),
        "terminos": detalle,
        "suma_producto_pesos": float(sum(d["producto"] for d in detalle)),
        "suma_peso_consulta_compartido": float(sum(q_nz[t] for t in compartidos)),
        "fraccion_energia_consulta_compartida": float(q_sq[idx_sh].sum() / q_sq.sum()) if q_sq.sum() > 0 else 0.0,
        "fraccion_energia_documento_compartida": float(d_sq[idx_sh].sum() / d_sq.sum()) if d_sq.sum() > 0 else 0.0,
        "tokens_compartidos_crudos": sorted(tok_q & tok_d),
        "terminos_de_la_consulta_fuera_del_vocabulario": sorted(tok_q - set(vocab)),
        "terminos_de_la_consulta_en_vocabulario_pero_ausentes_en_el_relevante": sorted(set(q_nz) - set(d_nz)),
        "sim_coseno_tfidf": float(S_tfidf[i, j]),
        "rank_tfidf": fila_por_q[q]["rank_tfidf"],
        "rank_lsa": fila_por_q[q]["rank_lsa"],
    }

# ---------------------------------------------------------------------------
# 6) Análisis en profundidad de q5 (el fallo común) y nota sobre q4 en LSA
# ---------------------------------------------------------------------------
q5 = "q5"
i5 = Q_IDS.index(q5)
rel5 = consultas[q5]["relevante"]
j5 = DOC_IDS.index(rel5)
q5_nz = no_ceros(Qd[i5])
d5_nz = no_ceros(Dd[j5])
sh5 = [d["termino"] for d in terminos_compartidos[q5]["terminos"]]
oov5 = terminos_compartidos[q5]["terminos_de_la_consulta_fuera_del_vocabulario"]
ausentes_set = set(d5_nz) - set(q5_nz)
ausentes_d09 = sorted(ausentes_set, key=lambda t: (-d5_nz[t], t))
ejemplos_ausentes = [t for t in analyzer(corpus[rel5]) if t in ausentes_set][:4]

orden_q5_tfidf = rank_tfidf[q5]
pos5 = orden_q5_tfidf.index(rel5)
sobre_d09 = []
for d in orden_q5_tfidf[:pos5]:
    jd = DOC_IDS.index(d)
    d_nz = no_ceros(Dd[jd])
    sh = sorted(set(q5_nz) & set(d_nz), key=lambda t: (-(q5_nz[t] * d_nz[t]), t))
    sobre_d09.append({
        "doc": d,
        "sim_tfidf_con_q5": float(S_tfidf[i5, jd]),
        "terminos_compartidos_con_q5": sh,
    })

orden_q5_lsa = rank_lsa[q5]
pos5_lsa = orden_q5_lsa.index(rel5)

if sh5:
    idf_la = terminos_compartidos[q5]["terminos"][0]["idf"]
    df_la = terminos_compartidos[q5]["terminos"][0]["df"]
else:
    idf_la, df_la = None, None
frac5 = terminos_compartidos[q5]["fraccion_energia_consulta_compartida"]
sim5 = fila_por_q[q5]["sim_tfidf_relevante"]
rank5 = fila_por_q[q5]["rank_tfidf"]
sim5_lsa = fila_por_q[q5]["sim_lsa_relevante"]
rank5_lsa = fila_por_q[q5]["rank_lsa"]

if sh5:
    conclusion_q5 = (
        f"FALLO EN AMBOS MÉTODOS. q5 («{consultas[q5]['texto']}») y su relevante {rel5} comparten un único "
        f"término del vocabulario: «{sh5[0]}» (df={df_la}/10, idf={idf_la:.4f}), que aporta solo el "
        f"{100 * frac5:.1f}% de la energía TF-IDF de la consulta; fuera de ese artículo, ningún término de "
        f"{rel5} aparece en la consulta (p. ej. {', '.join(ejemplos_ausentes)}) y los términos de contenido de "
        f"la consulta ({', '.join(oov5)}) no existen en el corpus. La similitud coseno TF-IDF es {sim5:.4f} y "
        f"{rel5} queda en el puesto {rank5} (fuera del top 3), por detrás de documentos que solo comparten "
        f"palabras funcionales o «modelo/al» con q5. LSA-4D no repara el fallo (varianza explicada "
        f"{100 * var4d:.1f}%): {rel5} sigue en el puesto {rank5_lsa} (sim {sim5_lsa:.4f}) porque la relación "
        f"«determinista ↔ temperatura baja» no tiene soporte de co-ocurrencia en un corpus de 10 documentos."
    )
else:
    conclusion_q5 = (
        f"FALLO EN AMBOS MÉTODOS. q5 no comparte ningún término del vocabulario con {rel5}; "
        f"sim TF-IDF={sim5:.4f} (rank {rank5}), sim LSA={sim5_lsa:.4f} (rank {rank5_lsa})."
    )

i4 = Q_IDS.index("q4")
top3_lsa_q4 = [{
    "doc": d,
    "sim_lsa": float(S_lsa[i4, DOC_IDS.index(d)]),
    "sim_tfidf": float(S_tfidf[i4, DOC_IDS.index(d)]),
} for d in rank_lsa["q4"][:3]]
tc4 = terminos_compartidos["q4"]
terms4 = ", ".join(f"«{d['termino']}»" for d in tc4["terminos"])
nota_q4 = (
    f"LSA no crea fallos nuevos de top 3 (el único fallo en ambos métodos es q5), pero degrada q4: el relevante "
    f"d05 pasa del puesto 1 (TF-IDF) al 3 (LSA); sigue siendo acierto Hit@3 pero pierde Hit@1 "
    f"({met_tfidf['Hit@1']:.4f} → {met_lsa['Hit@1']:.4f}). En el espacio latente d01 y d10 adelantan a d05 "
    f"aunque q4 solo comparte con d05 {tc4['n_terminos_compartidos']} términos ({terms4}); la proyección 4D "
    f"captura solo el {100 * var4d:.1f}% de la varianza y mezcla documentos por patrones globales de "
    f"co-ocurrencia más que por los términos de contenido compartidos."
)

explicacion_por_consulta = {}
for q in Q_IDS:
    if q == "q5":
        explicacion_por_consulta[q] = conclusion_q5
        continue
    tc = terminos_compartidos[q]
    ts = ", ".join(f"«{d['termino']}»" for d in tc["terminos"]) or "ninguno"
    f_ = fila_por_q[q]
    explicacion_por_consulta[q] = (
        f"Comparte {tc['n_terminos_compartidos']} término(s) con {tc['relevante']} ({ts}); cubren el "
        f"{100 * tc['fraccion_energia_consulta_compartida']:.1f}% de la energía de la consulta; "
        f"sim coseno TF-IDF = {f_['sim_tfidf_relevante']:.4f}; rank TF-IDF = {f_['rank_tfidf']} "
        f"({'FALLO' if f_['fallo_tfidf'] else 'ok'}), rank LSA = {f_['rank_lsa']} "
        f"({'FALLO' if f_['fallo_lsa'] else 'ok'})."
    )

# ---------------------------------------------------------------------------
# 7) Figuras (PNG, backend Agg)
# ---------------------------------------------------------------------------
col_azul, col_naranja, col_rojo = "#4C72B0", "#DD8452", "crimson"
cols_q5 = [col_rojo if q == "q5" else col_azul for q in Q_IDS]
xpos = np.arange(len(Q_IDS))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]
ax.bar(xpos - 0.2, ranks_t, 0.4, label="TF-IDF (T1)", color=col_azul)
ax.bar(xpos + 0.2, ranks_l, 0.4, label="LSA 4D (T2)", color=col_naranja)
ax.axhline(3, color=col_rojo, ls="--", lw=1.2)
ax.text(len(Q_IDS) - 0.45, 3.25, "umbral top-3", color=col_rojo, fontsize=8, ha="right")
ax.set_xticks(xpos)
ax.set_xticklabels(Q_IDS)
ax.set_ylim(0, 7)
ax.set_ylabel("rank del documento relevante")
ax.set_title("Rank del relevante por consulta y método")
ax.legend(fontsize=8, loc="upper left")

n_sh = [terminos_compartidos[q]["n_terminos_compartidos"] for q in Q_IDS]
ax = axes[1]
ax.bar(xpos, n_sh, color=cols_q5)
for xi, v in zip(xpos, n_sh):
    ax.text(xi, v + 0.12, str(v), ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(Q_IDS)
ax.set_ylim(0, max(n_sh) + 1.4)
ax.set_ylabel("número de términos compartidos")
ax.set_title("Intersección de vocabulario consulta–relevante")

fr = [100 * terminos_compartidos[q]["fraccion_energia_consulta_compartida"] for q in Q_IDS]
ax = axes[2]
ax.bar(xpos, fr, color=cols_q5)
for xi, v in zip(xpos, fr):
    ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(Q_IDS)
ax.set_ylim(0, max(fr) + 14)
ax.set_ylabel("% de la energía TF-IDF de la consulta")
ax.set_title("Energía de la consulta cubierta por términos compartidos")

fig.suptitle("T3 — Fallos por método (relevante fuera del top 3) y términos compartidos; q5 en rojo",
             y=1.03, fontsize=12)
fig.tight_layout()
fig.savefig("T3_fallos_y_terminos.png", dpi=120, bbox_inches="tight")
plt.close(fig)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
sh5_set = set(sh5)
items_q = sorted(q5_nz.items(), key=lambda kv: kv[1])
terms_q = [t for t, _ in items_q]
vals_q = [v for _, v in items_q]
axA.barh(terms_q, vals_q, color=[col_rojo if t in sh5_set else "#9BA3AF" for t in terms_q])
axA.set_title("Consulta q5 — pesos TF-IDF (solo términos del vocabulario)")
axA.set_xlabel("peso TF-IDF")

items_d = sorted(d5_nz.items(), key=lambda kv: kv[1])
terms_d = [t for t, _ in items_d]
vals_d = [v for _, v in items_d]
axB.barh(terms_d, vals_d, color=[col_rojo if t in sh5_set else col_azul for t in terms_d])
axB.set_title(f"Documento relevante {rel5} — pesos TF-IDF (rojo = compartido con q5)")
axB.set_xlabel("peso TF-IDF")

fig.suptitle(
    f"T3 — Por qué falla q5: único término compartido con {rel5}: «{sh5[0] if sh5 else '—'}».\n"
    f"Términos de la consulta fuera del vocabulario del corpus: {', '.join(oov5)}",
    fontsize=10, y=1.05,
)
fig.tight_layout()
fig.savefig("T3_q5_terminos.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8) Tablas markdown y verificación de consistencia con T1/T2
# ---------------------------------------------------------------------------
def md_tabla_fallos():
    lineas = [
        "| Consulta | Relevante | Rank TF-IDF | ¿Fallo TF-IDF? | Rank LSA | ¿Fallo LSA? | n términos compartidos | Términos compartidos |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in filas:
        q = f["consulta"]
        tc = terminos_compartidos[q]
        ts = ", ".join(d["termino"] for d in tc["terminos"]) or "—"
        lineas.append(
            f"| {q} | {f['relevante']} | {f['rank_tfidf']} | {'**sí**' if f['fallo_tfidf'] else 'no'} | "
            f"{f['rank_lsa']} | {'**sí**' if f['fallo_lsa'] else 'no'} | {tc['n_terminos_compartidos']} | {ts} |"
        )
    return "\n".join(lineas)


def md_tabla_terminos():
    lineas = [
        "| Consulta | Relevante | Término | peso consulta | peso documento | idf | df |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in Q_IDS:
        tc = terminos_compartidos[q]
        if not tc["terminos"]:
            lineas.append(f"| {q} | {tc['relevante']} | — | — | — | — | — |")
        for d in tc["terminos"]:
            lineas.append(
                f"| {q} | {tc['relevante']} | {d['termino']} | {d['peso_consulta']:.4f} | "
                f"{d['peso_documento']:.4f} | {d['idf']:.4f} | {d['df']} |"
            )
    return "\n".join(lineas)


dif_max_t1 = 0.0
sim_file = t1.get("similitud_coseno") or {}
for i, q in enumerate(Q_IDS):
    for j, d in enumerate(DOC_IDS):
        v = (sim_file.get(q) or {}).get(d)
        if v is not None:
            dif_max_t1 = max(dif_max_t1, abs(float(v) - float(S_tfidf[i, j])))

coincide_tfidf = all(rank_tfidf[q] == rank_rec_tfidf[q] for q in Q_IDS)
coincide_lsa = all(rank_lsa[q] == rank_rec_lsa[q] for q in Q_IDS)

met_t2_file = t2.get("metricas") or {}
dif_metricas_t2 = {}
for nombre, clave in [("tfidf", "tfidf"), ("lsa", "lsa")]:
    mio = met_tfidf if nombre == "tfidf" else met_lsa
    archivo = met_t2_file.get(clave) or {}
    difs = [abs(mio[k] - float(archivo[k])) for k in ("Hit@1", "Hit@3", "MRR") if k in archivo]
    dif_metricas_t2[nombre] = max(difs) if difs else None

var_t2_file = t2.get("varianza_explicada_total_4d")
dif_var = abs(float(var_t2_file) - var4d) if var_t2_file is not None else None

# ---------------------------------------------------------------------------
# 9) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


resultados = {
    "subtarea": "T3",
    "titulo": "Análisis por consulta: fallos (relevante fuera del top 3) por método y términos compartidos consulta–documento relevante",
    "config": {
        "criterio_fallo": "el documento relevante queda fuera del top 3 del ranking del método",
        "metodos": {
            "tfidf": "TF-IDF (TfidfVectorizer por defecto) + similitud coseno (T1)",
            "lsa": "LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF de T1 (T2)",
        },
        "fuentes": {
            "rankings_T1": "entrada/T1/resultados.json",
            "rankings_T2": "entrada/T2/resultados.json",
            "tfidf_y_lsa_recalculados": "misma parametrización que T1/T2; ajuste solo sobre los 10 documentos",
        },
        "tam_vocabulario": int(len(vocab)),
        "n_documentos": len(DOC_IDS),
        "n_consultas": len(Q_IDS),
        "varianza_explicada_lsa_4d": var4d,
    },
    "resumen_fallos": {
        "tfidf": {"n_fallos": len(fallos_tfidf), "consultas_fallidas": fallos_tfidf, **met_tfidf},
        "lsa": {"n_fallos": len(fallos_lsa), "consultas_fallidas": fallos_lsa, **met_lsa},
    },
    "tabla_fallos": [
        dict(f, **{
            "n_terminos_compartidos": terminos_compartidos[f["consulta"]]["n_terminos_compartidos"],
            "terminos_compartidos": [d["termino"] for d in terminos_compartidos[f["consulta"]]["terminos"]],
        })
        for f in filas
    ],
    "tabla_fallos_markdown": md_tabla_fallos(),
    "tabla_terminos_compartidos_markdown": md_tabla_terminos(),
    "rankings_usados": {"tfidf": rank_tfidf, "lsa": rank_lsa},
    "terminos_compartidos_por_consulta": terminos_compartidos,
    "analisis_q5": {
        "consulta": consultas[q5]["texto"],
        "relevante": rel5,
        "terminos_compartidos": terminos_compartidos[q5]["terminos"],
        "unico_termino_compartido": sh5[0] if sh5 else None,
        "peso_en_consulta": terminos_compartidos[q5]["terminos"][0]["peso_consulta"] if sh5 else None,
        "peso_en_documento": terminos_compartidos[q5]["terminos"][0]["peso_documento"] if sh5 else None,
        "idf_del_termino_compartido": idf_la,
        "df_del_termino_compartido": df_la,
        "fraccion_energia_consulta_compartida": frac5,
        "terminos_consulta_fuera_del_vocabulario": oov5,
        "terminos_del_relevante_ausentes_en_la_consulta": [
            {"termino": t, "peso_documento": d5_nz[t], "idf": float(idf_vec[vocab[t]]), "df": int(df_counts[vocab[t]])}
            for t in ausentes_d09
        ],
        "ejemplos_terminos_del_relevante_ausentes": ejemplos_ausentes,
        "sim_coseno_tfidf": sim5,
        "rank_tfidf": rank5,
        "sim_coseno_lsa": sim5_lsa,
        "rank_lsa": rank5_lsa,
        "ranking_tfidf_q5": rank_tfidf[q5],
        "ranking_lsa_q5": rank_lsa[q5],
        "documentos_que_superan_al_relevante_en_tfidf": sobre_d09,
        "conclusion": conclusion_q5,
    },
    "analisis_q4_lsa": {
        "nota": nota_q4,
        "rank_tfidf": fila_por_q["q4"]["rank_tfidf"],
        "rank_lsa": fila_por_q["q4"]["rank_lsa"],
        "top3_lsa_q4": top3_lsa_q4,
        "terminos_compartidos_q4_d05": [d["termino"] for d in tc4["terminos"]],
    },
    "explicacion_por_consulta": explicacion_por_consulta,
    "verificacion": {
        "rankings_recalculados_coinciden_con_T1": bool(coincide_tfidf),
        "rankings_recalculados_coinciden_con_T2": bool(coincide_lsa),
        "coincidencia_por_consulta": {
            q: {"tfidf": rank_tfidf[q] == rank_rec_tfidf[q], "lsa": rank_lsa[q] == rank_rec_lsa[q]}
            for q in Q_IDS
        },
        "dif_max_sim_tfidf_vs_T1": dif_max_t1,
        "dif_max_metricas_vs_T2": dif_metricas_t2,
        "dif_varianza_explicada_vs_T2": dif_var,
        "metricas_recalculadas": {"tfidf": met_tfidf, "lsa": met_lsa},
        "suma_producto_pesos_igual_a_coseno": {
            q: abs(terminos_compartidos[q]["suma_producto_pesos"] - terminos_compartidos[q]["sim_coseno_tfidf"])
            for q in Q_IDS
        },
        "archivos_previos_cargados": {"T1": bool(t1), "T2": bool(t2)},
    },
    "figuras": ["T3_fallos_y_terminos.png", "T3_q5_terminos.png"],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(jsonable(resultados), f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 10) Resumen breve
# ---------------------------------------------------------------------------
print("=" * 78)
print("T3 — Fallos por método (relevante fuera del top 3) y términos compartidos")
print("=" * 78)
df_tab = pd.DataFrame(filas)
print(df_tab[["consulta", "relevante", "rank_tfidf", "fallo_tfidf", "rank_lsa", "fallo_lsa"]].to_string(index=False))
print(f"\nTF-IDF: fallos={fallos_tfidf} | Hit@1={met_tfidf['Hit@1']:.4f} "
      f"Hit@3={met_tfidf['Hit@3']:.4f} MRR={met_tfidf['MRR']:.4f}")
print(f"LSA   : fallos={fallos_lsa} | Hit@1={met_lsa['Hit@1']:.4f} "
      f"Hit@3={met_lsa['Hit@3']:.4f} MRR={met_lsa['MRR']:.4f}")
print(f"\nq5 vs {rel5}: términos compartidos = {sh5} "
      f"(idf={idf_la if idf_la is not None else '—'}, df={df_la if df_la is not None else '—'}/10)")
print(f"       rank TF-IDF={rank5}, rank LSA={rank5_lsa}; fuera de vocabulario: {oov5}")
print(f"       {conclusion_q5}")
print("\nArchivos escritos: resultados.json, T3_fallos_y_terminos.png, T3_q5_terminos.png")
