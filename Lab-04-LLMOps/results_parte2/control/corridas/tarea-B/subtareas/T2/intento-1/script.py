# -*- coding: utf-8 -*-
"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1.

Entrada : entrada/T1/resultados.json  (distribución softmax con T=1 de la Parte 1)
Salidas : resultados.json             (contrato de la subtarea)
          T2_top_p_renormalizada.png  (figura comparativa)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P_UMBRAL = 0.9
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])


# ----------------------------------------------------------------------
# 1) Distribución de T = 1 de la Parte 1 (se lee de entrada/T1; si no
#    estuviera disponible, se recalcula con softmax estable).
# ----------------------------------------------------------------------
def softmax_con_temperatura(logits, T):
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - z.max()  # estabilidad numérica
    e = np.exp(z)
    return e / e.sum()


ruta_t1 = Path("entrada") / "T1" / "resultados.json"
fuente = "recalculada con softmax(T=1) a partir de los logits del enunciado"
if ruta_t1.exists():
    with open(ruta_t1, "r", encoding="utf-8") as f:
        res_t1 = json.load(f)
    p_t1 = np.array(res_t1["distribuciones"]["T=1.0"], dtype=float)
    fuente = "entrada/T1/resultados.json -> distribuciones['T=1.0']"
    # Verificación: debe coincidir con el softmax numéricamente estable
    assert np.allclose(p_t1, softmax_con_temperatura(LOGITS, 1.0), atol=1e-12)
else:
    p_t1 = softmax_con_temperatura(LOGITS, 1.0)

# ----------------------------------------------------------------------
# 2) Top-p: ordenar descendente, conjunto mínimo con acumulada >= p,
#    renormalizar las probabilidades del núcleo.
# ----------------------------------------------------------------------
orden = np.argsort(-p_t1, kind="stable")            # índices de mayor a menor
tokens_ord = [TOKENS[i] for i in orden]
probs_ord = p_t1[orden]
acum_ord = np.cumsum(probs_ord)

# k = número mínimo de tokens cuya probabilidad acumulada alcanza p = 0.9
k = int(np.searchsorted(acum_ord, P_UMBRAL, side="left")) + 1
idx_nucleo = orden[:k]
nucleo_set = set(idx_nucleo.tolist())
tokens_nucleo = [TOKENS[i] for i in idx_nucleo]
probs_nucleo = p_t1[idx_nucleo]
acum_nucleo = float(acum_ord[k - 1])

# Verificación de minimalidad: con k-1 tokens NO se alcanza p
assert acum_nucleo >= P_UMBRAL
assert (k == 1) or (float(acum_ord[k - 2]) < P_UMBRAL)

# Renormalización
probs_renorm_nucleo = probs_nucleo / probs_nucleo.sum()
p_renorm = np.zeros_like(p_t1)
p_renorm[idx_nucleo] = probs_renorm_nucleo

# ----------------------------------------------------------------------
# 3) Verificaciones del criterio de éxito
# ----------------------------------------------------------------------
suma_renorm = float(p_renorm.sum())
tokens_fuera = [t for i, t in enumerate(TOKENS) if i not in nucleo_set]
checks = {
    "suma_de_la_renormalizada_es_1": bool(abs(suma_renorm - 1.0) < 1e-12),
    "acumulada_del_nucleo_alcanza_p": bool(acum_nucleo >= P_UMBRAL),
    "es_el_conjunto_minimo": bool((k == 1) or (float(acum_ord[k - 2]) < P_UMBRAL)),
    "tokens_fuera_del_nucleo_con_prob_0": bool(
        all(p_renorm[i] == 0.0 for i in range(len(TOKENS)) if i not in nucleo_set)
    ),
    "todas_las_probs_son_no_negativas": bool(np.all(p_renorm >= 0.0)),
}
assert all(checks.values()), checks

# ----------------------------------------------------------------------
# 4) Tabla ordenada y resultados.json (contrato)
# ----------------------------------------------------------------------
tabla = []
for pos, i in enumerate(orden):
    tabla.append({
        "token": TOKENS[i],
        "probabilidad_T1": float(p_t1[i]),
        "acumulada": float(acum_ord[pos]),
        "en_nucleo": bool(i in nucleo_set),
        "probabilidad_renormalizada": float(p_renorm[i]),
    })

resultados = {
    "subtarea": "T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T=1",
    "p": P_UMBRAL,
    "fuente_distribucion_entrada": fuente,
    "distribucion_entrada_T1": {t: float(v) for t, v in zip(TOKENS, p_t1)},
    "tokens_ordenados_descendente": tokens_ord,
    "probabilidades_ordenadas": [float(v) for v in probs_ord],
    "acumuladas_ordenadas": [float(v) for v in acum_ord],
    "tokens_supervivientes": tokens_nucleo,
    "num_tokens_en_nucleo": k,
    "probabilidad_acumulada_nucleo": acum_nucleo,
    "probabilidades_nucleo_antes_de_renormalizar": {
        t: float(v) for t, v in zip(tokens_nucleo, probs_nucleo)
    },
    "distribucion_renormalizada_nucleo": {
        t: float(v) for t, v in zip(tokens_nucleo, probs_renorm_nucleo)
    },
    "distribucion_renormalizada_vocabulario_completo": {
        t: float(v) for t, v in zip(TOKENS, p_renorm)
    },
    "tokens_fuera_del_nucleo": tokens_fuera,
    "suma_distribucion_renormalizada": suma_renorm,
    "verificaciones": checks,
    "tabla_ordenada": tabla,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 5) Figura: distribución T=1 original vs top-p renormalizada
# ----------------------------------------------------------------------
x = np.arange(len(TOKENS))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w / 2, p_t1, width=w, color="#4C72B0", label="Softmax T=1 (original)")
ax.bar(x + w / 2, p_renorm, width=w, color="#DD8452", label="Top-p p=0.9 (renormalizada)")
for i in range(len(TOKENS)):
    ax.text(x[i] - w / 2, p_t1[i] + 0.008, f"{p_t1[i]:.4f}", ha="center", fontsize=8)
    if p_renorm[i] > 0:
        ax.text(x[i] + w / 2, p_renorm[i] + 0.008, f"{p_renorm[i]:.4f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_ylabel("Probabilidad")
ax.set_ylim(0, 1.05 * max(p_t1.max(), p_renorm.max()) + 0.05)
ax.set_title("Top-p (nucleus sampling), p = 0.9 — núcleo: " + ", ".join(tokens_nucleo))
ax.legend()
fig.tight_layout()
fig.savefig("T2_top_p_renormalizada.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 6) Resumen
# ----------------------------------------------------------------------
print("=" * 64)
print("T2 — Top-p con p = 0.9 sobre la distribución de T = 1")
print("=" * 64)
print("Distribución de entrada (T=1):")
for t, v in zip(TOKENS, p_t1):
    print(f"  {t}: {v:.6f}")
print(f"\nOrden descendente: {tokens_ord}")
print("Acumuladas: " + ", ".join(f"{a:.4f}" for a in acum_ord))
print(f"\nNúcleo (mínimo conjunto con acumulada >= 0.9): {tokens_nucleo}")
print(f"  Tokens conservados: {k} | Acumulada del núcleo: {acum_nucleo:.6f}")
print(f"  Tokens fuera del núcleo (prob. 0): {tokens_fuera}")
print("\nDistribución renormalizada (suma = {:.12f}):".format(suma_renorm))
for t, v in zip(TOKENS, p_renorm):
    print(f"  {t}: {v:.6f}")
print("\nArchivos escritos: resultados.json, T2_top_p_renormalizada.png")
