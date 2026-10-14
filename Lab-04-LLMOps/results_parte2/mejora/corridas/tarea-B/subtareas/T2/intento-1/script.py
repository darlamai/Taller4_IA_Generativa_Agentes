import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================
# T2: Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución
#     de T = 1 obtenida en T1 (Parte 1).
# Criterio del enunciado: ordenar por probabilidad descendente,
# conservar el conjunto MÍNIMO cuya probabilidad acumulada ALCANZA
# p = 0.9 y renormalizar las probabilidades de los supervivientes.
# ==================================================================

P = 0.9
TOKENS_ENUNCIADO = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)


def softmax_T(z, T):
    """Softmax con temperatura, numéricamente estable (resta el máximo)."""
    z = np.asarray(z, dtype=float)
    z_stable = (z - z.max()) / T
    e = np.exp(z_stable)
    return e / e.sum()


# ------------------------------------------------------------------
# 1) Distribución de entrada: la de T = 1 de la subtarea T1
# ------------------------------------------------------------------
ruta_t1 = Path("entrada/T1/resultados.json")
if ruta_t1.exists():
    with open(ruta_t1, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    probs_t1 = np.array(t1["distribuciones"]["T=1.0"], dtype=float)
    tokens = list(t1["tokens"])
    fuente = str(ruta_t1)
else:
    # Respaldo: recalcular con la softmax estable del enunciado
    probs_t1 = softmax_T(LOGITS, 1.0)
    tokens = list(TOKENS_ENUNCIADO)
    fuente = "recalculada con softmax estable (T=1)"

# Verificación de consistencia con la softmax T=1
assert np.allclose(probs_t1, softmax_T(LOGITS, 1.0)), (
    "La distribución de T1 no coincide con la softmax T=1 recalculada"
)

# ------------------------------------------------------------------
# 2) Ordenar tokens por probabilidad descendente
# ------------------------------------------------------------------
orden = np.argsort(-probs_t1, kind="stable")
tokens_ord = [tokens[i] for i in orden]
probs_ord = probs_t1[orden]

# ------------------------------------------------------------------
# 3) Probabilidad acumulada y corte top-p:
#    conjunto mínimo (prefijo en orden descendente) con acumulada >= p
# ------------------------------------------------------------------
cum = np.cumsum(probs_ord)
k = int(np.searchsorted(cum, P, side="left"))  # primer índice con cum >= p
n_sup = k + 1

tokens_sup = tokens_ord[:n_sup]
probs_sup = probs_ord[:n_sup]
cum_sup = cum[:n_sup]
acumulada_corte = float(cum[k])  # prob. acumulada del conjunto superviviente

tokens_desc = tokens_ord[n_sup:]
probs_desc = probs_ord[n_sup:]

# ------------------------------------------------------------------
# 4) Renormalizar las probabilidades de los supervivientes
# ------------------------------------------------------------------
probs_renorm = probs_sup / acumulada_corte
suma_renorm = float(probs_renorm.sum())

# ------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Muestreo de núcleo (top-p) con p=0.9 sobre la distribución de T=1 "
        "de la Parte 1: conjunto mínimo de tokens cuya probabilidad acumulada "
        "alcanza p, seguido de renormalización."
    ),
    "p": P,
    "temperatura": 1.0,
    "fuente_distribucion": fuente,
    "tokens": tokens,
    "distribucion_T1_entrada": probs_t1.tolist(),
    "orden_descendente": tokens_ord,
    "probs_ordenadas": probs_ord.tolist(),
    "prob_acumulada_ordenada": cum.tolist(),
    "tokens_supervivientes": tokens_sup,
    "prob_acumulada_antes_del_corte": acumulada_corte,
    "prob_acumulada_por_superviviente": [
        {"token": tok, "prob": float(pr), "acumulada": float(ac)}
        for tok, pr, ac in zip(tokens_sup, probs_sup, cum_sup)
    ],
    "distribucion_renormalizada": {
        tok: float(pr) for tok, pr in zip(tokens_sup, probs_renorm)
    },
    "distribucion_renormalizada_en_orden_original": {
        tok: float(probs_renorm[tokens_sup.index(tok)])
        for tok in tokens if tok in tokens_sup
    },
    "suma_distribucion_renormalizada": suma_renorm,
    "tokens_descartados": tokens_desc,
    "probs_descartados": probs_desc.tolist(),
    "masa_descartada": float(probs_desc.sum()),
    "tabla_4_decimales": {
        "columnas": ["token", "prob_T1", "acumulada", "superviviente",
                     "prob_renormalizada"],
        "filas": [
            [
                tok,
                f"{pr:.4f}",
                f"{ac:.4f}",
                "sí" if i < n_sup else "no",
                f"{probs_renorm[i]:.4f}" if i < n_sup else "-",
            ]
            for i, (tok, pr, ac) in enumerate(zip(tokens_ord, probs_ord, cum))
        ],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# Figura: corte top-p y distribución renormalizada
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
x = np.arange(len(tokens_ord))
colores = ["#2a9d8f" if i < n_sup else "#d3d3d3" for i in range(len(tokens_ord))]
ax.bar(x, probs_ord, color=colores, edgecolor="black", linewidth=0.5,
       label="prob. T=1")
ax.plot(x, cum, "o--", color="#e76f51", label="acumulada")
ax.axhline(P, color="red", linestyle=":", linewidth=1.2, label=f"p = {P}")
ax.set_xticks(x)
ax.set_xticklabels(tokens_ord)
ax.set_ylim(0, 1.08)
ax.set_ylabel("probabilidad")
ax.set_title("Top-p (p=0.9) sobre la distribución T=1\nverde = supervivientes del núcleo")
ax.legend(fontsize=8)

ax = axes[1]
ax.bar(tokens_sup, probs_renorm, color="#2a9d8f", edgecolor="black", linewidth=0.5)
for i, pr in enumerate(probs_renorm):
    ax.text(i, pr + 0.012, f"{pr:.4f}", ha="center", fontsize=9)
ax.set_ylim(0, max(probs_renorm) * 1.18)
ax.set_ylabel("probabilidad renormalizada")
ax.set_title(f"Distribución renormalizada (suma = {suma_renorm:.6f})")

fig.tight_layout()
fig.savefig("T2_top_p.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------
print("=== T2: Top-p (nucleus sampling) con p = 0.9 sobre la distribución T=1 ===")
print(f"Fuente de la distribución: {fuente}")
print("Distribución T=1 de entrada:")
for tok, pr in zip(tokens, probs_t1):
    print(f"  {tok}: {pr:.6f}")
print("Orden descendente con probabilidad acumulada:")
for tok, pr, ac in zip(tokens_ord, probs_ord, cum):
    marca = "  <- corte aquí" if tok == tokens_sup[-1] else ""
    print(f"  {tok}: p={pr:.6f}  acum={ac:.6f}{marca}")
print(f"Tokens supervivientes (conjunto mínimo con acumulada >= {P}): {tokens_sup}")
print(f"Probabilidad acumulada antes del corte: {acumulada_corte:.6f}")
print("Distribución renormalizada de los supervivientes:")
for tok, pr in zip(tokens_sup, probs_renorm):
    print(f"  {tok}: {pr:.6f}")
print(f"Suma de la distribución renormalizada: {suma_renorm:.12f}")
print(f"Tokens descartados: {tokens_desc} (masa descartada = {probs_desc.sum():.6f})")
print("Archivos escritos: resultados.json, T2_top_p.png")
