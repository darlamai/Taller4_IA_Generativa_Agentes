# -*- coding: utf-8 -*-
"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1 (Parte 1).

Pasos:
1) Carga la distribución de T = 1 desde entrada/T1/resultados.json (subtarea previa T1)
   y la verifica recalculando la softmax con temperatura de forma numéricamente estable.
2) Aplica top-p con p = 0.9: ordena los tokens por probabilidad descendente, conserva el
   conjunto MÍNIMO cuya probabilidad acumulada alcanza 0.9 y renormaliza (los tokens
   fuera del núcleo quedan con probabilidad 0).
3) Escribe resultados.json (contrato de la subtarea) y la figura top_p_nucleo.png.

Ejecución: python T2_top_p.py  (desde la carpeta de trabajo de la tarea)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parámetros fijos del enunciado
# ----------------------------------------------------------------------
P = 0.9
T = 1.0
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)


def softmax_con_temperatura(logits: np.ndarray, temperatura: float) -> np.ndarray:
    """Softmax con temperatura, numéricamente estable (resta el máximo antes de exponenciar)."""
    z = np.asarray(logits, dtype=float) / float(temperatura)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------
# 1) Distribución de T = 1 obtenida en la Parte 1 (subtarea T1)
# ----------------------------------------------------------------------
ruta_t1 = Path("entrada") / "T1" / "resultados.json"
if ruta_t1.exists():
    with ruta_t1.open("r", encoding="utf-8") as f:
        res_t1 = json.load(f)
    p_T1 = np.array(res_t1["distribuciones_sin_redondear"]["T=1.0"], dtype=float)
    fuente = "entrada/T1/resultados.json (distribucion de T=1 de la Parte 1)"
else:  # respaldo: recalcular si el archivo de T1 no estuviera disponible
    p_T1 = softmax_con_temperatura(LOGITS, T)
    fuente = "recalculada con softmax estable (entrada/T1 no disponible)"

p_verif = softmax_con_temperatura(LOGITS, T)
coincide_con_softmax = bool(np.allclose(p_T1, p_verif, rtol=0.0, atol=1e-12))

# ----------------------------------------------------------------------
# 2) Top-p (nucleus sampling) con p = 0.9
# ----------------------------------------------------------------------
orden = np.argsort(-p_T1, kind="stable")       # índices por probabilidad descendente
probs_ord = p_T1[orden]
acum_ord = np.cumsum(probs_ord)

alcanza = acum_ord >= P                        # primer rank cuya acumulada alcanza p
k = int(np.argmax(alcanza)) + 1                # tamaño del núcleo (conjunto mínimo)
nucleo_idx = orden[:k]
fuera_idx = orden[k:]
nucleo_set = set(nucleo_idx.tolist())

acum_nucleo = float(acum_ord[k - 1])                          # acumulada del núcleo (>= 0.9)
acum_sin_ultimo = float(acum_ord[k - 2]) if k >= 2 else 0.0   # para verificar minimalidad

# Renormalización: los tokens fuera del núcleo quedan con probabilidad 0
p_renorm = np.zeros_like(p_T1)
p_renorm[nucleo_idx] = p_T1[nucleo_idx] / acum_nucleo

tokens_nucleo = [TOKENS[i] for i in nucleo_idx]
tokens_fuera = [TOKENS[i] for i in fuera_idx]

dist_T1_dict = {TOKENS[i]: float(p_T1[i]) for i in range(len(TOKENS))}
dist_renorm_dict = {TOKENS[i]: float(p_renorm[i]) for i in range(len(TOKENS))}

# Tabla del procedimiento (orden descendente), con y sin redondeo
tabla_ordenada = []
for rank, idx in enumerate(orden, start=1):
    tabla_ordenada.append({
        "rank": rank,
        "token": TOKENS[idx],
        "p_T1": float(p_T1[idx]),
        "acumulada": float(acum_ord[rank - 1]),
        "en_nucleo": bool(idx in nucleo_set),
        "p_renormalizada": float(p_renorm[idx]),
    })

tabla_4_decimales = [
    {
        "rank": fila["rank"],
        "token": fila["token"],
        "p_T1": f"{fila['p_T1']:.4f}",
        "acumulada": f"{fila['acumulada']:.4f}",
        "en_nucleo": "si" if fila["en_nucleo"] else "no",
        "p_renormalizada": f"{fila['p_renormalizada']:.4f}",
    }
    for fila in tabla_ordenada
]

# ----------------------------------------------------------------------
# 3) Verificaciones del criterio de éxito
# ----------------------------------------------------------------------
suma_renorm = float(p_renorm.sum())
verificacion = {
    "distribucion_T1_coincide_con_softmax_estable": coincide_con_softmax,
    "suma_distribucion_renormalizada": suma_renorm,
    "renormalizada_suma_1": bool(np.isclose(suma_renorm, 1.0, rtol=0.0, atol=1e-12)),
    "nucleo_acumula_al_menos_p": bool(acum_nucleo >= P),
    "acumulada_nucleo": acum_nucleo,
    "acumulada_sin_el_ultimo_token_del_nucleo": acum_sin_ultimo,
    "conjunto_es_minimo": bool(acum_sin_ultimo < P),
    "tokens_fuera_quedan_en_cero": (bool(np.all(p_renorm[fuera_idx] == 0.0))
                                    if len(fuera_idx) > 0 else True),
}

# ----------------------------------------------------------------------
# 4) Figura
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
colores = ["#2a9d8f" if i in nucleo_set else "#e76f51" for i in range(len(TOKENS))]
x = np.arange(len(TOKENS))

ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(x, p_T1, color=colores, edgecolor="black", linewidth=0.5)
ax1.set_title("Distribución original (T = 1, Parte 1)")
ax1.set_xticks(x)
ax1.set_xticklabels(TOKENS)
ax1.set_ylabel("probabilidad")
ax1.set_ylim(0, float(p_T1.max()) * 1.18)
for i, v in enumerate(p_T1):
    ax1.text(i, v + 0.006, f"{v:.4f}", ha="center", fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(x, p_renorm, color=colores, edgecolor="black", linewidth=0.5)
ax2.set_title(f"Distribución renormalizada tras top-p (p = {P})")
ax2.set_xticks(x)
ax2.set_xticklabels(TOKENS)
ax2.set_ylim(0, float(p_renorm.max()) * 1.18)
for i, v in enumerate(p_renorm):
    ax2.text(i, v + 0.006, f"{v:.4f}", ha="center", fontsize=8)

ax3 = fig.add_subplot(gs[1, :])
ranks = np.arange(1, len(TOKENS) + 1)
ax3.step(ranks, acum_ord, where="post", color="#264653", marker="o",
         label="acumulada (orden descendente)")
ax3.axhline(P, color="#e76f51", linestyle="--", label=f"umbral p = {P}")
ax3.axvline(k, color="#2a9d8f", linestyle=":", label=f"borde del núcleo (k = {k})")
ax3.plot([k], [acum_nucleo], "o", color="#2a9d8f", markersize=9)
ax3.annotate(f"núcleo {{{', '.join(tokens_nucleo)}}}\nacumulada = {acum_nucleo:.4f}",
             xy=(k, acum_nucleo), xytext=(k + 0.4, acum_nucleo - 0.20),
             arrowprops=dict(arrowstyle="->", color="#2a9d8f"))
ax3.set_xticks(ranks)
ax3.set_xticklabels([f"{rank}: {TOKENS[idx]}" for rank, idx in enumerate(orden, start=1)])
ax3.set_xlabel("tokens ordenados por probabilidad descendente")
ax3.set_ylabel("probabilidad acumulada")
ax3.set_ylim(0, 1.05)
ax3.set_title("Construcción del conjunto mínimo con acumulada ≥ 0.9")
ax3.legend(loc="lower right")
ax3.grid(alpha=0.3)

fig.suptitle("T2 — Muestreo de núcleo (top-p, p = 0.9) sobre la distribución de T = 1", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("top_p_nucleo.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5) Contrato: resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2 - Muestreo de nucleo (top-p) con p=0.9 sobre la distribucion de T=1",
    "p": P,
    "temperatura": T,
    "tokens": TOKENS,
    "logits": LOGITS.tolist(),
    "fuente_distribucion_T1": fuente,
    "distribucion_T1_sin_redondear": dist_T1_dict,
    "distribucion_T1_lista": p_T1.tolist(),
    "orden_descendente": [TOKENS[i] for i in orden],
    "probs_ordenadas_desc_sin_redondear": probs_ord.tolist(),
    "acumulada_ordenada_desc_sin_redondear": acum_ord.tolist(),
    "tokens_nucleo": tokens_nucleo,
    "num_tokens_nucleo": k,
    "probabilidad_acumulada_nucleo": acum_nucleo,
    "tokens_fuera_nucleo": tokens_fuera,
    "probabilidad_fuera_nucleo": float(1.0 - acum_nucleo),
    "distribucion_renormalizada_sin_redondear": dist_renorm_dict,
    "distribucion_renormalizada_lista": p_renorm.tolist(),
    "distribucion_renormalizada_4_decimales": {t: f"{v:.4f}" for t, v in dist_renorm_dict.items()},
    "tabla_ordenada_sin_redondear": tabla_ordenada,
    "tabla_4_decimales": tabla_4_decimales,
    "verificacion": verificacion,
    "figura": "top_p_nucleo.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Resumen
# ----------------------------------------------------------------------
print("=" * 72)
print("T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1")
print("=" * 72)
print(f"Fuente de la distribución T=1: {fuente}")
print("Distribución T=1: " + ", ".join(f"{t}={p_T1[i]:.4f}" for i, t in enumerate(TOKENS)))
print(f"Orden descendente: {' > '.join(TOKENS[i] for i in orden)}")
print("Acumuladas:        " + ", ".join(f"{a:.4f}" for a in acum_ord))
print(f"Núcleo (conjunto mínimo con acumulada >= 0.9): {{{', '.join(tokens_nucleo)}}}")
print(f"Probabilidad acumulada del núcleo: {acum_nucleo:.6f} "
      f"(sin el último token: {acum_sin_ultimo:.6f} < 0.9 -> el conjunto es mínimo)")
print(f"Tokens fuera del núcleo (probabilidad 0): {tokens_fuera if tokens_fuera else 'ninguno'}")
print("Distribución renormalizada: "
      + ", ".join(f"{t}={p_renorm[i]:.4f}" for i, t in enumerate(TOKENS)))
print(f"Suma de la distribución renormalizada: {suma_renorm:.6f}")
print(f"Verificación: conjunto mínimo = {verificacion['conjunto_es_minimo']}, "
      f"suma 1 = {verificacion['renormalizada_suma_1']}, "
      f"fuera en cero = {verificacion['tokens_fuera_quedan_en_cero']}")
print("Archivos escritos: resultados.json, top_p_nucleo.png")
