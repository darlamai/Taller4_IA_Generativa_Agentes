# -*- coding: utf-8 -*-
"""
T3 — Comprobación empírica del muestreo de núcleo (top-p, p = 0.9).

Toma 10,000 muestras de la distribución renormalizada de la Parte 2 (resultado de T2)
usando numpy.random.default_rng(0), compara las frecuencias observadas con las
probabilidades teóricas en un gráfico de barras y calcula la divergencia KL
(en bits) de la distribución empírica respecto de la teórica:

    KL(emp || teo) = sum_i emp_i * log2(emp_i / teo_i)

Salidas:
  - resultados.json (contrato de la subtarea)
  - T3_frecuencias_observadas_vs_teoricas.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKENS = [f"t{i}" for i in range(6)]
N_MUESTRAS = 10_000
SEMILLA = 0
P_TOP = 0.9

# ------------------------------------------------------------------ entrada
# Fuente principal: resultado de T2. Respaldo: recalcular desde los logits del enunciado.
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        res_t2 = json.load(f)
    dist_teo_dict = res_t2["distribucion_renormalizada_vocabulario_completo"]
    p_teo = np.array([dist_teo_dict[t] for t in TOKENS], dtype=float)
    nucleo = list(res_t2["tokens_supervivientes"])
    fuente = "entrada/T2/resultados.json -> distribucion_renormalizada_vocabulario_completo"
else:
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()                      # softmax T=1 numéricamente estable
    p1 = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p1)                        # descendente
    acum = np.cumsum(p1[orden])
    k = int(np.searchsorted(acum, P_TOP, side="left") + 1)  # conjunto mínimo que alcanza p
    idx_nuc = np.sort(orden[:k])
    p_nuc = p1[idx_nuc] / p1[idx_nuc].sum()        # renormalización
    p_teo = np.zeros(len(TOKENS), dtype=float)
    p_teo[idx_nuc] = p_nuc
    nucleo = [TOKENS[i] for i in idx_nuc]
    fuente = "recalculada desde logits (softmax T=1 + top-p p=0.9)"

en_nucleo = np.array([t in nucleo for t in TOKENS])

# ------------------------------------------------------------------ muestreo
rng = np.random.default_rng(SEMILLA)
# Normalización numérica (~1e-16) solo para que rng.choice acepte p; no altera la distribución.
p_muestreo = p_teo / p_teo.sum()
muestras = rng.choice(len(TOKENS), size=N_MUESTRAS, p=p_muestreo)

conteos = np.bincount(muestras, minlength=len(TOKENS))
f_obs = conteos / N_MUESTRAS

# ------------------------------------------------------------------ divergencia KL (bits)
# KL(emp || teo) = sum_i emp_i * log2(emp_i / teo_i).
# Convención: términos con emp_i = 0 aportan 0 (aquí solo t4, t5, con teo_i = 0 también).
mask = f_obs > 0
kl_bits = float(np.sum(f_obs[mask] * np.log2(f_obs[mask] / p_teo[mask])))

contrib_kl = {
    t: (float(fq * np.log2(fq / p)) if fq > 0 else 0.0)
    for t, fq, p in zip(TOKENS, f_obs, p_teo)
}

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(figsize=(9.5, 5.5))
x = np.arange(len(TOKENS))
w = 0.38
b1 = ax.bar(x - w / 2, f_obs, width=w, color="#4C72B0", edgecolor="black", linewidth=0.5,
            label=f"Frecuencia observada ({N_MUESTRAS:,} muestras, semilla {SEMILLA})")
b2 = ax.bar(x + w / 2, p_teo, width=w, color="#DD8452", edgecolor="black", linewidth=0.5,
            label="Probabilidad teórica (top-p renormalizada)")
for barras in (b1, b2):
    for rect in barras:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    (rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.axvline(len(nucleo) - 0.5, color="gray", linestyle="--", linewidth=1)
ax.text(len(nucleo) - 0.5 + 0.1, max(f_obs.max(), p_teo.max()) * 0.97,
        "fuera del núcleo\n(p = 0)", fontsize=8, color="gray", va="top")

ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_ylim(0, max(f_obs.max(), p_teo.max()) * 1.18)
ax.set_title(
    "Parte 3 — Comprobación empírica del muestreo de núcleo (top-p, p = 0.9)\n"
    f"Frecuencias observadas vs. probabilidades teóricas | n = {N_MUESTRAS:,}, "
    f"semilla {SEMILLA} | KL(emp||teo) = {kl_bits:.6f} bits"
)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("T3_frecuencias_observadas_vs_teoricas.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ resultados.json
resultados = {
    "subtarea": ("T3 — Comprobación empírica: 10,000 muestras de la distribución renormalizada "
                 "de la Parte 2 (top-p, p=0.9) con numpy.random.default_rng(0); comparación "
                 "frecuencias observadas vs. probabilidades teóricas y divergencia KL en bits"),
    "n_muestras": N_MUESTRAS,
    "semilla": SEMILLA,
    "generador": "numpy.random.default_rng(0)",
    "p_top": P_TOP,
    "fuente_distribucion_entrada": fuente,
    "tokens": TOKENS,
    "tokens_en_nucleo": nucleo,
    "tokens_fuera_del_nucleo": [t for t in TOKENS if t not in nucleo],
    "distribucion_teorica": {t: float(p) for t, p in zip(TOKENS, p_teo)},
    "conteos_observados": {t: int(c) for t, c in zip(TOKENS, conteos)},
    "frecuencias_observadas": {t: float(fq) for t, fq in zip(TOKENS, f_obs)},
    "divergencia_KL_bits": kl_bits,
    "contribuciones_KL_bits_por_token": contrib_kl,
    "figura": "T3_frecuencias_observadas_vs_teoricas.png",
    "verificaciones": {
        "suma_distribucion_teorica": float(p_teo.sum()),
        "suma_frecuencias_observadas": float(f_obs.sum()),
        "total_conteos": int(conteos.sum()),
        "tokens_fuera_del_nucleo_nunca_muestreados": bool(conteos[~en_nucleo].sum() == 0),
        "kl_no_negativa": bool(kl_bits >= 0.0),
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ resumen
print("T3 — Comprobación empírica (top-p p=0.9, n=10,000, semilla 0)")
print(f"Distribución teórica obtenida de: {fuente}")
print(f"{'token':<6}{'p_teorica':>12}{'conteo':>9}{'f_observada':>13}")
for t, p, c, fq in zip(TOKENS, p_teo, conteos, f_obs):
    print(f"{t:<6}{p:>12.6f}{c:>9d}{fq:>13.5f}")
print(f"\nDivergencia KL(empírica || teórica) = {kl_bits:.8f} bits  (valor pequeño, cercano a 0)")
print("Figura guardada: T3_frecuencias_observadas_vs_teoricas.png")
print("Resultados escritos: resultados.json")
