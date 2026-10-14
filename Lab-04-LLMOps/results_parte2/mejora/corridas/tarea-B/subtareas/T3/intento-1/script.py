# -*- coding: utf-8 -*-
"""
T3 — Comprobación empírica (Tarea B, MMIA 6013)
Toma 10 000 muestras de la distribución renormalizada de T2 (top-p p=0.9 sobre
la distribución de T=1) con numpy.random.default_rng(0), grafica las frecuencias
observadas frente a las probabilidades teóricas y calcula la divergencia KL
(en bits) de la distribución empírica respecto de la teórica.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Parámetros fijados por el enunciado
# ------------------------------------------------------------------
N_MUESTRAS = 10_000
SEMILLA = 0
P_TOP = 0.9
TEMPERATURA = 1.0

# ------------------------------------------------------------------
# 1) Cargar la distribución renormalizada de T2
# ------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    dist_dict = dict(t2["distribucion_renormalizada_en_orden_original"])
else:
    # Respaldo determinista: recalcular exactamente como en T2
    # (softmax con T=1, numéricamente estable, + top-p con p=0.9)
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    p_t1 = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p_t1)                      # descendente
    acum = np.cumsum(p_t1[orden])
    k = int(np.searchsorted(acum, P_TOP) + 1)      # conjunto mínimo que alcanza p
    supervivientes = np.sort(orden[:k])
    dist_dict = {f"t{i}": float(p_t1[i] / p_t1[supervivientes].sum())
                 for i in supervivientes}

tokens = list(dist_dict.keys())
p_teo = np.array([dist_dict[t] for t in tokens], dtype=float)
p_teo = p_teo / p_teo.sum()  # seguridad numérica (debe sumar 1)

# ------------------------------------------------------------------
# 2) Muestreo: 10 000 muestras con numpy.random.default_rng(0)
# ------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA)
muestras = rng.choice(len(tokens), size=N_MUESTRAS, p=p_teo)

conteos = np.bincount(muestras, minlength=len(tokens))
freq_obs = conteos / N_MUESTRAS

# ------------------------------------------------------------------
# 3) Divergencia KL en bits: KL(empírica || teórica)
#    KL = Σ_i q_i * log2(q_i / p_i), con 0·log(0/p) := 0
# ------------------------------------------------------------------
mask = freq_obs > 0
kl_bits = float(np.sum(freq_obs[mask] * np.log2(freq_obs[mask] / p_teo[mask])))
kl_nats = float(kl_bits * np.log(2.0))

# ------------------------------------------------------------------
# 4) Diagrama de barras: frecuencias observadas vs. probabilidades teóricas
# ------------------------------------------------------------------
x = np.arange(len(tokens))
ancho = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - ancho / 2, freq_obs, ancho, color="#1f77b4",
            label="Frecuencia observada (10 000 muestras)")
b2 = ax.bar(x + ancho / 2, p_teo, ancho, color="#ff7f0e",
            label="Probabilidad teórica (top-p p=0.9 renormalizada)")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia relativa")
ax.set_title("T3: Frecuencias observadas vs. probabilidades teóricas\n"
             f"(muestreo de la distribución renormalizada top-p, T={TEMPERATURA}, p={P_TOP})")
ax.set_ylim(0, max(freq_obs.max(), p_teo.max()) * 1.18)
ax.legend()
plt.tight_layout()
plt.savefig("T3_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) Escribir resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": (
        "Comprobación empírica: 10 000 muestras de la distribución renormalizada de T2 "
        "(top-p p=0.9 sobre la distribución de T=1) con numpy.random.default_rng(0); "
        "comparación de frecuencias observadas vs. probabilidades teóricas en un diagrama "
        "de barras y divergencia KL (en bits) de la empírica respecto de la teórica."
    ),
    "n_muestras": N_MUESTRAS,
    "semilla": SEMILLA,
    "generador": "numpy.random.default_rng(0)",
    "fuente_distribucion": "entrada/T2/resultados.json",
    "tokens": tokens,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens, p_teo)},
    "probabilidades_teoricas_lista": [float(p) for p in p_teo],
    "suma_probabilidades_teoricas": float(p_teo.sum()),
    "frecuencias_observadas_conteo": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas_conteo_lista": [int(c) for c in conteos],
    "frecuencias_observadas_relativas": {t: float(fr) for t, fr in zip(tokens, freq_obs)},
    "frecuencias_observadas_relativas_lista": [float(fr) for fr in freq_obs],
    "suma_conteos_observados": int(conteos.sum()),
    "suma_frecuencias_relativas": float(freq_obs.sum()),
    "divergencia_KL_bits": kl_bits,
    "divergencia_KL_nats": kl_nats,
    "figura": "T3_frecuencias_vs_teoricas.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("T3 — Comprobación empírica")
print(f"Muestras: {N_MUESTRAS} con numpy.random.default_rng({SEMILLA})")
print(f"Tokens (soporte top-p): {tokens}")
print("Probabilidades teóricas :", np.round(p_teo, 4))
print("Frecuencias observadas  :", np.round(freq_obs, 4))
print(f"Conteos observados      : {conteos.tolist()} (suma = {int(conteos.sum())})")
print(f"Divergencia KL(empírica || teórica) = {kl_bits:.6f} bits")
print("Figura guardada: T3_frecuencias_vs_teoricas.png")
print("Resultados escritos en resultados.json")
