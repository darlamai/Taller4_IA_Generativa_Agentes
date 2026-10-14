# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Softmax con temperatura (NumPy, estable) + entropía en bits.
Tarea B — Temperatura, top-p y entropía · MMIA 6013
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1) Softmax con temperatura, numéricamente estable (se resta el máximo)
# ----------------------------------------------------------------------
def softmax_con_temperatura(logits, T):
    """p_i = exp(z_i/T) / sum_j exp(z_j/T), restando el máximo para estabilidad."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - np.max(z)          # estabilidad numérica: evita overflow de exp
    e = np.exp(z)
    return e / np.sum(e)


def entropia_bits(p):
    """H(p) = -sum p_i log2 p_i, en bits (se ignoran p_i = 0 por convención 0*log2(0)=0)."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


# ----------------------------------------------------------------------
# 2) Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]

distribuciones = {T: softmax_con_temperatura(logits, T) for T in temperaturas}
entropias = {T: entropia_bits(distribuciones[T]) for T in temperaturas}

# ----------------------------------------------------------------------
# 3) Tabla con cuatro decimales (distribuciones + entropías)
# ----------------------------------------------------------------------
filas = []
for T in temperaturas:
    filas.append([f"{T:.1f}"] + [f"{p:.4f}" for p in distribuciones[T]]
                 + [f"{entropias[T]:.4f}"])
tabla = pd.DataFrame(filas, columns=["T"] + tokens + ["H_bits"])
print("Tabla (4 decimales):")
print(tabla.to_string(index=False))

# Verificación: cada fila de probabilidades suma 1.0
sumas = {f"T={T}": float(np.sum(distribuciones[T])) for T in temperaturas}
print("\nSumas de cada fila de probabilidades:", sumas)

# ----------------------------------------------------------------------
# 4) Figura: distribuciones para cada temperatura
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(tokens))
ancho = 0.25
for i, T in enumerate(temperaturas):
    ax.bar(x + (i - 1) * ancho, distribuciones[T], ancho, label=f"T = {T}")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("p_i")
ax.set_xlabel("Token")
ax.set_title("Softmax con temperatura (logits 2.0, 1.0, 0.5, 0.2, -1.0, -3.0)")
ax.legend()
fig.tight_layout()
fig.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {f"T={T}": distribuciones[T].tolist() for T in temperaturas},
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "suma_probabilidades_por_fila": sumas,
    "tabla_4_decimales": {
        "columnas": ["T"] + tokens + ["H_bits"],
        "filas": filas,
    },
}
Path("resultados.json").write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ----------------------------------------------------------------------
# 6) Resumen
# ----------------------------------------------------------------------
print("\nResumen T1:")
for T in temperaturas:
    print(f"  T={T}: p={np.round(distribuciones[T], 4).tolist()}  H={entropias[T]:.4f} bits")
print("Archivos generados: resultados.json, t1_softmax_temperatura.png")
