# T1 — Softmax con temperatura y entropía en bits (implementación en NumPy)
# Entregables: resultados.json (contrato) + figura PNG + resumen en consola.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def softmax_con_temperatura(logits, T):
    """p_i = exp(z_i/T) / sum_j exp(z_j/T), numéricamente estable:
    se resta el máximo de los logits escalados antes de exponenciar."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - np.max(z)  # estabilidad numérica
    e = np.exp(z)
    return e / np.sum(e)


def entropia_bits(p):
    """H = -sum p_i * log2(p_i), en bits (ignora términos con p_i = 0)."""
    p = np.asarray(p, dtype=float)
    m = p > 0
    return float(-np.sum(p[m] * np.log2(p[m])))


# ----------------------------- Datos del enunciado -----------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]
temperaturas = [0.5, 1.0, 2.0]

# ----------------------------- Cálculos -----------------------------
distribuciones = {T: softmax_con_temperatura(logits, T) for T in temperaturas}
entropias = {T: entropia_bits(distribuciones[T]) for T in temperaturas}

# ----------------------------- Tabla con cuatro decimales -----------------------------
tabla = pd.DataFrame(index=tokens)
for T in temperaturas:
    tabla[f"T={T}"] = [f"{p:.4f}" for p in distribuciones[T]]
tabla.loc["entropia_bits"] = {f"T={T}": f"{entropias[T]:.4f}" for T in temperaturas}

# ----------------------------- Verificaciones del criterio de éxito -----------------------------
sumas = {f"T={T}": float(np.sum(distribuciones[T])) for T in temperaturas}
sumas_ok = all(abs(s - 1.0) < 1e-12 for s in sumas.values())
entropia_crece = bool(entropias[0.5] < entropias[1.0] < entropias[2.0])

# ----------------------------- Figura -----------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.5))
x = np.arange(len(tokens))
ancho = 0.26
colores = ["#4C72B0", "#55A868", "#C44E52"]
for j, T in enumerate(temperaturas):
    ax.bar(
        x + (j - 1) * ancho,
        distribuciones[T],
        width=ancho,
        color=colores[j],
        label=f"T={T}  (H={entropias[T]:.4f} bits)",
    )
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura — logits: 2.0, 1.0, 0.5, 0.2, -1.0, -3.0")
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------- Tabla en formato lista de filas (para JSON) -----------------------------
tabla_json = []
for i, tok in enumerate(tokens):
    fila = {"token": tok, "logit": logits[i]}
    for T in temperaturas:
        fila[f"T={T}"] = f"{distribuciones[T][i]:.4f}"
    tabla_json.append(fila)
fila_h = {"token": "entropia_bits", "logit": None}
for T in temperaturas:
    fila_h[f"T={T}"] = f"{entropias[T]:.4f}"
tabla_json.append(fila_h)

# ----------------------------- Contrato: resultados.json -----------------------------
resultados = {
    "subtarea": "T1 - Softmax con temperatura y entropia en bits",
    "tokens": tokens,
    "logits": logits,
    "temperaturas": temperaturas,
    "distribuciones_sin_redondear": {
        f"T={T}": distribuciones[T].tolist() for T in temperaturas
    },
    "entropias_bits_sin_redondear": {f"T={T}": entropias[T] for T in temperaturas},
    "tabla_4_decimales": {
        "distribuciones": {
            f"T={T}": [f"{p:.4f}" for p in distribuciones[T]] for T in temperaturas
        },
        "entropias_bits": {f"T={T}": f"{entropias[T]:.4f}" for T in temperaturas},
    },
    "tabla_4_decimales_filas": tabla_json,
    "verificacion": {
        "suma_por_temperatura": sumas,
        "cada_distribucion_suma_1": sumas_ok,
        "entropia_crece_con_T": entropia_crece,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------- Resumen -----------------------------
print("=== T1: Softmax con temperatura ===")
print("Logits:", dict(zip(tokens, logits)))
print()
print("Tabla (4 decimales):")
print(tabla.to_string())
print()
print("Entropías en bits (sin redondear):")
for T in temperaturas:
    print(f"  T={T}: H = {entropias[T]:.10f} bits")
print()
print(f"Verificación: sumas = {sumas} (todas = 1: {sumas_ok})")
print(f"Verificación: H(T=0.5) < H(T=1) < H(T=2) -> {entropia_crece}")
print("Archivos generados: resultados.json, t1_softmax_temperatura.png")
