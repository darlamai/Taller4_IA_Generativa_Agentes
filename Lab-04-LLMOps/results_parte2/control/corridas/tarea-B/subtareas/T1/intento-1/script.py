import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Datos del enunciado (Parte 1 — Softmax con temperatura)
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]


def softmax_con_temperatura(z, T):
    """Softmax con temperatura p_i = exp(z_i/T) / sum_j exp(z_j/T).

    Numéricamente estable: se resta el máximo de z/T antes de exponenciar,
    de modo que el mayor exponente es 0 y no hay desbordamiento.
    """
    z = np.asarray(z, dtype=float)
    escalados = z / float(T)
    escalados = escalados - np.max(escalados)  # estabilidad numérica
    e = np.exp(escalados)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -sum p_i log2(p_i), con 0·log2(0)=0."""
    p = np.asarray(p, dtype=float)
    p_pos = p[p > 0]
    return float(-np.sum(p_pos * np.log2(p_pos)))


# ----------------------------------------------------------------------
# Cálculo de distribuciones y entropías
# ----------------------------------------------------------------------
distribuciones = {T: softmax_con_temperatura(logits, T) for T in temperaturas}
entropias = {T: entropia_bits(distribuciones[T]) for T in temperaturas}
sumas = {T: float(np.sum(distribuciones[T])) for T in temperaturas}

# Verificaciones del criterio de éxito
sumas_ok = all(abs(sumas[T] - 1.0) < 1e-12 for T in temperaturas)
monotona = bool(entropias[0.5] < entropias[1.0] < entropias[2.0])

# ----------------------------------------------------------------------
# Tabla con cuatro decimales (distribuciones + entropía en bits)
# ----------------------------------------------------------------------
cols = [f"T={T}" for T in temperaturas]
tabla = pd.DataFrame(
    {c: [f"{distribuciones[T][i]:.4f}" for i in range(len(tokens))]
     for c, T in zip(cols, temperaturas)},
    index=tokens,
)
tabla.loc["H (bits)"] = [f"{entropias[T]:.4f}" for T in temperaturas]

# ----------------------------------------------------------------------
# Figura: distribuciones para cada temperatura
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(tokens))
ancho = 0.25
for i, T in enumerate(temperaturas):
    ax.bar(x + (i - 1) * ancho, distribuciones[T], width=ancho, label=f"T = {T}")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_xlabel("Token")
ax.set_title("Softmax con temperatura (logits: 2.0, 1.0, 0.5, 0.2, -1.0, -3.0)")
ax.legend()
fig.tight_layout()
plt.savefig("softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# Contrato: resultados.json (valores sin redondear + tabla a 4 decimales)
# ----------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {f"T={T}": distribuciones[T].tolist() for T in temperaturas},
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "suma_de_cada_distribucion": {f"T={T}": sumas[T] for T in temperaturas},
    "verificacion_suma_1": bool(sumas_ok),
    "entropia_monotona_creciente_con_T": monotona,
    "tabla_4_decimales": {
        "columnas": ["token"] + cols,
        "filas": (
            [[tok] + [f"{distribuciones[T][i]:.4f}" for T in temperaturas]
             for i, tok in enumerate(tokens)]
            + [["H (bits)"] + [f"{entropias[T]:.4f}" for T in temperaturas]]
        ),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# Tabla también en CSV (opcional, solo la presentación a 4 decimales)
tabla.to_csv("tabla_softmax_temperatura.csv")

# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
print("Softmax con temperatura — Tarea B, Parte 1")
print(f"Logits: {logits.tolist()}")
print()
print(tabla.to_string())
print()
for T in temperaturas:
    print(f"T={T}: suma={sumas[T]:.12f}  H={entropias[T]:.10f} bits")
print(f"Sumas iguales a 1: {sumas_ok}")
print(f"Entropía monótona creciente con T (H(0.5) < H(1) < H(2)): {monotona}")
print("Archivos generados: resultados.json, softmax_temperatura.png, tabla_softmax_temperatura.csv")
