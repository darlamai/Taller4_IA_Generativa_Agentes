# T1: Carga de Breast Cancer Wisconsin (Diagnostic), reporte de cifras y
# creación de la división única entrenamiento/prueba 70/30 estratificada
# con random_state=42, guardada para las subtareas siguientes.

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.30

# ------------------------------------------------------------------
# 1) Carga del conjunto de datos
# ------------------------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = [str(f) for f in data.feature_names]
target_names = [str(t) for t in data.target_names]

n_ejemplos = int(X.shape[0])
n_atributos = int(X.shape[1])

clases, conteos = np.unique(y, return_counts=True)
conteo_por_clase = {target_names[int(c)]: int(n) for c, n in zip(clases, conteos)}
conteo_por_clase_etiqueta = {int(c): int(n) for c, n in zip(clases, conteos)}

# ------------------------------------------------------------------
# 2) División única 70/30 estratificada por clase, random_state=42
#    (el conjunto de prueba no se usa para ajustar nada)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

train_clases, train_conteos = np.unique(y_train, return_counts=True)
test_clases, test_conteos = np.unique(y_test, return_counts=True)
train_por_clase = {target_names[int(c)]: int(n) for c, n in zip(train_clases, train_conteos)}
test_por_clase = {target_names[int(c)]: int(n) for c, n in zip(test_clases, test_conteos)}

# ------------------------------------------------------------------
# 3) Contrato: resultados.json con todas las cifras pedidas
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "total_ejemplos": n_ejemplos,
    "numero_atributos": n_atributos,
    "nombres_atributos": feature_names,
    "nombres_clases": target_names,
    "casos_por_clase": conteo_por_clase,
    "casos_por_clase_etiqueta_numerica": conteo_por_clase_etiqueta,
    "division": {
        "test_size": TEST_SIZE,
        "train_size": 1.0 - TEST_SIZE,
        "estratificado_por_clase": True,
        "random_state": RANDOM_STATE,
        "tam_entrenamiento": int(X_train.shape[0]),
        "tam_prueba": int(X_test.shape[0]),
        "proporcion_entrenamiento": float(X_train.shape[0] / n_ejemplos),
        "proporcion_prueba": float(X_test.shape[0] / n_ejemplos),
        "casos_por_clase_entrenamiento": train_por_clase,
        "casos_por_clase_prueba": test_por_clase,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 4) La división queda disponible para las subtareas siguientes
# ------------------------------------------------------------------
np.savez(
    "division_train_test.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=np.array(feature_names),
    target_names=np.array(target_names),
)

df_train = pd.DataFrame(X_train, columns=feature_names)
df_train.insert(0, "target", y_train.astype(int))
df_train.to_csv("train.csv", index=False)

df_test = pd.DataFrame(X_test, columns=feature_names)
df_test.insert(0, "target", y_test.astype(int))
df_test.to_csv("test.csv", index=False)

# ------------------------------------------------------------------
# 5) Figura: distribución de clases en total/entrenamiento/prueba
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(len(clases), dtype=float)
width = 0.25
ax.bar(xpos - width, conteos.astype(float), width, label="Total")
ax.bar(xpos, [float(train_por_clase.get(target_names[int(c)], 0)) for c in clases],
       width, label="Entrenamiento (70 %)")
ax.bar(xpos + width, [float(test_por_clase.get(target_names[int(c)], 0)) for c in clases],
       width, label="Prueba (30 %)")
ax.set_xticks(xpos)
ax.set_xticklabels([f"{target_names[int(c)]} (etiqueta {int(c)})" for c in clases])
ax.set_ylabel("Número de casos")
ax.set_title("Breast Cancer Wisconsin (Diagnostic): casos por clase")
ax.legend()
fig.tight_layout()
plt.savefig("distribucion_clases.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T1: resumen ===")
print(f"Dataset: Breast Cancer Wisconsin (Diagnostic)")
print(f"Total de ejemplos: {n_ejemplos}")
print(f"Número de atributos: {n_atributos}")
print(f"Casos por clase: {conteo_por_clase}")
print(f"División 70/30 estratificada, random_state={RANDOM_STATE}")
print(f"  Entrenamiento: {X_train.shape[0]} ejemplos "
      f"({100.0 * X_train.shape[0] / n_ejemplos:.2f} %) -> {train_por_clase}")
print(f"  Prueba:        {X_test.shape[0]} ejemplos "
      f"({100.0 * X_test.shape[0] / n_ejemplos:.2f} %) -> {test_por_clase}")
print("Archivos generados: resultados.json, division_train_test.npz, "
      "train.csv, test.csv, distribucion_clases.png")
