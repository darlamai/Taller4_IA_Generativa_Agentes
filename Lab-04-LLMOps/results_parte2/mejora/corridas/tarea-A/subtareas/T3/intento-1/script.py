# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje (Tarea A — Generativo contra discriminativo, MMIA 6013).

Sobre la MISMA división 70/30 estratificada (random_state=42) de la Parte 1,
entrena Naive Bayes gaussiano y Regresión logística con 5%, 10%, 25%, 50% y 100%
del conjunto de entrenamiento (submuestras estratificadas, random_state=42),
evalúa la exactitud de cada uno SIEMPRE sobre el mismo conjunto de prueba y
dibuja la curva de aprendizaje (una curva por modelo), guardándola como PNG.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
PORCENTAJES = [5, 10, 25, 50, 100]
FRACCIONES = [p / 100.0 for p in PORCENTAJES]
FIGURA_PNG = "t3_curva_aprendizaje.png"
NOMBRE_CLASE = {0: "malignant", 1: "benign"}

# ---------------------------------------------------------------------------
# 1) Misma división de la Parte 1 (entrada/T1): 70% train / 30% test, seed 42
# ---------------------------------------------------------------------------
ruta_train = Path("entrada/T1/train.csv")
ruta_test = Path("entrada/T1/test.csv")

if ruta_train.exists() and ruta_test.exists():
    train_df = pd.read_csv(ruta_train)
    test_df = pd.read_csv(ruta_test)
    target_col = "target" if "target" in train_df.columns else train_df.columns[-1]

    def separar(df):
        cols = [c for c in df.columns
                if c != target_col and not str(c).startswith("Unnamed")]
        return df[cols].to_numpy(dtype=float), df[target_col].to_numpy()

    X_train_full, y_train_full = separar(train_df)
    X_test, y_test = separar(test_df)
    origen_division = "entrada/T1 (train.csv / test.csv generados en la Parte 1)"
else:
    # Respaldo: recrear exactamente la división de la Parte 1
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data.data, data.target, test_size=0.3, stratify=data.target,
        random_state=RANDOM_STATE,
    )
    origen_division = "recreada con load_breast_cancer (70/30 estratificada, random_state=42)"

n_train_total = int(len(y_train_full))
n_test = int(len(y_test))
print(f"División usada: {origen_division}")
print(f"n_train={n_train_total} | n_test={n_test} (debe ser 398/171 según T1)")

# ---------------------------------------------------------------------------
# 2) Bucle sobre porcentajes: submuestra estratificada -> entrenar -> evaluar
#    en el MISMO conjunto de prueba. El escalador se ajusta SOLO con la
#    submuestra de entrenamiento correspondiente (nunca con el test).
# ---------------------------------------------------------------------------
filas = []
composicion = []
curvas = {"naive_bayes_gaussiano": [], "regresion_logistica": []}
n_vals = []

for p, frac in zip(PORCENTAJES, FRACCIONES):
    if frac >= 1.0:
        X_sub, y_sub = X_train_full, y_train_full
    else:
        X_sub, _, y_sub, _ = train_test_split(
            X_train_full, y_train_full,
            train_size=frac,
            stratify=y_train_full,
            random_state=RANDOM_STATE,
        )
    n_sub = int(len(y_sub))

    clases_sub, conteos_sub = np.unique(y_sub, return_counts=True)
    comp = {"porcentaje_entrenamiento": p, "n_ejemplos": n_sub}
    for c, k in zip(clases_sub, conteos_sub):
        comp[NOMBRE_CLASE.get(int(c), f"clase_{int(c)}")] = int(k)
    composicion.append(comp)

    # Escalador ajustado únicamente con la submuestra de entrenamiento
    scaler = StandardScaler().fit(X_sub)
    X_sub_z = scaler.transform(X_sub)
    X_test_z = scaler.transform(X_test)  # misma transformación, sin reajustar

    # Modelo generativo
    nb = GaussianNB()
    nb.fit(X_sub_z, y_sub)
    acc_nb = float(accuracy_score(y_test, nb.predict(X_test_z)))

    # Modelo discriminativo
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_sub_z, y_sub)
    acc_lr = float(accuracy_score(y_test, lr.predict(X_test_z)))

    n_vals.append(n_sub)
    curvas["naive_bayes_gaussiano"].append(acc_nb)
    curvas["regresion_logistica"].append(acc_lr)
    filas.append({
        "porcentaje_entrenamiento": p,
        "n_ejemplos_entrenamiento": n_sub,
        "accuracy_naive_bayes_gaussiano": acc_nb,
        "accuracy_regresion_logistica": acc_lr,
    })

# Modelo ganador por tamaño y detección de cruce
ganadores = [
    "naive_bayes_gaussiano"
    if r["accuracy_naive_bayes_gaussiano"] >= r["accuracy_regresion_logistica"]
    else "regresion_logistica"
    for r in filas
]
hay_cruce = len(set(ganadores)) > 1

# ---------------------------------------------------------------------------
# 3) Figura: curva de aprendizaje (eje x = nº de ejemplos de entrenamiento)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(n_vals, curvas["naive_bayes_gaussiano"], marker="o", color="tab:blue",
        label="Naive Bayes gaussiano (generativo)")
ax.plot(n_vals, curvas["regresion_logistica"], marker="s", color="tab:red",
        label="Regresión logística (discriminativo)")

ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
ax.set_title("Curva de aprendizaje — Breast Cancer\n(división 70/30 de la Parte 1, prueba fija)")
ax.set_xticks(n_vals)
ax.set_xticklabels([f"{n}\n({p}%)" for n, p in zip(n_vals, PORCENTAJES)])
ax.grid(True, alpha=0.3)
todos = curvas["naive_bayes_gaussiano"] + curvas["regresion_logistica"]
ax.set_ylim(max(0.0, min(todos) - 0.05), 1.02)
for x, y in zip(n_vals, curvas["naive_bayes_gaussiano"]):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8, color="tab:blue")
for x, y in zip(n_vals, curvas["regresion_logistica"]):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color="tab:red")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURA_PNG, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Verificación de consistencia con T2 (el punto 100% debe reproducirlo)
# ---------------------------------------------------------------------------
consistencia = {"nota": "No se pudo leer entrada/T2/resultados.json"}
try:
    with open("entrada/T2/resultados.json", "r", encoding="utf-8") as f:
        t2 = json.load(f)
    acc_nb_t2 = float(t2["modelos"]["naive_bayes_gaussiano"]["accuracy_test"])
    acc_lr_t2 = float(t2["modelos"]["regresion_logistica"]["accuracy_test"])
    consistencia = {
        "accuracy_naive_bayes_T2": acc_nb_t2,
        "accuracy_naive_bayes_T3_100pct": curvas["naive_bayes_gaussiano"][-1],
        "coincide_naive_bayes": bool(abs(acc_nb_t2 - curvas["naive_bayes_gaussiano"][-1]) < 1e-12),
        "accuracy_regresion_logistica_T2": acc_lr_t2,
        "accuracy_regresion_logistica_T3_100pct": curvas["regresion_logistica"][-1],
        "coincide_regresion_logistica": bool(abs(acc_lr_t2 - curvas["regresion_logistica"][-1]) < 1e-12),
    }
except Exception:
    pass

# ---------------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": ("Curva de aprendizaje: Naive Bayes gaussiano y regresión logística "
                    "entrenados con 5%, 10%, 25%, 50% y 100% del conjunto de entrenamiento "
                    "(submuestras estratificadas, random_state=42) y evaluados siempre "
                    "sobre el mismo conjunto de prueba de la Parte 1."),
    "division_origen": origen_division,
    "n_train_total": n_train_total,
    "n_test": n_test,
    "submuestreo": {
        "estratificado": True,
        "random_state": 42,
        "porcentajes": PORCENTAJES,
        "metodo": "train_test_split(train_size=fraccion, stratify=y_train, random_state=42)",
        "escalador": "StandardScaler ajustado solo con cada submuestra de entrenamiento (nunca con el conjunto de prueba)",
        "conjunto_de_prueba": "fijo e intacto en todas las evaluaciones (171 ejemplos)",
    },
    "n_ejemplos_entrenamiento": n_vals,
    "tabla": filas,
    "accuracy_por_porcentaje": {
        str(r["porcentaje_entrenamiento"]): {
            "n_ejemplos_entrenamiento": r["n_ejemplos_entrenamiento"],
            "naive_bayes_gaussiano": r["accuracy_naive_bayes_gaussiano"],
            "regresion_logistica": r["accuracy_regresion_logistica"],
        }
        for r in filas
    },
    "accuracy_por_modelo": {
        "naive_bayes_gaussiano": {
            str(n): a for n, a in zip(n_vals, curvas["naive_bayes_gaussiano"])
        },
        "regresion_logistica": {
            str(n): a for n, a in zip(n_vals, curvas["regresion_logistica"])
        },
    },
    "curva_aprendizaje": {
        "eje_x_n_ejemplos_entrenamiento": n_vals,
        "naive_bayes_gaussiano": curvas["naive_bayes_gaussiano"],
        "regresion_logistica": curvas["regresion_logistica"],
    },
    "composicion_submuestras": composicion,
    "modelo_ganador_por_tamano": ganadores,
    "hay_cruce_entre_curvas": hay_cruce,
    "consistencia_con_T2_100pct": consistencia,
    "archivos_generados": {
        "figura_curva_aprendizaje": FIGURA_PNG,
        "resultados": "resultados.json",
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 6) Resumen
# ---------------------------------------------------------------------------
tabla_df = pd.DataFrame(filas)
print("\n=== T3 — Tabla de accuracy (evaluada siempre en el mismo test de 171 ejemplos) ===")
print(tabla_df.to_string(index=False))
print(f"\nGanador por tamaño: {ganadores}")
print(f"¿Hay cruce entre las curvas?: {hay_cruce}")
print(f"Figura guardada: {FIGURA_PNG}")
print("Resultados guardados: resultados.json")
