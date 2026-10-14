# -*- coding: utf-8 -*-
"""
T1 — Carga de Breast Cancer Wisconsin (Diagnostic) y creación de la única
división train/test 70/30 estratificada por clase (random_state=42).

Salidas (carpeta actual):
  - resultados.json            : cifras solicitadas por la subtarea (contrato).
  - division_train_test.json   : índices de la división, reutilizable en T2+.
  - t1_distribucion_clases.png : distribución de clases (completo/train/test).
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEMILLA = 42
TEST_SIZE = 0.30


def conteo_por_clase(y, target_names):
    """Devuelve (dict clase->conteo, valores de clase, conteos)."""
    valores, conteos = np.unique(y, return_counts=True)
    dicc = {
        f"clase_{int(v)}_{target_names[int(v)]}": int(c)
        for v, c in zip(valores, conteos)
    }
    return dicc, valores, conteos


def main():
    # ---------- 1) Carga del conjunto de datos ----------
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    conteo_total, valores_clase, _ = conteo_por_clase(y, target_names)

    # ---------- 2) División ÚNICA 70/30 estratificada por clase, random_state=42 ----------
    # Se divide sobre los índices para poder guardarla y reproducirla exactamente.
    indices = np.arange(n_ejemplos)
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=y, random_state=SEMILLA
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    conteo_train, _, _ = conteo_por_clase(y_train, target_names)
    conteo_test, _, _ = conteo_por_clase(y_test, target_names)

    n_train, n_test = int(len(train_idx)), int(len(test_idx))

    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "total_ejemplos": n_ejemplos,
        "numero_atributos": n_atributos,
        "nombres_atributos": feature_names,
        "mapa_clases": {str(int(v)): target_names[int(v)] for v in valores_clase},
        "conteo_por_clase": conteo_total,
        "proporcion_por_clase": {
            k: float(c) / n_ejemplos for k, c in conteo_total.items()
        },
        "division_train_test": {
            "test_size": TEST_SIZE,
            "train_size": float(1.0 - TEST_SIZE),
            "estratificada_por_clase": True,
            "random_state": SEMILLA,
            "n_train": n_train,
            "n_test": n_test,
            "fraccion_train": float(n_train) / n_ejemplos,
            "fraccion_test": float(n_test) / n_ejemplos,
            "conteo_por_clase_train": conteo_train,
            "conteo_por_clase_test": conteo_test,
        },
        "archivos_generados": [
            "resultados.json",
            "division_train_test.json",
            "t1_distribucion_clases.png",
        ],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ---------- 3) Guardar la división para reutilizarla en las demás subtareas ----------
    with open("division_train_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "descripcion": (
                    "Única división train/test 70/30 estratificada por clase "
                    "(random_state=42). X_train = X[train_indices], etc."
                ),
                "random_state": SEMILLA,
                "test_size": TEST_SIZE,
                "estratificada_por_clase": True,
                "train_indices": [int(i) for i in train_idx],
                "test_indices": [int(i) for i in test_idx],
                "y_train": [int(v) for v in y_train],
                "y_test": [int(v) for v in y_test],
            },
            f,
            indent=2,
        )

    # ---------- 4) Figura: distribución de clases ----------
    claves = [f"clase_{int(v)}_{target_names[int(v)]}" for v in valores_clase]
    etiquetas = [f"clase {int(v)}\n({target_names[int(v)]})" for v in valores_clase]
    v_full = [float(conteo_total[k]) for k in claves]
    v_train = [float(conteo_train[k]) for k in claves]
    v_test = [float(conteo_test[k]) for k in claves]

    x = np.arange(len(claves))
    ancho = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - ancho, v_full, ancho, label=f"Completo (n={n_ejemplos})")
    ax.bar(x, v_train, ancho, label=f"Entrenamiento 70% (n={n_train})")
    ax.bar(x + ancho, v_test, ancho, label=f"Prueba 30% (n={n_test})")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Número de ejemplos")
    ax.set_title("Breast Cancer Wisconsin (Diagnostic): distribución de clases")
    ax.legend()
    fig.tight_layout()
    fig.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ---------- 5) Resumen ----------
    print("=" * 62)
    print("T1 — Breast Cancer Wisconsin (Diagnostic)")
    print(f"Total de ejemplos  : {n_ejemplos}")
    print(f"Número de atributos: {n_atributos}")
    for k, c in conteo_total.items():
        print(f"  {k}: {c} casos ({100.0 * c / n_ejemplos:.2f}%)")
    print(f"División 70/30 estratificada (random_state={SEMILLA}):")
    print(f"  Entrenamiento: {n_train} ejemplos -> {conteo_train}")
    print(f"  Prueba       : {n_test} ejemplos -> {conteo_test}")
    print(
        "Archivos generados: resultados.json, division_train_test.json, "
        "t1_distribucion_clases.png"
    )
    print("=" * 62)


if __name__ == "__main__":
    main()
