# T1: Carga de Breast Cancer Wisconsin (Diagnostic), reporte de cifras y
# creación de la división entrenamiento/prueba 70/30 estratificada (random_state=42).

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]  # 0: malignant, 1: benign

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    valores_clase, conteos_clase = np.unique(y, return_counts=True)
    conteo_por_clase = {
        target_names[int(v)]: int(c) for v, c in zip(valores_clase, conteos_clase)
    }
    conteo_por_clase_numerico = {
        int(v): int(c) for v, c in zip(valores_clase, conteos_clase)
    }

    # ------------------------------------------------------------------
    # 2) División 70/30 estratificada por clase, random_state=42
    #    (única división usada en toda la tarea; el test no se toca)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(n_ejemplos),
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])

    vals_tr, cnt_tr = np.unique(y_train, return_counts=True)
    vals_te, cnt_te = np.unique(y_test, return_counts=True)
    conteo_train = {target_names[int(v)]: int(c) for v, c in zip(vals_tr, cnt_tr)}
    conteo_test = {target_names[int(v)]: int(c) for v, c in zip(vals_te, cnt_te)}

    # ------------------------------------------------------------------
    # 3) Guardar las particiones para las subtareas siguientes
    # ------------------------------------------------------------------
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train.insert(0, "original_index", idx_train.astype(int))
    df_train["target"] = y_train.astype(int)

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test.insert(0, "original_index", idx_test.astype(int))
    df_test["target"] = y_test.astype(int)

    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)

    meta = {
        "feature_names": feature_names,
        "target_names": target_names,
        "split": {
            "test_size": 0.30,
            "train_size": 0.70,
            "stratify": "target",
            "random_state": 42,
        },
    }
    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # 4) Figura: distribución de clases (global, train y test)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    conjuntos = [
        ("Completo", conteo_por_clase),
        ("Entrenamiento (70%)", conteo_train),
        ("Prueba (30%)", conteo_test),
    ]
    for ax, (titulo, conteo) in zip(axes, conjuntos):
        ax.bar(list(conteo.keys()), list(conteo.values()), color=["#c44e52", "#4c72b0"])
        ax.set_title(titulo)
        ax.set_ylabel("Número de casos")
        for i, v in enumerate(conteo.values()):
            ax.text(i, v, str(v), ha="center", va="bottom")
    fig.suptitle("Breast Cancer Wisconsin (Diagnostic): distribución de clases")
    fig.tight_layout()
    plt.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "total_ejemplos": n_ejemplos,
        "numero_atributos": n_atributos,
        "clases": target_names,
        "conteo_por_clase": conteo_por_clase,
        "conteo_por_clase_numerico": conteo_por_clase_numerico,
        "division": {
            "test_size": 0.30,
            "train_size": 0.70,
            "estratificada": True,
            "random_state": 42,
        },
        "n_train": n_train,
        "n_test": n_test,
        "conteo_train_por_clase": conteo_train,
        "conteo_test_por_clase": conteo_test,
        "archivos_generados": {
            "train": "train.csv",
            "test": "test.csv",
            "meta": "meta.json",
            "figura": "t1_distribucion_clases.png",
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Total de ejemplos : {n_ejemplos}")
    print(f"Número de atributos: {n_atributos}")
    print(f"Conteo por clase   : {conteo_por_clase}")
    print(f"División 70/30 estratificada (random_state=42):")
    print(f"  Entrenamiento: {n_train} ejemplos -> {conteo_train}")
    print(f"  Prueba       : {n_test} ejemplos -> {conteo_test}")
    print("Archivos: train.csv, test.csv, meta.json, t1_distribucion_clases.png, resultados.json")


if __name__ == "__main__":
    main()
