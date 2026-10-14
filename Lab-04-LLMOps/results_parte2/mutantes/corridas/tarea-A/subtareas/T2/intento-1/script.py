# -*- coding: utf-8 -*-
"""
Subtarea T2 — Tarea A (MMIA 6013), Parte 2: dos clasificadores.

Entrena, sobre la MISMA división 70/30 estratificada (random_state=42) creada en
la Parte 1 (T1), un Naive Bayes gaussiano y una regresión logística (con los
atributos estandarizados; el escalador se ajusta SOLO con el conjunto de
entrenamiento) y mide accuracy y F1 macro de cada uno sobre el conjunto de
prueba. Presenta los resultados en una tabla (consola, CSV y PNG) y escribe
todas las cifras en resultados.json.

Salidas:
  - resultados.json               (contrato de la subtarea: 4 cifras + verificaciones)
  - tabla_parte2_metricas.png     (tabla como figura)
  - tabla_parte2_metricas.csv     (tabla en CSV)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

TEST_SIZE = 0.3
RANDOM_STATE = 42
T1_DIR = Path("entrada/T1")


def cargar_resultados_t1():
    ruta = T1_DIR / "resultados.json"
    if ruta.exists():
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def cargar_split_previo(X, y):
    """Intenta reutilizar la división de la Parte 1 guardada en entrada/T1/.

    Acepta, si existen: (a) X_train/X_test/y_train/y_test .npy, o
    (b) train_indices/test_indices .npy. Devuelve None si no hay archivos.
    """
    nombres_arrays = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    if all((T1_DIR / n).exists() for n in nombres_arrays):
        X_tr = np.load(T1_DIR / "X_train.npy")
        X_te = np.load(T1_DIR / "X_test.npy")
        y_tr = np.load(T1_DIR / "y_train.npy")
        y_te = np.load(T1_DIR / "y_test.npy")
        return X_tr, X_te, y_tr, y_te, "archivos de entrada/T1 (X_*/y_*.npy de la Parte 1)"
    if (T1_DIR / "train_indices.npy").exists() and (T1_DIR / "test_indices.npy").exists():
        idx_tr = np.load(T1_DIR / "train_indices.npy").astype(int)
        idx_te = np.load(T1_DIR / "test_indices.npy").astype(int)
        return (
            X[idx_tr],
            X[idx_te],
            y[idx_tr],
            y[idx_te],
            "archivos de entrada/T1 (índices de la Parte 1)",
        )
    return None


def conteos_por_clase(y, nombres_clases):
    return {nombres_clases[int(c)]: int(np.sum(y == c)) for c in np.unique(y)}


def main():
    # ------------------------------------------------------------------ datos
    data = load_breast_cancer()
    X, y = data.data, data.target
    nombres_clases = list(data.target_names)  # ['malignant', 'benign']

    t1 = cargar_resultados_t1()

    # ------------------------------------------------- división de la Parte 1
    split_previo = cargar_split_previo(X, y)
    if split_previo is not None:
        X_tr, X_te, y_tr, y_te, fuente_split = split_previo
    else:
        # Misma división exacta que la Parte 1: 70/30 estratificada, random_state=42
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        fuente_split = (
            "recreada de forma determinista: train_test_split(test_size=0.3, "
            "stratify=y, random_state=42), idéntica a la de la Parte 1"
        )

    # Verificación de la división contra los resultados de T1
    verif_split = {"fuente": fuente_split}
    if t1 is not None:
        d = t1.get("division", {})
        verif_split["tam_entrenamiento_coincide_con_T1"] = bool(
            d.get("tam_entrenamiento") == len(y_tr)
        )
        verif_split["tam_prueba_coincide_con_T1"] = bool(d.get("tam_prueba") == len(y_te))
        esp_tr = {
            nombres_clases.index(k): v
            for k, v in d.get("casos_por_clase_entrenamiento", {}).items()
        }
        esp_te = {
            nombres_clases.index(k): v
            for k, v in d.get("casos_por_clase_prueba", {}).items()
        }
        obs_tr = {int(c): int(np.sum(y_tr == c)) for c in np.unique(y_tr)}
        obs_te = {int(c): int(np.sum(y_te == c)) for c in np.unique(y_te)}
        verif_split["casos_por_clase_entrenamiento_coincide_con_T1"] = bool(esp_tr == obs_tr)
        verif_split["casos_por_clase_prueba_coincide_con_T1"] = bool(esp_te == obs_te)

    # ------------------------------------- escalador (ajustado SOLO con train)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)  # fit únicamente con el entrenamiento
    X_te_s = scaler.transform(X_te)      # el test solo se transforma, nunca se ajusta

    verif_scaler = {
        "ajustado_solo_con_entrenamiento": True,
        "media_scaler_igual_a_media_del_entrenamiento": bool(
            np.allclose(scaler.mean_, X_tr.mean(axis=0))
        ),
        "media_scaler_distinta_a_media_de_todos_los_datos": bool(
            not np.allclose(scaler.mean_, X.mean(axis=0))
        ),
        "max_dif_abs_media_scaler_vs_entrenamiento": float(
            np.max(np.abs(scaler.mean_ - X_tr.mean(axis=0)))
        ),
        "max_dif_abs_media_scaler_vs_todos_los_datos": float(
            np.max(np.abs(scaler.mean_ - X.mean(axis=0)))
        ),
        "media_global_atributos_test_estandarizados": float(X_te_s.mean()),
        "desv_est_global_atributos_test_estandarizados": float(X_te_s.std()),
    }

    # ------------------------------------------------------- modelos y métricas
    modelos = {
        "Naive Bayes gaussiano": GaussianNB(),
        "Regresión logística": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    }

    filas = []
    metricas = {}
    detalles = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_tr_s, y_tr)
        y_pred = modelo.predict(X_te_s)
        acc = float(accuracy_score(y_te, y_pred))
        f1m = float(f1_score(y_te, y_pred, average="macro"))
        filas.append({"Modelo": nombre, "Accuracy (prueba)": acc, "F1 macro (prueba)": f1m})
        metricas[nombre] = {"accuracy_prueba": acc, "f1_macro_prueba": f1m}
        detalles[nombre] = {
            "tipo": type(modelo).__name__,
            "parametros": {k: (str(v) if isinstance(v, np.ndarray) else v)
                           for k, v in modelo.get_params().items()},
            "n_ejemplos_entrenamiento": int(len(y_tr)),
            "n_ejemplos_prueba": int(len(y_te)),
        }

    tabla = pd.DataFrame(filas).set_index("Modelo")

    # ------------------------------------------------------------- salidas
    tabla.to_csv("tabla_parte2_metricas.csv", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    ax.axis("off")
    celdas = [
        [r["Modelo"], f"{r['Accuracy (prueba)']:.4f}", f"{r['F1 macro (prueba)']:.4f}"]
        for r in filas
    ]
    tb = ax.table(
        cellText=celdas,
        colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
        loc="center",
        cellLoc="center",
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 2.0)
    ax.set_title(
        "Parte 2 — Accuracy y F1 macro en el conjunto de prueba\n"
        "(división 70/30 estratificada, random_state=42; escalador ajustado solo con entrenamiento)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("tabla_parte2_metricas.png", dpi=120)
    plt.close(fig)

    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "Parte 2: Naive Bayes gaussiano y regresión logística (atributos "
            "estandarizados, escalador ajustado solo con el entrenamiento) "
            "entrenados sobre la división 70/30 de la Parte 1; accuracy y F1 "
            "macro medidos sobre el conjunto de prueba."
        ),
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "division_usada": {
            "test_size": TEST_SIZE,
            "estratificada_por_clase": True,
            "random_state": RANDOM_STATE,
            "tam_entrenamiento": int(len(y_tr)),
            "tam_prueba": int(len(y_te)),
            "casos_por_clase_entrenamiento": conteos_por_clase(y_tr, nombres_clases),
            "casos_por_clase_prueba": conteos_por_clase(y_te, nombres_clases),
            "verificacion_contra_T1": verif_split,
        },
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "atributos_estandarizados": True,
            "ajustado_solo_con_entrenamiento": True,
            "verificacion": verif_scaler,
        },
        "modelos": {
            "naive_bayes_gaussiano": {
                **detalles["Naive Bayes gaussiano"],
                **metricas["Naive Bayes gaussiano"],
            },
            "regresion_logistica": {
                **detalles["Regresión logística"],
                **metricas["Regresión logística"],
            },
        },
        "metricas_prueba": {
            "accuracy_naive_bayes_gaussiano": metricas["Naive Bayes gaussiano"]["accuracy_prueba"],
            "f1_macro_naive_bayes_gaussiano": metricas["Naive Bayes gaussiano"]["f1_macro_prueba"],
            "accuracy_regresion_logistica": metricas["Regresión logística"]["accuracy_prueba"],
            "f1_macro_regresion_logistica": metricas["Regresión logística"]["f1_macro_prueba"],
        },
        "tabla": {
            "columnas": ["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
            "filas": [
                [r["Modelo"], r["Accuracy (prueba)"], r["F1 macro (prueba)"]] for r in filas
            ],
        },
        "archivos_generados": [
            "resultados.json",
            "tabla_parte2_metricas.png",
            "tabla_parte2_metricas.csv",
        ],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------- resumen
    print("=" * 66)
    print("T2 — Parte 2: dos clasificadores (división 70/30 de la Parte 1)")
    print("=" * 66)
    print(f"Fuente de la división: {fuente_split}")
    print(f"Tamaños: entrenamiento={len(y_tr)}, prueba={len(y_te)}")
    if t1 is not None:
        claves = [k for k in verif_split if k.endswith("_con_T1")]
        ok = all(verif_split[k] for k in claves) if claves else True
        print(f"Verificación contra T1: {'OK' if ok else 'REVISAR'}")
        for k in claves:
            print(f"  - {k}: {verif_split[k]}")
    print("\nTabla de resultados (conjunto de prueba):")
    print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
    print(
        "\nEscalador ajustado solo con entrenamiento: "
        f"{verif_scaler['media_scaler_igual_a_media_del_entrenamiento']} "
        f"(dif. máx. vs. media de todos los datos: "
        f"{verif_scaler['max_dif_abs_media_scaler_vs_todos_los_datos']:.3e})"
    )
    print("Archivos generados: resultados.json, tabla_parte2_metricas.png, tabla_parte2_metricas.csv")


if __name__ == "__main__":
    main()
