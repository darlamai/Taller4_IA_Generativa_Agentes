# -*- coding: utf-8 -*-
"""
T2 — Tarea A (MMIA 6013): Generativo contra discriminativo con pocos datos.

Subtarea T2: entrenar un Naive Bayes gaussiano y una regresión logística
(con los atributos estandarizados; el escalador se ajusta SOLO con el
conjunto de entrenamiento) sobre la división 70/30 estratificada de la
Parte 1, y medir accuracy y F1 macro de cada modelo sobre el conjunto
de prueba, presentados en una tabla.

Entradas (subtarea previa T1, carpeta entrada/T1/):
    - train.csv, test.csv : la división única de la Parte 1
    - meta.json           : metadatos (opcional)
Salidas:
    - resultados.json       : contrato de la subtarea (4 cifras + tabla)
    - t2_tabla_metricas.png : tabla accuracy / F1 macro en prueba
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RUTA_T1 = Path("entrada") / "T1"


def cargar_meta():
    ruta = RUTA_T1 / "meta.json"
    if ruta.exists():
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def identificar_objetivo(df, meta):
    """Localiza la columna objetivo en el CSV generado por la Parte 1."""
    if isinstance(meta, dict):
        for clave in ("target_column", "target", "columna_objetivo",
                      "label_column", "clase", "label"):
            valor = meta.get(clave)
            if isinstance(valor, str) and valor in df.columns:
                return valor
    for candidato in ("target", "diagnosis", "class", "label", "y", "clase", "objetivo"):
        if candidato in df.columns:
            return candidato
    no_numericas = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(no_numericas) == 1:
        return no_numericas[0]
    return df.columns[-1]


def cargar_split(nombre_csv, meta):
    df = pd.read_csv(RUTA_T1 / nombre_csv)
    for col_indice in ("Unnamed: 0", "index"):
        if col_indice in df.columns:
            df = df.drop(columns=[col_indice])
    objetivo = identificar_objetivo(df, meta)
    y = df[objetivo]
    X = df.drop(columns=[objetivo]).select_dtypes(include="number")
    return X, y, objetivo


def split_fallback():
    """Recrea la división de la Parte 1 con los mismos parámetros exactos
    (70/30 estratificada, random_state=42) si los CSV no estuvieran disponibles."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, train_size=0.7, stratify=y, random_state=42
    )
    return X_tr, y_tr, X_te, y_te, "target"


def main():
    meta = cargar_meta()

    # --- 1) Cargar la división única de la Parte 1 --------------------------
    try:
        X_train, y_train, col_objetivo = cargar_split("train.csv", meta)
        X_test, y_test, _ = cargar_split("test.csv", meta)
        origen = "Parte 1 (entrada/T1): 70%/30% estratificada, random_state=42"
    except (FileNotFoundError, OSError):
        X_train, y_train, X_test, y_test, col_objetivo = split_fallback()
        origen = ("recreada con load_breast_cancer y train_test_split"
                  "(test_size=0.3, stratify=y, random_state=42), mismos parámetros de la Parte 1")

    # Verificación de consistencia con la subtarea T1
    t1 = {}
    ruta_t1 = RUTA_T1 / "resultados.json"
    if ruta_t1.exists():
        try:
            t1 = json.loads(ruta_t1.read_text(encoding="utf-8"))
        except Exception:
            t1 = {}
    coincide_t1 = bool(t1) and (t1.get("n_train") == len(X_train)
                                and t1.get("n_test") == len(X_test))

    # --- 2) Estandarización: el escalador se ajusta SOLO con entrenamiento --
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)   # fit únicamente con train
    X_test_std = scaler.transform(X_test)         # test solo se transforma

    # --- 3) Entrenamiento de los dos clasificadores -------------------------
    nb = GaussianNB()
    nb.fit(X_train_std, y_train)
    pred_nb = nb.predict(X_test_std)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_std, y_train)
    pred_lr = lr.predict(X_test_std)

    # --- 4) Métricas sobre el conjunto de prueba ----------------------------
    filas = []
    for modelo, pred in (("Naive Bayes gaussiano", pred_nb),
                         ("Regresión logística", pred_lr)):
        filas.append({
            "modelo": modelo,
            "accuracy_prueba": float(accuracy_score(y_test, pred)),
            "f1_macro_prueba": float(f1_score(y_test, pred, average="macro")),
        })
    tabla = pd.DataFrame(filas).set_index("modelo")

    # --- 5) Tabla como figura PNG -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.axis("off")
    celdas = [[f"{f['accuracy_prueba']:.4f}", f"{f['f1_macro_prueba']:.4f}"] for f in filas]
    tb = ax.table(
        cellText=celdas,
        rowLabels=[f["modelo"] for f in filas],
        colLabels=["Accuracy (prueba)", "F1 macro (prueba)"],
        cellLoc="center",
        loc="center",
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 2.0)
    ax.set_title(
        "Parte 2 — Accuracy y F1 macro en el conjunto de prueba\n"
        "(división 70/30 estratificada, random_state=42; escalador ajustado solo con train)",
        fontsize=10,
    )
    plt.savefig("t2_tabla_metricas.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- 6) Contrato: resultados.json ---------------------------------------
    nb_fila, lr_fila = filas
    resultados = {
        "subtarea": "T2",
        "tarea": "Tarea A — Generativo contra discriminativo con pocos datos",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "division": {
            "origen": origen,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "coincide_con_T1": bool(coincide_t1),
        },
        "columna_objetivo": str(col_objetivo),
        "preprocesamiento": {
            "estandarizacion": "StandardScaler ajustado únicamente con el conjunto de entrenamiento",
            "aplicado_a": ["Naive Bayes gaussiano", "Regresión logística"],
        },
        "modelos": {
            "naive_bayes_gaussiano": {
                "accuracy_test": nb_fila["accuracy_prueba"],
                "f1_macro_test": nb_fila["f1_macro_prueba"],
            },
            "regresion_logistica": {
                "accuracy_test": lr_fila["accuracy_prueba"],
                "f1_macro_test": lr_fila["f1_macro_prueba"],
            },
        },
        "tabla": filas,
        "accuracy_naive_bayes_gaussiano": nb_fila["accuracy_prueba"],
        "f1_macro_naive_bayes_gaussiano": nb_fila["f1_macro_prueba"],
        "accuracy_regresion_logistica": lr_fila["accuracy_prueba"],
        "f1_macro_regresion_logistica": lr_fila["f1_macro_prueba"],
        "archivos_generados": {
            "tabla_png": "t2_tabla_metricas.png",
        },
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 7) Resumen breve ----------------------------------------------------
    print("T2 — Accuracy y F1 macro en el conjunto de prueba (división 70/30 de la Parte 1)")
    print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nn_train={len(X_train)}  n_test={len(X_test)}  (coincide con T1: {coincide_t1})")
    print("Escalador: StandardScaler ajustado solo con el entrenamiento.")
    print("Archivos generados: resultados.json, t2_tabla_metricas.png")


if __name__ == "__main__":
    main()
