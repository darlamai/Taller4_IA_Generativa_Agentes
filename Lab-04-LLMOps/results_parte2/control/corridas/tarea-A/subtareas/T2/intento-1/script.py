# T2 — Entrenar GaussianNB y regresión logística (atributos estandarizados) sobre el
# entrenamiento de T1 y evaluar ambos sobre el conjunto de prueba (división 70/30 de T1).

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RUTA_T1 = Path("entrada") / "T1"
RANDOM_STATE = 42
TEST_SIZE = 0.3


def leer_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def obtener_division(y):
    """Recupera la división de T1: usa índices guardados si existen; si no,
    recrea la división 70/30 estratificada con random_state=42 (determinista)."""
    info_t1 = leer_json(RUTA_T1 / "resultados.json") or {}
    div_t1 = leer_json(RUTA_T1 / "division_train_test.json") or {}

    # 1) Intentar usar índices guardados por T1
    pares = [("indices_train", "indices_test"), ("train_indices", "test_indices"),
             ("idx_train", "idx_test"), ("indices_entrenamiento", "indices_prueba")]
    if isinstance(div_t1, dict):
        for ci, cj in pares:
            if ci in div_t1 and cj in div_t1:
                try:
                    itr = np.asarray(div_t1[ci], dtype=int)
                    ite = np.asarray(div_t1[cj], dtype=int)
                    if len(itr) > 0 and len(ite) > 0 and itr.max() < len(y) and ite.max() < len(y):
                        return itr, ite, f"índices guardados en entrada/T1/division_train_test.json ({ci}/{cj})", info_t1
                except Exception:
                    pass

    # 2) Recrear la división con los parámetros registrados por T1
    d = info_t1.get("division_train_test", {}) if isinstance(info_t1, dict) else {}
    if not isinstance(d, dict):
        d = {}
    test_size = float(d.get("test_size", TEST_SIZE))
    random_state = int(d.get("random_state", RANDOM_STATE))
    estratificada = bool(d.get("estratificada_por_clase", True))
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=test_size, random_state=random_state,
                                stratify=y if estratificada else None)
    origen = (f"recreada con test_size={test_size}, random_state={random_state}, "
              f"estratificada={estratificada} (determinista, idéntica a T1)")
    return itr, ite, origen, info_t1


def main():
    # ---------- Datos y división de T1 ----------
    data = load_breast_cancer()
    X, y = data.data, data.target

    idx_train, idx_test, origen, info_t1 = obtener_division(y)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # Verificación contra lo reportado por T1
    d_t1 = info_t1.get("division_train_test", {}) if isinstance(info_t1, dict) else {}
    n_tr_esp, n_te_esp = d_t1.get("n_train"), d_t1.get("n_test")
    conteo_tr = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    conteo_te = {int(c): int(n) for c, n in zip(*np.unique(y_test, return_counts=True))}
    coincide = True
    if n_tr_esp is not None:
        coincide &= (len(idx_train) == int(n_tr_esp))
    if n_te_esp is not None:
        coincide &= (len(idx_test) == int(n_te_esp))
    if d_t1.get("conteo_por_clase_train"):
        esp = {0: d_t1["conteo_por_clase_train"].get("clase_0_malignant"),
               1: d_t1["conteo_por_clase_train"].get("clase_1_benign")}
        coincide &= all(esp[c] is None or esp[c] == conteo_tr[c] for c in (0, 1))
    if d_t1.get("conteo_por_clase_test"):
        esp = {0: d_t1["conteo_por_clase_test"].get("clase_0_malignant"),
               1: d_t1["conteo_por_clase_test"].get("clase_1_benign")}
        coincide &= all(esp[c] is None or esp[c] == conteo_te[c] for c in (0, 1))

    # ---------- Modelo 1: GaussianNB (atributos originales) ----------
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)

    # ---------- Modelo 2: regresión logística con estandarización ----------
    # El escalador se ajusta SOLO con el conjunto de entrenamiento.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    logreg.fit(X_train_s, y_train)
    pred_lr = logreg.predict(X_test_s)

    # Sanidad: GaussianNB es invariante a transformaciones afines por atributo
    nb_esc = GaussianNB().fit(X_train_s, y_train)
    nb_invariante = bool(np.array_equal(nb_esc.predict(X_test_s), pred_nb))

    # ---------- Métricas sobre el conjunto de prueba ----------
    metricas = {
        "Naive Bayes gaussiano": {
            "accuracy": float(accuracy_score(y_test, pred_nb)),
            "f1_macro": float(f1_score(y_test, pred_nb, average="macro")),
        },
        "Regresión logística": {
            "accuracy": float(accuracy_score(y_test, pred_lr)),
            "f1_macro": float(f1_score(y_test, pred_lr, average="macro")),
        },
    }
    tabla = [{"modelo": m, "accuracy": v["accuracy"], "f1_macro": v["f1_macro"]}
             for m, v in metricas.items()]

    lineas = ["| Modelo | Accuracy (test) | F1 macro (test) |", "|---|---|---|"]
    for fila in tabla:
        lineas.append(f"| {fila['modelo']} | {fila['accuracy']:.4f} | {fila['f1_macro']:.4f} |")
    tabla_md = "\n".join(lineas)

    cm_nb = confusion_matrix(y_test, pred_nb).tolist()
    cm_lr = confusion_matrix(y_test, pred_lr).tolist()

    # ---------- Figura ----------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    etiquetas = ["GaussianNB", "Reg. logística"]
    x = np.arange(len(etiquetas))
    width = 0.35
    acc = [metricas[m]["accuracy"] for m in metricas]
    f1m = [metricas[m]["f1_macro"] for m in metricas]
    ax.bar(x - width / 2, acc, width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, f1m, width, label="F1 macro", color="#DD8452")
    for i, (a, f) in enumerate(zip(acc, f1m)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor en el conjunto de prueba")
    ax.set_title("Parte 2 — Desempeño en prueba (división 70/30 estratificada de T1)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("t2_metricas_test.png", dpi=120)
    plt.close(fig)

    # ---------- Contrato: resultados.json ----------
    resultados = {
        "subtarea": "T2",
        "descripcion": ("GaussianNB y regresión logística (atributos estandarizados con "
                        "StandardScaler ajustado solo con el entrenamiento) evaluados sobre "
                        "el conjunto de prueba de la división 70/30 estratificada de T1."),
        "origen_division": origen,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "conteo_por_clase_train": {f"clase_0_malignant": conteo_tr.get(0, 0),
                                   "clase_1_benign": conteo_tr.get(1, 0)},
        "conteo_por_clase_test": {f"clase_0_malignant": conteo_te.get(0, 0),
                                  "clase_1_benign": conteo_te.get(1, 0)},
        "verificacion_division_contra_T1": {
            "n_train_esperado": n_tr_esp, "n_test_esperado": n_te_esp,
            "coincide": bool(coincide),
        },
        "preprocesamiento": {
            "gaussian_nb": "atributos originales del entrenamiento (sin escalar)",
            "regresion_logistica": "StandardScaler ajustado únicamente con el entrenamiento",
        },
        "parametros_modelos": {
            "GaussianNB": "valores por defecto de sklearn",
            "LogisticRegression": {"solver": "lbfgs", "max_iter": 5000, "random_state": RANDOM_STATE},
            "StandardScaler": "valores por defecto (ajustado solo con X_train)",
        },
        "tabla_metricas_test": tabla,
        "metricas_por_modelo": metricas,
        "tabla_markdown": tabla_md,
        "matrices_de_confusion_test": {
            "gaussian_nb": {"clases": [0, 1], "matriz": cm_nb},
            "regresion_logistica": {"clases": [0, 1], "matriz": cm_lr},
        },
        "nota_gaussian_nb_invariante_a_escala": nb_invariante,
        "figura": "t2_metricas_test.png",
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ---------- Resumen ----------
    print("T2 — GaussianNB vs Regresión logística (conjunto de prueba, división 70/30 de T1)")
    print(f"División: {origen}")
    print(f"n_train={len(idx_train)}, n_test={len(idx_test)} | coincide con T1: {coincide}")
    for fila in tabla:
        print(f"  {fila['modelo']}: accuracy={fila['accuracy']:.4f} | F1 macro={fila['f1_macro']:.4f}")
    print("Archivos generados: resultados.json, t2_metricas_test.png")


if __name__ == "__main__":
    main()
