#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T3 — Parte 3: curva de aprendizaje (Ng & Jordan con datos reales).

Sobre la MISMA división 70/30 estratificada de la Parte 1 (random_state=42),
entrena el Naive Bayes gaussiano y la regresión logística de la Parte 2 con
5%, 10%, 25%, 50% y 100% del conjunto de entrenamiento (submuestras
estratificadas, random_state=42), evalúa cada punto siempre sobre el MISMO
conjunto de prueba, dibuja la exactitud frente al número de ejemplos de
entrenamiento (una curva por modelo) y guarda la figura como PNG.
Todas las cifras se escriben en resultados.json.
"""

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
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
PORCENTAJES = [5, 10, 25, 50, 100]
FIGURA = "curva_aprendizaje_parte3.png"


def contar_clases(y):
    """Casos por clase (0 = malignant, 1 = benign)."""
    c = np.bincount(np.asarray(y, dtype=int), minlength=2)
    return {"malignant": int(c[0]), "benign": int(c[1])}


def main():
    # ------------------------------------------------------------------
    # 1) Misma división de la Parte 1: 70/30 estratificada, random_state=42
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    n_train_total, n_test = int(X_train.shape[0]), int(X_test.shape[0])
    clases_train, clases_test = contar_clases(y_train), contar_clases(y_test)

    # Verificación contra la división reportada por T1
    verif_t1 = {"archivo": "entrada/T1/resultados.json", "encontrado": False}
    p_t1 = Path("entrada") / "T1" / "resultados.json"
    if p_t1.exists():
        try:
            d1 = json.loads(p_t1.read_text(encoding="utf-8")).get("division", {})
            verif_t1.update(
                encontrado=True,
                tam_entrenamiento_coincide=(d1.get("tam_entrenamiento") == n_train_total),
                tam_prueba_coincide=(d1.get("tam_prueba") == n_test),
                casos_por_clase_entrenamiento_coincide=(d1.get("casos_por_clase_entrenamiento") == clases_train),
                casos_por_clase_prueba_coincide=(d1.get("casos_por_clase_prueba") == clases_test),
            )
        except Exception as exc:
            verif_t1["error"] = repr(exc)

    # ------------------------------------------------------------------
    # 2) Preprocesamiento de la Parte 2: StandardScaler ajustado SOLO con
    #    el conjunto de entrenamiento completo (398 ejemplos), igual que en
    #    la Parte 2. Las submuestras se toman de los atributos ya
    #    estandarizados y el conjunto de prueba se transforma siempre con
    #    ese mismo escalador (nunca se ajusta con datos de prueba).
    # ------------------------------------------------------------------
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 3) Curva de aprendizaje: 5%, 10%, 25%, 50% y 100% del entrenamiento
    #    (submuestras estratificadas, random_state=42); prueba fija.
    # ------------------------------------------------------------------
    curva = {
        "naive_bayes_gaussiano": {
            "n_ejemplos_entrenamiento": [], "accuracy_prueba": [],
            "f1_macro_prueba": [], "casos_por_clase_submuestra": [],
        },
        "regresion_logistica": {
            "n_ejemplos_entrenamiento": [], "accuracy_prueba": [],
            "f1_macro_prueba": [],
        },
    }
    filas = []
    mejor_por_tamano = []

    for pct, frac in zip(PORCENTAJES, FRACCIONES):
        if frac >= 1.0:
            X_sub, y_sub = X_train_std, y_train
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_train_std, y_train,
                train_size=frac, stratify=y_train, random_state=RANDOM_STATE,
            )
        n_sub = int(len(y_sub))

        nb = GaussianNB().fit(X_sub, y_sub)
        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(X_sub, y_sub)

        pred_nb, pred_lr = nb.predict(X_test_std), lr.predict(X_test_std)
        acc_nb = float(accuracy_score(y_test, pred_nb))
        acc_lr = float(accuracy_score(y_test, pred_lr))
        f1_nb = float(f1_score(y_test, pred_nb, average="macro"))
        f1_lr = float(f1_score(y_test, pred_lr, average="macro"))

        curva["naive_bayes_gaussiano"]["n_ejemplos_entrenamiento"].append(n_sub)
        curva["naive_bayes_gaussiano"]["accuracy_prueba"].append(acc_nb)
        curva["naive_bayes_gaussiano"]["f1_macro_prueba"].append(f1_nb)
        curva["naive_bayes_gaussiano"]["casos_por_clase_submuestra"].append(contar_clases(y_sub))
        curva["regresion_logistica"]["n_ejemplos_entrenamiento"].append(n_sub)
        curva["regresion_logistica"]["accuracy_prueba"].append(acc_lr)
        curva["regresion_logistica"]["f1_macro_prueba"].append(f1_lr)

        filas.append([pct, n_sub, acc_nb, acc_lr])
        mejor_por_tamano.append({
            "porcentaje": pct, "n_ejemplos": n_sub,
            "modelo_con_mayor_accuracy": ("naive_bayes_gaussiano" if acc_nb >= acc_lr
                                          else "regresion_logistica"),
        })

    n_lista = curva["naive_bayes_gaussiano"]["n_ejemplos_entrenamiento"]
    acc_nb_lista = curva["naive_bayes_gaussiano"]["accuracy_prueba"]
    acc_lr_lista = curva["regresion_logistica"]["accuracy_prueba"]

    # Verificación del punto 100% contra los resultados de la Parte 2 (T2)
    verif_t2 = {"archivo": "entrada/T2/resultados.json", "encontrado": False}
    p_t2 = Path("entrada") / "T2" / "resultados.json"
    if p_t2.exists():
        try:
            m2 = json.loads(p_t2.read_text(encoding="utf-8")).get("metricas_prueba", {})
            verif_t2.update(
                encontrado=True,
                accuracy_nb_100pct_coincide_con_T2=bool(
                    abs(acc_nb_lista[-1] - m2.get("accuracy_naive_bayes_gaussiano", float("nan"))) < 1e-12),
                accuracy_lr_100pct_coincide_con_T2=bool(
                    abs(acc_lr_lista[-1] - m2.get("accuracy_regresion_logistica", float("nan"))) < 1e-12),
            )
        except Exception as exc:
            verif_t2["error"] = repr(exc)

    # ------------------------------------------------------------------
    # 4) Figura: exactitud vs número de ejemplos de entrenamiento
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(n_lista, acc_nb_lista, marker="o", color="tab:blue",
            label="Naive Bayes gaussiano (generativo)")
    ax.plot(n_lista, acc_lr_lista, marker="s", color="tab:red",
            label="Regresión logística (discriminativo)")
    for n, a in zip(n_lista, acc_nb_lista):
        ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")
    for n, a in zip(n_lista, acc_lr_lista):
        ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="tab:red")
    ax.set_xticks(n_lista)
    ax.set_xticklabels([str(n) for n in n_lista])
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title(
        "Curva de aprendizaje — Parte 3\n"
        f"Prueba fija de la Parte 1 ({n_test} ejemplos); submuestras estratificadas "
        f"del entrenamiento ({n_train_total} ejemplos)"
    )
    todos = acc_nb_lista + acc_lr_lista
    ax.set_ylim(max(0.0, min(todos) - 0.06), 1.015)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    salida = {
        "subtarea": "T3",
        "descripcion": (
            "Parte 3: curva de aprendizaje. Naive Bayes gaussiano y regresión logística "
            "entrenados con 5%, 10%, 25%, 50% y 100% del conjunto de entrenamiento de la "
            "Parte 1 (submuestras estratificadas, random_state=42) y evaluados siempre "
            "sobre el mismo conjunto de prueba de la Parte 1."
        ),
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "division_usada": {
            "test_size": 0.3,
            "estratificada_por_clase": True,
            "random_state": 42,
            "tam_entrenamiento": n_train_total,
            "tam_prueba": n_test,
            "casos_por_clase_entrenamiento": clases_train,
            "casos_por_clase_prueba": clases_test,
            "identica_a_la_division_de_T1": (
                bool(verif_t1.get("tam_entrenamiento_coincide")
                     and verif_t1.get("tam_prueba_coincide")
                     and verif_t1.get("casos_por_clase_entrenamiento_coincide")
                     and verif_t1.get("casos_por_clase_prueba_coincide"))
                if verif_t1.get("encontrado") else None
            ),
        },
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "ajustado_solo_con": (
                "conjunto de entrenamiento completo de la Parte 1 "
                f"({n_train_total} ejemplos), igual que en la Parte 2"
            ),
            "submuestras": (
                "tomadas sobre los atributos ya estandarizados; el conjunto de prueba "
                "se transforma siempre con el mismo escalador (nunca ajustado con prueba)"
            ),
        },
        "porcentajes_entrenamiento": PORCENTAJES,
        "fracciones_entrenamiento": FRACCIONES,
        "conjunto_de_prueba_fijo": {"n_ejemplos": n_test, "usado_para_todos_los_puntos": True},
        "curva_aprendizaje": curva,
        "tabla": {
            "columnas": [
                "% del entrenamiento", "n_ejemplos_entrenamiento",
                "accuracy_prueba_naive_bayes_gaussiano",
                "accuracy_prueba_regresion_logistica",
            ],
            "filas": filas,
        },
        "modelo_con_mayor_accuracy_por_tamano": mejor_por_tamano,
        "figura_png": FIGURA,
        "verificaciones": {
            "division_contra_T1": verif_t1,
            "punto_100pct_contra_T2": verif_t2,
        },
    }
    Path("resultados.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T3 — Curva de aprendizaje (Parte 3) ===")
    print(f"División de la Parte 1: {n_train_total} entrenamiento / {n_test} prueba "
          f"(estratificada, random_state={RANDOM_STATE})")
    print(f"{'% ent.':>7} {'n_ent':>6} {'acc NB':>9} {'acc LR':>9}")
    for pct, n_sub, a_nb, a_lr in filas:
        print(f"{pct:>6}% {n_sub:>6} {a_nb:>9.4f} {a_lr:>9.4f}")
    print(f"Figura guardada en: {FIGURA}")
    print("Resultados escritos en: resultados.json")
    if verif_t1.get("encontrado"):
        print(f"División idéntica a la de T1: {salida['division_usada']['identica_a_la_division_de_T1']}")
    if verif_t2.get("encontrado"):
        print(f"Punto 100% coincide con T2 -> NB: {verif_t2.get('accuracy_nb_100pct_coincide_con_T2')}, "
              f"LR: {verif_t2.get('accuracy_lr_100pct_coincide_con_T2')}")


if __name__ == "__main__":
    main()
