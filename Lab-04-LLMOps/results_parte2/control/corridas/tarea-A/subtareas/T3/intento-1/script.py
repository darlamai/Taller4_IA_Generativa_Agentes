#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje (Parte 3 del enunciado).

Sobre la MISMA división 70/30 estratificada de T1 (random_state=42), entrena
GaussianNB y Regresión logística con submuestras estratificadas del 5 %, 10 %,
25 %, 50 % y 100 % del entrenamiento (random_state=42), evalúa cada tamaño
siempre sobre el MISMO conjunto de prueba, y grafica exactitud contra número
de ejemplos de entrenamiento (una curva por modelo), guardando la figura PNG.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

SEMILLA = 42
PORCENTAJES = [5, 10, 25, 50, 100]
NOMBRE_FIGURA = "t3_curva_aprendizaje.png"
RUTA_DIVISION_T1 = Path("entrada/T1/division_train_test.json")


def cargar_division_t1(y):
    """Devuelve (train_idx, test_idx, origen) de la división única de T1."""
    if RUTA_DIVISION_T1.exists():
        try:
            with open(RUTA_DIVISION_T1, "r", encoding="utf-8") as f:
                div = json.load(f)
            bloque = div.get("division_train_test", div)
            train_idx = np.asarray(bloque["train_indices"], dtype=int)
            test_idx = np.asarray(bloque["test_indices"], dtype=int)
            if len(train_idx) > 0 and len(test_idx) > 0:
                return train_idx, test_idx, "entrada/T1/division_train_test.json"
        except Exception:
            pass
    # Respaldo: recrear exactamente la división de la Parte 1
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, stratify=y, random_state=SEMILLA
    )
    return (train_idx, test_idx,
            "división recreada: train_test_split(test_size=0.3, stratify=y, random_state=42)")


def main():
    # ----------------- Datos y división única de T1 -----------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    train_idx, test_idx, origen_division = cargar_division_t1(y)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    n_train_total, n_test = int(len(y_train)), int(len(y_test))

    # ----------------- Curva de aprendizaje -----------------
    modelos = ["Naive Bayes gaussiano", "Regresión logística"]
    curvas = {m: {"porcentajes": [], "n_ejemplos": [], "accuracy": [], "f1_macro": []}
              for m in modelos}
    tabla = []
    composicion = []

    for pct in PORCENTAJES:
        # Submuestra estratificada del entrenamiento (random_state=42)
        if pct == 100:
            X_sub, y_sub = X_train, y_train
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train,
                train_size=pct / 100.0,
                stratify=y_train,
                random_state=SEMILLA,
            )
        n_sub = int(len(y_sub))
        composicion.append({
            "porcentaje": pct,
            "n_ejemplos": n_sub,
            "clase_0_malignant": int((y_sub == 0).sum()),
            "clase_1_benign": int((y_sub == 1).sum()),
        })

        # Generativo: GaussianNB con atributos originales (igual que en T2)
        nb = GaussianNB().fit(X_sub, y_sub)
        pred_nb = nb.predict(X_test)
        acc_nb = float(accuracy_score(y_test, pred_nb))
        f1_nb = float(f1_score(y_test, pred_nb, average="macro"))

        # Discriminativo: Regresión logística; escalador ajustado SOLO con la
        # submuestra de entrenamiento (el conjunto de prueba nunca se usa para ajustar)
        scaler = StandardScaler().fit(X_sub)
        logreg = LogisticRegression(solver="lbfgs", max_iter=5000,
                                    random_state=SEMILLA)
        logreg.fit(scaler.transform(X_sub), y_sub)
        pred_lr = logreg.predict(scaler.transform(X_test))
        acc_lr = float(accuracy_score(y_test, pred_lr))
        f1_lr = float(f1_score(y_test, pred_lr, average="macro"))

        for modelo, acc, f1 in (("Naive Bayes gaussiano", acc_nb, f1_nb),
                                ("Regresión logística", acc_lr, f1_lr)):
            curvas[modelo]["porcentajes"].append(pct)
            curvas[modelo]["n_ejemplos"].append(n_sub)
            curvas[modelo]["accuracy"].append(acc)
            curvas[modelo]["f1_macro"].append(f1)
            tabla.append({
                "porcentaje_entrenamiento": pct,
                "n_ejemplos_entrenamiento": n_sub,
                "modelo": modelo,
                "accuracy_test": acc,
                "f1_macro_test": f1,
            })

    # Diferencia de exactitud (NB - LR) por tamaño (insumo factual para la Parte 4)
    diffs = {
        str(p): curvas["Naive Bayes gaussiano"]["accuracy"][i]
              - curvas["Regresión logística"]["accuracy"][i]
        for i, p in enumerate(PORCENTAJES)
    }

    # ----------------- Figura: curva de aprendizaje -----------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    estilos = {"Naive Bayes gaussiano": dict(marker="o", color="tab:blue"),
               "Regresión logística": dict(marker="s", color="tab:orange")}
    n_vals = curvas["Naive Bayes gaussiano"]["n_ejemplos"]

    for modelo in modelos:
        c = curvas[modelo]
        ax.plot(c["n_ejemplos"], c["accuracy"], label=modelo,
                linewidth=2, markersize=7, **estilos[modelo])
        desplaz = 9 if modelo == "Naive Bayes gaussiano" else -15
        for xi, yi in zip(c["n_ejemplos"], c["accuracy"]):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, desplaz), ha="center", fontsize=8)

    ax.set_xticks(n_vals)
    ax.set_xticklabels([f"{n}\n({p}%)" for n, p in zip(n_vals, PORCENTAJES)])
    todos = [a for m in modelos for a in curvas[m]["accuracy"]]
    ax.set_ylim(max(0.45, min(todos) - 0.07), 1.02)
    ax.set_xlabel("Número de ejemplos de entrenamiento (porcentaje del entrenamiento de T1)")
    ax.set_ylabel("Exactitud en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje: Naive Bayes gaussiano vs. Regresión logística\n"
                 "Breast Cancer Wisconsin — división 70/30 de T1; prueba fija (n=171)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(NOMBRE_FIGURA, dpi=120)
    plt.close(fig)

    # ----------------- Tabla markdown -----------------
    lineas = ["| % entrenamiento | n ejemplos | Modelo | Accuracy (test) | F1 macro (test) |",
              "|---|---|---|---|---|"]
    for t in tabla:
        lineas.append(f"| {t['porcentaje_entrenamiento']}% | "
                      f"{t['n_ejemplos_entrenamiento']} | {t['modelo']} | "
                      f"{t['accuracy_test']:.4f} | {t['f1_macro_test']:.4f} |")
    tabla_md = "\n".join(lineas)

    # ----------------- resultados.json (contrato) -----------------
    resultados = {
        "subtarea": "T3",
        "descripcion": ("Curva de aprendizaje: GaussianNB y regresión logística entrenados con "
                        "submuestras estratificadas del 5/10/25/50/100 % del entrenamiento de la "
                        "división 70/30 de T1 (random_state=42) y evaluados siempre sobre el mismo "
                        "conjunto de prueba."),
        "origen_division": origen_division,
        "verificacion_division_contra_T1": {
            "n_train_esperado": 398,
            "n_test_esperado": 171,
            "n_train": n_train_total,
            "n_test": n_test,
            "coincide": bool(n_train_total == 398 and n_test == 171),
        },
        "porcentajes": PORCENTAJES,
        "random_state_submuestreo": SEMILLA,
        "preprocesamiento": {
            "gaussian_nb": "atributos originales del entrenamiento (sin escalar), igual que en T2",
            "regresion_logistica": "StandardScaler ajustado únicamente con cada submuestra de entrenamiento",
        },
        "parametros_modelos": {
            "GaussianNB": "valores por defecto de sklearn",
            "LogisticRegression": {"solver": "lbfgs", "max_iter": 5000, "random_state": 42},
            "StandardScaler": "valores por defecto (ajustado solo con la submuestra de entrenamiento)",
        },
        "composicion_submuestras": composicion,
        "curva_aprendizaje": curvas,
        "tabla_por_tamano": tabla,
        "diferencia_accuracy_NB_menos_LR_por_porcentaje": diffs,
        "tabla_markdown": tabla_md,
        "figura": NOMBRE_FIGURA,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ----------------- Resumen -----------------
    print("=" * 70)
    print("T3 — Curva de aprendizaje (evaluación fija en el conjunto de prueba)")
    print("=" * 70)
    print(f"División usada: {origen_division}")
    print(f"n_train={n_train_total}, n_test={n_test}")
    print()
    print(tabla_md)
    print()
    print("Diferencia de exactitud (NB - LR) por porcentaje de entrenamiento:")
    for p in PORCENTAJES:
        print(f"  {p:>3}%: {diffs[str(p)]:+.4f}")
    print(f"\nFigura guardada: {NOMBRE_FIGURA}")
    print("Resultados escritos en resultados.json")


if __name__ == "__main__":
    main()
