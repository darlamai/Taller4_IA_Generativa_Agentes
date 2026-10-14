import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cargar_distribucion_renormalizada():
    """Carga la distribución renormalizada de la Parte 2 (T2); si no está
    disponible, la recalcula de forma determinista a partir de los logits."""
    ruta_t2 = Path("entrada/T2/resultados.json")
    if ruta_t2.exists():
        with ruta_t2.open("r", encoding="utf-8") as f:
            res_t2 = json.load(f)
        tokens = list(res_t2["tokens"])
        p_teo = np.asarray(res_t2["distribucion_renormalizada_lista"], dtype=float)
        fuente = "entrada/T2/resultados.json (distribucion renormalizada top-p p=0.9 sobre T=1)"
        return tokens, p_teo, fuente

    # Fallback determinista: softmax(T=1) + top-p p=0.9 + renormalización
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()  # softmax numéricamente estable, T = 1
    p1 = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p1)
    acum = np.cumsum(p1[orden])
    k = int(np.searchsorted(acum, 0.9) + 1)  # conjunto mínimo que alcanza 0.9
    nucleo = orden[:k]
    p_teo = np.zeros_like(p1)
    p_teo[nucleo] = p1[nucleo] / p1[nucleo].sum()
    fuente = "recalculada en T3 (softmax T=1 + top-p p=0.9 + renormalizacion)"
    return tokens, p_teo, fuente


def main():
    # ------------------------------------------------------------------
    # 1) Distribución teórica: renormalizada de la Parte 2
    # ------------------------------------------------------------------
    tokens, p_teo, fuente = cargar_distribucion_renormalizada()
    p_teo = p_teo / p_teo.sum()  # seguridad numérica (no-op si ya suma 1)

    # ------------------------------------------------------------------
    # 2) Muestreo: 10,000 muestras con numpy.random.default_rng(0)
    # ------------------------------------------------------------------
    n_muestras = 10_000
    rng = np.random.default_rng(0)
    muestras = rng.choice(len(tokens), size=n_muestras, p=p_teo)

    conteos = np.bincount(muestras, minlength=len(tokens))
    frec_obs = conteos / n_muestras

    # ------------------------------------------------------------------
    # 3) Divergencia KL (en bits) de la empírica respecto de la teórica
    #    KL(emp || teo) = Σ_i p_emp(i) * log2(p_emp(i) / p_teo(i))
    #    convención: términos con p_emp(i) = 0 aportan 0
    # ------------------------------------------------------------------
    mask = frec_obs > 0
    if np.any(p_teo[mask] <= 0.0):
        kl_bits = float("inf")
        kl_nats = float("inf")
    else:
        kl_bits = float(np.sum(frec_obs[mask] * np.log2(frec_obs[mask] / p_teo[mask])))
        kl_nats = float(np.sum(frec_obs[mask] * np.log(frec_obs[mask] / p_teo[mask])))

    # ------------------------------------------------------------------
    # 4) Gráfico de barras: frecuencias observadas vs probabilidades teóricas
    # ------------------------------------------------------------------
    x = np.arange(len(tokens))
    ancho = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    b1 = ax.bar(x - ancho / 2, frec_obs, width=ancho,
                label="Frecuencia observada (10,000 muestras)",
                color="#4C72B0", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + ancho / 2, p_teo, width=ancho,
                label="Probabilidad teórica (renormalizada, top-p 0.9)",
                color="#DD8452", edgecolor="black", linewidth=0.6)
    for barras in (b1, b2):
        for b in barras:
            h = b.get_height()
            ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.set_xlabel("Token")
    ax.set_ylabel("Probabilidad / frecuencia relativa")
    ax.set_ylim(0, max(frec_obs.max(), p_teo.max()) * 1.20)
    ax.set_title("Parte 3 — Comprobación empírica\n"
                 "Frecuencias observadas vs. probabilidades teóricas "
                 f"(n={n_muestras:,}, default_rng(0))\n"
                 f"KL(emp||teo) = {kl_bits:.6f} bits")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("parte3_frecuencias_vs_teoricas.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    tabla = [
        {
            "token": tok,
            "probabilidad_teorica": float(p_teo[i]),
            "conteo_observado": int(conteos[i]),
            "frecuencia_observada": float(frec_obs[i]),
            "diferencia_obs_menos_teo": float(frec_obs[i] - p_teo[i]),
        }
        for i, tok in enumerate(tokens)
    ]

    resultados = {
        "subtarea": ("T3 - Comprobacion empirica: 10,000 muestras de la "
                     "distribucion renormalizada de la Parte 2 (top-p p=0.9, T=1)"),
        "n_muestras": n_muestras,
        "semilla": 0,
        "generador": "numpy.random.default_rng(0)",
        "fuente_distribucion_renormalizada": fuente,
        "tokens": tokens,
        "probabilidades_teoricas_sin_redondear": p_teo.tolist(),
        "conteos_observados": conteos.tolist(),
        "frecuencias_observadas_sin_redondear": frec_obs.tolist(),
        "tabla_comparativa_sin_redondear": tabla,
        "tabla_comparativa_4_decimales": [
            {
                "token": t["token"],
                "probabilidad_teorica": f"{t['probabilidad_teorica']:.4f}",
                "conteo_observado": t["conteo_observado"],
                "frecuencia_observada": f"{t['frecuencia_observada']:.4f}",
                "diferencia": f"{t['diferencia_obs_menos_teo']:+.4f}",
            }
            for t in tabla
        ],
        "divergencia_KL_bits": kl_bits,
        "divergencia_KL_nats": kl_nats,
        "convencion_KL": ("KL(emp||teo) = sum_i p_emp(i)*log2(p_emp(i)/p_teo(i)); "
                          "terminos con p_emp(i)=0 aportan 0"),
        "suma_frecuencias_observadas": float(frec_obs.sum()),
        "figura": "parte3_frecuencias_vs_teoricas.png",
        "verificacion": {
            "frecuencias_suman_1": bool(np.isclose(frec_obs.sum(), 1.0)),
            "kl_pequena_cercana_a_cero": bool(np.isfinite(kl_bits) and kl_bits < 0.01),
            "tokens_sin_soporte_sin_muestras": bool(conteos[p_teo == 0].sum() == 0),
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T3 - Comprobacion empirica ===")
    print(f"Muestras: {n_muestras} con numpy.random.default_rng(0)")
    print(f"Fuente distribucion teorica: {fuente}")
    print(f"{'token':<6}{'p_teo':>12}{'conteo':>10}{'f_obs':>12}{'diff':>12}")
    for i, tok in enumerate(tokens):
        print(f"{tok:<6}{p_teo[i]:>12.6f}{conteos[i]:>10d}"
              f"{frec_obs[i]:>12.6f}{frec_obs[i] - p_teo[i]:>12.6f}")
    print(f"Divergencia KL (emp || teo) = {kl_bits:.6f} bits ({kl_nats:.6f} nats)")
    print("Figura guardada: parte3_frecuencias_vs_teoricas.png")
    print("Resultados escritos en resultados.json")


if __name__ == "__main__":
    main()
