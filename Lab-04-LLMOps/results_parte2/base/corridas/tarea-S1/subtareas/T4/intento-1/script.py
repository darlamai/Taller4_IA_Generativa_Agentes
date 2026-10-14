import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# T4: Resumen por servidor a partir del DataFrame de la Tarea 3
# Columnas exactas y en orden: server, observations, mean_power_w,
# max_temperature_c, mean_load, review_count
# ----------------------------------------------------------------------------

COLS = ["server", "observations", "mean_power_w", "max_temperature_c",
        "mean_load", "review_count"]


def a_native(obj):
    """Convierte tipos de numpy/pandas a tipos nativos para JSON."""
    if isinstance(obj, dict):
        return {k: a_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_native(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def asegurar_columnas(df):
    """Garantiza la presencia de load_score y requires_review."""
    if "load_score" not in df.columns:
        for ruta in ("entrada/T3/resultados.json", "entrada/T2/resultados.json"):
            p = Path(ruta)
            if p.is_file():
                with open(p, encoding="utf-8") as fh:
                    datos = json.load(fh)
                if isinstance(datos.get("load_score"), list):
                    df["load_score"] = np.asarray(datos["load_score"], dtype=float)
                    break
    if "requires_review" not in df.columns and "load_score" in df.columns:
        df["requires_review"] = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)
    return df


def cargar_df_t3():
    """Localiza el DataFrame resultante de la T3 o lo reconstruye."""
    t3_dir = Path("entrada/T3")
    if t3_dir.is_dir():
        for patron in ("*.parquet", "*.csv"):
            archivos = sorted(t3_dir.rglob(patron))
            if archivos:
                f = archivos[0]
                if f.suffix == ".parquet":
                    df = pd.read_parquet(f)
                    fuente = f"entrada/T3/{f.name}"
                else:
                    df = pd.read_csv(f, parse_dates=["timestamp"])
                    fuente = f"entrada/T3/{f.name}"
                return asegurar_columnas(df), fuente
    # Reconstrucción: datos originales + listas guardadas por subtareas previas
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
    r3 = Path("entrada/T3/resultados.json")
    if r3.is_file():
        with open(r3, encoding="utf-8") as fh:
            t3 = json.load(fh)
        if isinstance(t3.get("load_score"), list) and isinstance(t3.get("requires_review"), list):
            df["load_score"] = np.asarray(t3["load_score"], dtype=float)
            df["requires_review"] = np.asarray(t3["requires_review"], dtype=bool)
            return df, "data/server_measurements.csv + entrada/T3/resultados.json (reconstruido)"
    with open("entrada/T2/resultados.json", encoding="utf-8") as fh:
        t2 = json.load(fh)
    df["load_score"] = np.asarray(t2["load_score"], dtype=float)
    df["requires_review"] = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)
    return df, "data/server_measurements.csv + entrada/T2/resultados.json (reconstruido)"


df, fuente = cargar_df_t3()

# Normaliza requires_review a bool puro
if df["requires_review"].dtype == object:
    df["requires_review"] = df["requires_review"].map(
        lambda v: bool(v) if isinstance(v, (bool, np.bool_))
        else str(v).strip().lower() == "true"
    )
df["requires_review"] = df["requires_review"].astype(bool)

# ----------------------------------------------------------------------------
# Tabla resumen: una fila por servidor, columnas exactas y en orden
# ----------------------------------------------------------------------------
resumen = (
    df.groupby("server", sort=True)
      .agg(
          observations=("server", "size"),
          mean_power_w=("power_w", "mean"),
          max_temperature_c=("temperature_c", "max"),
          mean_load=("load_score", "mean"),
          review_count=("requires_review", "sum"),
      )
      .reset_index()
)
resumen = resumen[COLS].copy()
resumen["observations"] = resumen["observations"].astype("int64")
resumen["review_count"] = resumen["review_count"].astype("int64")

# ----------------------------------------------------------------------------
# Archivo exigido por el enunciado
# ----------------------------------------------------------------------------
Path("output").mkdir(parents=True, exist_ok=True)
resumen.to_parquet("output/server_analysis.parquet", index=False)

# ----------------------------------------------------------------------------
# Figura resumen (PNG)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(resumen["server"], resumen["mean_power_w"], color="#4C72B0")
axes[0].set_title("Potencia media por servidor")
axes[0].set_ylabel("mean_power_w (W)")
axes[0].tick_params(axis="x", rotation=45)
axes[1].bar(resumen["server"], resumen["review_count"], color="#C44E52")
axes[1].set_title("Observaciones marcadas para revisión")
axes[1].set_ylabel("review_count")
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig("t4_resumen_servidores.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# Verificación contra el resultado esperado del enunciado (AI-SRV-01)
# ----------------------------------------------------------------------------
fila01 = resumen.loc[resumen["server"] == "AI-SRV-01"].iloc[0]
verif = {
    "seis_filas": int(len(resumen)) == 6,
    "columnas_en_orden_exacto": list(resumen.columns) == COLS,
    "observations_50": int(fila01["observations"]) == 50,
    "mean_power_w_363.4598": abs(float(fila01["mean_power_w"]) - 363.4598) <= 5e-5,
    "max_temperature_c_86.50": abs(float(fila01["max_temperature_c"]) - 86.50) <= 1e-6,
    "mean_load_-0.614527": abs(float(fila01["mean_load"]) - (-0.614527)) <= 5e-7,
    "review_count_2": int(fila01["review_count"]) == 2,
}

# Contraste con el review_count por servidor reportado por T3
coincide_t3 = None
try:
    with open("entrada/T3/resultados.json", encoding="utf-8") as fh:
        t3 = json.load(fh)
    ref = t3.get("review_count_por_servidor")
    if isinstance(ref, dict):
        calc = {r["server"]: int(r["review_count"]) for r in resumen.to_dict(orient="records")}
        coincide_t3 = all(int(ref.get(k, -1)) == v for k, v in calc.items())
except Exception:
    pass

# ----------------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
tabla_registros = a_native(resumen.to_dict(orient="records"))

tabla_redondeada = resumen.copy()
tabla_redondeada["mean_power_w"] = tabla_redondeada["mean_power_w"].round(4)
tabla_redondeada["max_temperature_c"] = tabla_redondeada["max_temperature_c"].round(2)
tabla_redondeada["mean_load"] = tabla_redondeada["mean_load"].round(6)

resultados = {
    "subtarea": "T4",
    "descripcion": ("Tabla resumen por servidor con columnas server, observations, "
                    "mean_power_w, max_temperature_c, mean_load, review_count"),
    "fuente_dataframe": fuente,
    "n_filas_entrada": int(len(df)),
    "n_servidores": int(len(resumen)),
    "columnas": COLS,
    "orden_columnas_correcto": list(resumen.columns) == COLS,
    "tabla": tabla_registros,
    "resumen_por_servidor": tabla_registros,
    "por_servidor": {r["server"]: r for r in tabla_registros},
    "AI-SRV-01": next(r for r in tabla_registros if r["server"] == "AI-SRV-01"),
    "tabla_redondeada_estilo_enunciado": a_native(tabla_redondeada.to_dict(orient="records")),
    "review_count_por_servidor": {r["server"]: int(r["review_count"]) for r in tabla_registros},
    "total_observations": int(resumen["observations"].sum()),
    "total_review_count": int(resumen["review_count"].sum()),
    "coincide_review_count_con_T3": coincide_t3,
    "verificacion_AI-SRV-01": verif,
    "exito": all(verif.values()),
    "archivo_parquet": "output/server_analysis.parquet",
    "figura_png": "t4_resumen_servidores.png",
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------------
# Resumen por consola
# ----------------------------------------------------------------------------
print("Fuente del DataFrame:", fuente)
print("\nTabla resumen por servidor (valores redondeados para lectura):")
print(tabla_redondeada.to_string(index=False))
print("\nVerificación AI-SRV-01 (enunciado):")
for k, v in verif.items():
    print(f"  {k}: {v}")
print("Éxito global:", all(verif.values()))
print("Coincide review_count con T3:", coincide_t3)
print("Parquet escrito en: output/server_analysis.parquet")
print("Figura guardada en: t4_resumen_servidores.png")
