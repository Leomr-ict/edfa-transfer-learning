"""Validación cruzada: ¿qué tan bien predice cada modelo mediciones que no vio?

Compara tres modelos en cada EDFA:
- base:        modelo preentrenado con COSMOS, sin ajustar a nuestros equipos.
- desde_cero:  entrenado solo con las mediciones del laboratorio.
- transfer:    modelo base especializado con transfer learning.

Esquemas de partición:
- por_forma: cada pliegue deja fuera 2 formas de espectro completas (54 mediciones).
  La prueba no comparte mediciones casi idénticas con el entrenamiento.
- al_azar:   90 % / 10 % al azar, como en el notebook original. Sirve de referencia:
  entrena y prueba con repeticiones de la misma entrada, así que el error sale optimista.

Uso:  python -m edfa.validacion            (3 semillas, ~10 min en CPU)
      python -m edfa.validacion --rapido   (1 semilla)
"""

import argparse

import numpy as np
import pandas as pd
import torch

from . import datos, entrenamiento
from .modelo import RUTA_BASE, RedEDFA, predecir

CARPETA = datos.RAIZ / "resultados"
PLIEGUES = 5


def pliegues_por_forma(df: pd.DataFrame):
    """Pliegue k: prueba con las formas k (95 canales) y k+5 (48 canales)."""
    for k in range(PLIEGUES):
        prueba = (df["forma"] % PLIEGUES == k).to_numpy()
        yield np.where(~prueba)[0], np.where(prueba)[0]


def pliegues_al_azar(n: int, semilla: int = 0):
    rng = np.random.default_rng(semilla)
    for _ in range(PLIEGUES):
        orden = rng.permutation(n)
        corte = int(0.9 * n)
        yield orden[:corte], orden[corte:]


def evaluar(semillas: list[int]) -> pd.DataFrame:
    pesos_base = torch.load(RUTA_BASE, map_location="cpu", weights_only=True)
    base = RedEDFA()
    base.load_state_dict(pesos_base)
    filas = []

    for amp in datos.AMPLIFICADORES:
        df = datos.cargar_amplificador(amp)
        X, y, m = datos.matrices(df)
        esquemas = {"por_forma": list(pliegues_por_forma(df)), "al_azar": list(pliegues_al_azar(len(df)))}

        for esquema, pliegues in esquemas.items():
            for k, (ent, pru) in enumerate(pliegues):
                registro = dict(amplificador=amp, esquema=esquema, pliegue=k)
                mae_base = entrenamiento.mae_enmascarado(predecir(base, X[pru]), y[pru], m[pru])
                filas.append(registro | dict(modelo="base", semilla=-1, mae_db=mae_base))

                for semilla in semillas:
                    for nombre, entrenar in [
                        ("desde_cero", lambda: entrenamiento.desde_cero(X[ent], y[ent], m[ent], semilla)),
                        ("transfer", lambda: entrenamiento.transfer_learning(pesos_base, X[ent], y[ent], m[ent], semilla)),
                    ]:
                        modelo = entrenar()
                        mae = entrenamiento.mae_enmascarado(predecir(modelo, X[pru]), y[pru], m[pru])
                        filas.append(registro | dict(modelo=nombre, semilla=semilla, mae_db=mae))
                print(f"{amp} {esquema} pliegue {k}: listo")

    return pd.DataFrame(filas)


def resumir(resultados: pd.DataFrame) -> pd.DataFrame:
    """Error medio y desviación estándar (entre pliegues y semillas) por EDFA y modelo."""
    return (resultados.groupby(["esquema", "amplificador", "modelo"])["mae_db"]
            .agg(mae_db="mean", desv_db="std").round(3).reset_index())


def graficar(resumen: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    tabla = resumen[resumen["esquema"] == "por_forma"]
    modelos = [("base", "Base COSMOS (sin ajuste)", "#94a3b8"),
               ("desde_cero", "Desde cero (solo laboratorio)", "#f59e0b"),
               ("transfer", "Transfer learning", "#1f6a80")]
    x = np.arange(len(datos.AMPLIFICADORES))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for i, (clave, etiqueta, color) in enumerate(modelos):
        fila = tabla[tabla["modelo"] == clave].set_index("amplificador").loc[datos.AMPLIFICADORES]
        ax.bar(x + (i - 1) * 0.27, fila["mae_db"], 0.27, yerr=fila["desv_db"].fillna(0),
               capsize=3, color=color, label=etiqueta)
    ax.set_xticks(x, [f"EDFA {a}" for a in datos.AMPLIFICADORES])
    ax.set_ylabel("Error absoluto medio de ganancia [dB]")
    ax.set_title("Validación cruzada por forma de espectro (mediciones no vistas)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CARPETA / "validacion_cruzada.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rapido", action="store_true", help="usar 1 semilla en vez de 3")
    args = parser.parse_args()

    CARPETA.mkdir(exist_ok=True)
    resultados = evaluar([0] if args.rapido else [0, 1, 2])
    resultados.to_csv(CARPETA / "validacion_cruzada_detalle.csv", index=False)
    resumen = resumir(resultados)
    resumen.to_csv(CARPETA / "validacion_cruzada.csv", index=False)
    graficar(resumen)
    print(resumen.pivot_table(index=["esquema", "amplificador"], columns="modelo", values="mae_db"))


if __name__ == "__main__":
    main()
