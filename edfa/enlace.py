"""Evaluación en el enlace de 4 tramos (200 km), con y sin incertidumbre de potencia.

Diseño experimental (ver docs/montaje_experimental.png):
- 3 regímenes nominales de potencia de entrada: -24, -17 y -9 dBm.
- Incertidumbre: una caída de -3 dB antes del primer EDFA (-27, -20 y -12 dBm),
  sin tocar el resto del enlace, así que la perturbación se propaga.

Dos modos de evaluación:
- medida:    cada modelo recibe el espectro medido a la entrada de su EDFA. Aísla el
             error del modelo frente a espectros que ya traen ASE e inclinación acumulados.
- propagada: gemelo digital; cada tramo recibe el espectro que predijo el anterior,
             así que además se acumulan los errores de predicción.

Estas mediciones del enlace nunca se usan para entrenar.

Uso:  python -m edfa.enlace                 (3 semillas; entrena los especialistas si no existen)
      python -m edfa.enlace --rapido        (1 semilla)
      python -m edfa.enlace --comando-20db  (el modelo recibe la ganancia comandada, no la medida)
"""

import argparse

import numpy as np
import pandas as pd
import torch

from . import datos, entrenamiento
from .modelo import RUTA_BASE, RedEDFA, cargar, predecir

CARPETA = datos.RAIZ / "resultados"
CARPETA_ESPECIALISTAS = datos.RAIZ / "modelos" / "especialistas"

# régimen -> (carpeta nominal, carpeta con caída de -3 dB)
REGIMENES = {
    "-24 dBm": ("PIN_24_NOMINAL", "PIN_27"),
    "-17 dBm": ("PIN_17_NOMINAL", "PIN_20"),
    "-9 dBm": ("PIN_9_NOMINAL", "PIN_12"),
}
GANANCIA_COMANDADA_DB = 20.0
ESTILOS = [("base", "Base COSMOS (sin ajuste)", "#94a3b8"),
           ("desde_cero", "Desde cero (solo laboratorio)", "#f59e0b"),
           ("transfer", "Transfer learning", "#1f6a80")]


def especialistas(semilla: int, reentrenar: bool) -> dict[str, dict[str, RedEDFA]]:
    """Modelos por EDFA entrenados con todas sus mediciones.

    Los de la semilla 0 se guardan en modelos/especialistas/ (son los modelos finales);
    las otras semillas solo se usan para medir cuánto varía el resultado.
    """
    pesos_base = torch.load(RUTA_BASE, map_location="cpu", weights_only=True)
    CARPETA_ESPECIALISTAS.mkdir(parents=True, exist_ok=True)
    modelos = {"transfer": {}, "desde_cero": {}}
    for amp in datos.AMPLIFICADORES:
        X, y, m = datos.matrices(datos.cargar_amplificador(amp))
        for tipo, entrenar in [("transfer", lambda: entrenamiento.transfer_learning(pesos_base, X, y, m, semilla)),
                               ("desde_cero", lambda: entrenamiento.desde_cero(X, y, m, semilla))]:
            ruta = CARPETA_ESPECIALISTAS / f"{tipo}_{amp}.pth"
            if semilla != 0:
                modelos[tipo][amp] = entrenar()
                continue
            if reentrenar or not ruta.exists():
                torch.save(entrenar().state_dict(), ruta)
            modelos[tipo][amp] = cargar(ruta)
    return modelos


def simular(medido: pd.DataFrame, modelos_por_tramo: dict[str, RedEDFA], propagar: bool,
            comando_20db: bool = False) -> list[float]:
    """Error (dB) de cada tramo del enlace.

    El modelo recibe como "ganancia objetivo" la ganancia media medida del tramo (igual que
    en el entrenamiento), o los 20 dB comandados si `comando_20db` es verdadero.
    """
    tramos = [medido[medido["Span"] == t].sort_values("Wavelength") for t in datos.AMPLIFICADORES]
    entrada = tramos[0]["Pin_GT"].to_numpy(np.float32)
    errores = []
    for i, (amp, tramo) in enumerate(zip(datos.AMPLIFICADORES, tramos)):
        ganancia_real = tramo["Gain_GT"].to_numpy(np.float32)
        if not propagar:
            entrada = tramo["Pin_GT"].to_numpy(np.float32)
        objetivo = GANANCIA_COMANDADA_DB if comando_20db else float(ganancia_real.mean())
        ganancia_pred = predecir(modelos_por_tramo[amp], datos.entradas_enlace(entrada, objetivo)[None, :])[0]
        errores.append(float(np.abs(ganancia_pred - ganancia_real).mean()))
        if propagar and i + 1 < len(tramos):
            salida_real = tramo["Pin_GT"].to_numpy() + ganancia_real
            atenuacion = salida_real - tramos[i + 1]["Pin_GT"].to_numpy()
            entrada = (entrada + ganancia_pred - atenuacion).astype(np.float32)
    return errores


def evaluar(semillas: list[int], reentrenar: bool, comando_20db: bool) -> pd.DataFrame:
    base = cargar(RUTA_BASE)
    versiones = {"medidos": datos.cargar_enlace(), "ajuste_s3_tesis": datos.cargar_enlace(ajuste_s3=True)}
    filas = []
    for semilla in semillas:
        candidatos = especialistas(semilla, reentrenar)
        if semilla == semillas[0]:
            candidatos = {"base": {amp: base for amp in datos.AMPLIFICADORES}} | candidatos
        for version, enlace in versiones.items():
            for regimen, carpetas in REGIMENES.items():
                for estado, carpeta in zip(["nominal", "perturbado"], carpetas):
                    medido = enlace[enlace["Folder"] == carpeta]
                    for modo in ["medida", "propagada"]:
                        for nombre, por_tramo in candidatos.items():
                            errores = simular(medido, por_tramo, modo == "propagada", comando_20db)
                            for amp, error in zip(datos.AMPLIFICADORES, errores):
                                filas.append(dict(datos_s3=version, semilla=semilla, regimen=regimen,
                                                  estado=estado, modo=modo, modelo=nombre, tramo=amp,
                                                  mae_db=error))
    return pd.DataFrame(filas)


def tabla_nominal(resultados: pd.DataFrame, version: str = "medidos") -> pd.DataFrame:
    """Error por tramo en condiciones nominales (promedio de regímenes y semillas)."""
    r = resultados[(resultados["datos_s3"] == version) & (resultados["estado"] == "nominal")]
    tabla = r.pivot_table(index="tramo", columns=["modo", "modelo"], values="mae_db").round(3)
    tabla.loc["promedio"] = tabla.mean().round(3)
    return tabla


def tabla_penalizacion(resultados: pd.DataFrame) -> pd.DataFrame:
    """Aumento del error por la caída de -3 dB (perturbado - nominal), entrada medida."""
    r = resultados[(resultados["datos_s3"] == "medidos") & (resultados["modo"] == "medida")
                   & (resultados["modelo"] != "desde_cero")]
    media = r.groupby(["regimen", "modelo", "tramo", "estado"])["mae_db"].mean().unstack("estado")
    return (media["perturbado"] - media["nominal"]).unstack("tramo").round(3)


def graficar(resultados: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    r = resultados[(resultados["datos_s3"] == "medidos") & (resultados["estado"] == "nominal")]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, sharey=True)
    titulos = {"medida": "(a) Entrada medida en cada tramo", "propagada": "(b) Gemelo digital (entrada propagada)"}
    for eje, (modo, titulo) in zip(ejes, titulos.items()):
        promedio = r[r["modo"] == modo].groupby(["modelo", "tramo"])["mae_db"].mean().unstack()
        for clave, etiqueta, color in ESTILOS:
            eje.plot(datos.AMPLIFICADORES, promedio.loc[clave, datos.AMPLIFICADORES], "o-",
                     color=color, lw=2, label=etiqueta)
        eje.set_title(titulo)
        eje.set_xlabel("Tramo del enlace")
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("Error absoluto medio de ganancia [dB]")
    ejes[0].legend(frameon=False)
    fig.suptitle("Enlace de 4 tramos, condiciones nominales (promedio de -24, -17 y -9 dBm)")
    fig.tight_layout()
    fig.savefig(CARPETA / "enlace.png")

    r = resultados[(resultados["datos_s3"] == "medidos") & (resultados["modo"] == "medida")]
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharey=True)
    for eje, (regimen, _) in zip(ejes, REGIMENES.items()):
        sub = r[r["regimen"] == regimen].groupby(["modelo", "estado", "tramo"])["mae_db"].mean()
        for clave, etiqueta, color in [ESTILOS[0], ESTILOS[2]]:
            nominal = sub.loc[(clave, "nominal")][datos.AMPLIFICADORES]
            perturbado = sub.loc[(clave, "perturbado")][datos.AMPLIFICADORES]
            eje.plot(datos.AMPLIFICADORES, nominal, "o--", color=color, alpha=0.7, label=f"{etiqueta}, nominal")
            eje.plot(datos.AMPLIFICADORES, perturbado, "s-", color=color, lw=2, label=f"{etiqueta}, -3 dB")
            eje.fill_between(datos.AMPLIFICADORES, nominal, perturbado, color=color, alpha=0.12)
        eje.set_title(f"Régimen {regimen} → {int(regimen.split()[0]) - 3} dBm")
        eje.set_xlabel("Tramo del enlace")
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("Error absoluto medio de ganancia [dB]")
    ejes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Efecto de una caída de -3 dB en la potencia de entrada (entrada medida en cada tramo)")
    fig.tight_layout()
    fig.savefig(CARPETA / "incertidumbre.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reentrenar", action="store_true", help="volver a entrenar los especialistas")
    parser.add_argument("--rapido", action="store_true", help="usar 1 semilla en vez de 3")
    parser.add_argument("--comando-20db", action="store_true",
                        help="dar al modelo la ganancia comandada (20 dB) en vez de la medida")
    args = parser.parse_args()

    CARPETA.mkdir(exist_ok=True)
    resultados = evaluar([0] if args.rapido else [0, 1, 2], args.reentrenar, args.comando_20db)
    sufijo = "_comando_20db" if args.comando_20db else ""
    resultados.to_csv(CARPETA / f"enlace_detalle{sufijo}.csv", index=False)

    nominal = tabla_nominal(resultados)
    print("Error por tramo, condiciones nominales (datos medidos):\n", nominal)
    print("\nPromedio con el ajuste de S3 de la tesis:\n", tabla_nominal(resultados, "ajuste_s3_tesis").loc["promedio"])
    if args.comando_20db:
        return
    penalizacion = tabla_penalizacion(resultados)
    print("\nAumento del error por la caída de -3 dB (entrada medida):\n", penalizacion)
    por_semilla = (resultados[(resultados["datos_s3"] == "medidos") & (resultados["estado"] == "nominal")]
                   .groupby(["modo", "modelo", "semilla"])["mae_db"].mean().groupby(["modo", "modelo"]).std())
    print("\nDesviación entre semillas del error promedio:\n", por_semilla.round(3))
    nominal.to_csv(CARPETA / "enlace.csv")
    penalizacion.to_csv(CARPETA / "incertidumbre.csv")
    graficar(resultados)


if __name__ == "__main__":
    main()
