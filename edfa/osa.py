"""Lectura de trazas del analizador de espectro óptico (OSA) y extracción por canal.

Es el paso que convierte las capturas del OSA (archivos CSV con encabezado del equipo)
en las potencias y ganancias por canal que usan los datasets del laboratorio.

Ejemplo:
    entrada = leer_traza("datos/osa_ejemplo/s1_entrada_95ch.csv")
    salida = leer_traza("datos/osa_ejemplo/s1_salida_95ch.csv")
    canales = detectar_canales(entrada, n_canales=95)
    ganancia = potencia_en_canales(salida, canales) - potencia_en_canales(entrada, canales)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

SEPARACION_NM = 0.4  # grilla de 50 GHz


def leer_traza(ruta: str | Path) -> pd.DataFrame:
    """Lee un CSV del OSA y devuelve longitud de onda (nm) y potencia (dBm).

    El equipo escribe ~70 líneas de configuración antes de los datos; se salta
    todo hasta la primera línea que empieza con un número. Si la potencia viene
    en escala lineal (mW), se convierte a dBm.
    """
    with open(ruta, encoding="utf-8", errors="ignore") as f:
        lineas = f.readlines()
    inicio = next(i for i, linea in enumerate(lineas) if linea[:1].isdigit())
    traza = pd.read_csv(ruta, skiprows=inicio, header=None, usecols=[0, 1], names=["lambda_nm", "potencia"])
    traza = traza.apply(pd.to_numeric, errors="coerce").dropna().sort_values("lambda_nm")
    potencia = traza["potencia"].to_numpy()
    if potencia.min() > 0:  # escala lineal
        potencia = 10 * np.log10(potencia)
    return pd.DataFrame({"lambda_nm": traza["lambda_nm"].to_numpy(), "potencia_dbm": potencia})


def detectar_canales(traza: pd.DataFrame, n_canales: int) -> np.ndarray:
    """Longitudes de onda centrales de los `n_canales` picos más potentes.

    Se usa sobre la traza de entrada, donde los canales WDM sobresalen claramente
    del ruido; esa grilla se reutiliza para la salida del mismo amplificador.
    """
    paso = np.median(np.diff(traza["lambda_nm"]))
    distancia = max(1, int(0.75 * SEPARACION_NM / paso))
    picos, _ = find_peaks(traza["potencia_dbm"], distance=distancia)
    mas_potentes = picos[np.argsort(traza["potencia_dbm"].to_numpy()[picos])[-n_canales:]]
    return np.sort(traza["lambda_nm"].to_numpy()[mas_potentes])


def potencia_en_canales(traza: pd.DataFrame, centros_nm: np.ndarray, ventana_nm: float = 0.15) -> np.ndarray:
    """Potencia máxima (dBm) dentro de ±ventana alrededor de cada canal."""
    lam, pot = traza["lambda_nm"].to_numpy(), traza["potencia_dbm"].to_numpy()
    return np.array([pot[np.abs(lam - c) <= ventana_nm].max() for c in centros_nm])

