"""Carga de las mediciones de laboratorio y armado de las entradas del modelo.

Cada fila de `datos/laboratorio/dataset_SX.csv` es una medición de un EDFA:
espectro de entrada (95 canales), canales encendidos (máscara del WSS),
ganancia medida por canal y potencias totales.

Las filas siguen un orden fijo, que usamos para separar entrenamiento y prueba:
- 270 filas = 2 cargas (95 y 48 canales) x 5 formas de espectro x 9 potencias x 3 ganancias.
- Un "triplete" (3 filas seguidas) tiene la misma entrada con 3 ajustes de ganancia.
- Una "forma" (27 filas seguidas) es un mismo perfil de espectro a 9 potencias.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_LAB = RAIZ / "datos" / "laboratorio"

AMPLIFICADORES = ["S1", "S2", "S3", "S4"]
N_CANALES = 95
FILAS_POR_FORMA = 27

# El rango medido va de -27 a -10 dBm; S3 tiene una fila con +8 dBm (error de medición).
PIN_MAXIMA_DBM = -7.0

ESCALARES = ["target_gain", "EDFA_input_power_total", "EDFA_output_power_total"]


def columnas(df: pd.DataFrame, prefijo: str) -> list[str]:
    """Columnas por canal (sufijo 00..94) en orden de longitud de onda."""
    return sorted(c for c in df.columns if c.startswith(prefijo))[:N_CANALES]


def cargar_amplificador(amp: str) -> pd.DataFrame:
    """Lee el dataset de un EDFA, agrega los grupos y quita mediciones fuera de rango."""
    df = pd.read_csv(CARPETA_LAB / f"dataset_{amp}.csv").copy()
    df["forma"] = np.arange(len(df)) // FILAS_POR_FORMA
    df = df[df["EDFA_input_power_total"] <= PIN_MAXIMA_DBM]
    return df.reset_index(drop=True)


def matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve X (193 entradas), y (95 ganancias) y la máscara de canales encendidos.

    X = espectro de entrada (95) + máscara (95) + [ganancia objetivo, Pin total, Pout total],
    el mismo formato del dataset COSMOS con el que se preentrenó el modelo base.
    """
    espectro = df[columnas(df, "EDFA_input_spectra_")].to_numpy(np.float32)
    mascara = df[columnas(df, "DUT_WSS_activated_channel_index_")].to_numpy(np.float32)
    escalares = df[ESCALARES].to_numpy(np.float32)
    ganancia = df[columnas(df, "calculated_gain_spectra_")].to_numpy(np.float32)
    return np.hstack([espectro, mascara, escalares]), ganancia, mascara


def entradas_enlace(espectro_dbm: np.ndarray, ganancia_objetivo: float) -> np.ndarray:
    """Arma la entrada del modelo para un tramo del enlace (95 canales encendidos)."""
    pin_total = 10 * np.log10(np.sum(10 ** (espectro_dbm / 10)))
    escalares = [ganancia_objetivo, pin_total, pin_total + ganancia_objetivo]
    return np.concatenate([espectro_dbm, np.ones(N_CANALES), escalares]).astype(np.float32)


def cargar_enlace(ajuste_s3: bool = False) -> pd.DataFrame:
    """Mediciones del enlace completo: 4 tramos x 9 potencias x 95 canales.

    En las capturas del OSA, la entrada de S3 aparece ~3 dB más alta y su ganancia
    ~2,5 dB más baja que en los otros tramos, sin una causa identificada. Por defecto
    se usan los datos tal como se midieron. Con `ajuste_s3=True` se reproduce el ajuste
    usado en la tesis: la ganancia de S3 se desplaza para igualar el promedio de S2.
    """
    enlace = pd.read_csv(CARPETA_LAB / "enlace_4_tramos.csv")
    if ajuste_s3:
        for _, grupo in enlace.groupby("Folder"):
            s2 = grupo.index[grupo["Span"] == "S2"]
            s3 = grupo.index[grupo["Span"] == "S3"]
            enlace.loc[s3, "Gain_GT"] += enlace.loc[s2, "Gain_GT"].mean() - enlace.loc[s3, "Gain_GT"].mean()
    return enlace
