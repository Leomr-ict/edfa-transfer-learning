"""Entrenamiento de modelos especialistas: desde cero o con transfer learning.

El error se mide solo en los canales encendidos (máscara del WSS): en los canales
apagados no hay señal y la "ganancia" del archivo es un valor de relleno.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .modelo import RedEDFA


def fijar_semilla(semilla: int) -> None:
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def mse_enmascarado(pred, y, mascara):
    return ((pred - y) ** 2 * mascara).sum() / mascara.sum()


def mae_enmascarado(pred, y, mascara) -> float:
    """Error absoluto medio (dB) en los canales encendidos."""
    pred, y, mascara = (np.asarray(a, dtype=np.float64) for a in (pred, y, mascara))
    return float((np.abs(pred - y) * mascara).sum() / mascara.sum())


def _cargador(X, y, mascara, tamano_lote: int, semilla: int) -> DataLoader:
    datos = TensorDataset(*(torch.tensor(a, dtype=torch.float32) for a in (X, y, mascara)))
    generador = torch.Generator().manual_seed(semilla)
    return DataLoader(datos, batch_size=tamano_lote, shuffle=True, generator=generador)


def _entrenar(modelo, cargador, parametros, lr: float, epocas: int) -> None:
    optimizador = torch.optim.Adam(parametros, lr=lr)
    modelo.train()
    for _ in range(epocas):
        for bx, by, bm in cargador:
            optimizador.zero_grad()
            mse_enmascarado(modelo(bx), by, bm).backward()
            optimizador.step()
    modelo.eval()


def desde_cero(X, y, mascara, semilla: int = 0, epocas: int = 300, lr: float = 1e-3,
               tamano_lote: int = 32) -> RedEDFA:
    """Entrena un especialista solo con las mediciones del laboratorio."""
    fijar_semilla(semilla)
    modelo = RedEDFA()
    _entrenar(modelo, _cargador(X, y, mascara, tamano_lote, semilla), modelo.parameters(), lr, epocas)
    return modelo


def transfer_learning(pesos_base: dict, X, y, mascara, semilla: int = 0,
                      epocas_fase1: int = 150, epocas_fase2: int = 150,
                      lr_fase1: float = 5e-3, lr_fase2: float = 1e-4,
                      tamano_lote: int = 32) -> RedEDFA:
    """Especializa el modelo base (preentrenado con COSMOS) en un EDFA, en dos fases.

    Fase 1: se congelan todas las capas menos la última, que se reinicia y se entrena.
    Fase 2: se descongela todo y se ajusta la red completa con una tasa de aprendizaje baja.
    """
    fijar_semilla(semilla)
    modelo = RedEDFA()
    modelo.load_state_dict(pesos_base)
    cargador = _cargador(X, y, mascara, tamano_lote, semilla)

    ultima = modelo.net[-1]
    for p in modelo.parameters():
        p.requires_grad = False
    nn.init.kaiming_normal_(ultima.weight, nonlinearity="relu")
    nn.init.zeros_(ultima.bias)
    for p in ultima.parameters():
        p.requires_grad = True
    _entrenar(modelo, cargador, ultima.parameters(), lr_fase1, epocas_fase1)

    for p in modelo.parameters():
        p.requires_grad = True
    _entrenar(modelo, cargador, modelo.parameters(), lr_fase2, epocas_fase2)
    return modelo
