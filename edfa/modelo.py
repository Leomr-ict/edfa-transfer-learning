"""Red neuronal que predice la ganancia por canal de un EDFA."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

RAIZ = Path(__file__).resolve().parents[1]
RUTA_BASE = RAIZ / "modelos" / "base_cosmos.pth"


class RedEDFA(nn.Module):
    """Perceptrón multicapa de Wang et al.: 193 entradas -> 95 ganancias (dB)."""

    def __init__(self, entradas: int = 193, salidas: int = 95):
        super().__init__()
        # El nombre `net` debe mantenerse para poder cargar los pesos guardados.
        self.net = nn.Sequential(
            nn.Linear(entradas, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, salidas),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cargar(ruta: Path = RUTA_BASE) -> RedEDFA:
    """Carga un modelo guardado y lo deja en modo evaluación."""
    modelo = RedEDFA()
    modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
    return modelo.eval()


@torch.no_grad()
def predecir(modelo: RedEDFA, X) -> np.ndarray:
    """Predice la ganancia (dB) para una matriz de entradas (numpy o tensor)."""
    modelo.eval()
    return modelo(torch.as_tensor(X, dtype=torch.float32)).numpy()
