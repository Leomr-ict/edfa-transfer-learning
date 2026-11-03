"""Preentrenamiento del modelo base con el dataset público COSMOS (preamplificadores).

El CSV (126.320 mediciones, ~290 MB) no se incluye en el repositorio por su tamaño.
Debe tener las columnas EDFA_input_spectra_XX, DUT_WSS_activated_channel_index_XX,
calculated_gain_spectra_XX (XX = 00..94), target_gain, EDFA_input_power_total y
EDFA_output_power_total. Los pesos resultantes ya están en modelos/base_cosmos.pth.

Uso:  python -m edfa.base --csv ruta/wang_preamps_compat.csv   (300 épocas; ~horas en CPU)
"""

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from . import datos
from .entrenamiento import fijar_semilla, mse_enmascarado
from .modelo import RUTA_BASE, RedEDFA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, help="CSV con las mediciones COSMOS")
    parser.add_argument("--epocas", type=int, default=300)
    parser.add_argument("--filas", type=int, default=None, help="leer solo N filas (para probar)")
    parser.add_argument("--salida", default=str(RUTA_BASE))
    args = parser.parse_args()

    fijar_semilla(0)
    X, y, m = datos.matrices(pd.read_csv(args.csv, nrows=args.filas))
    conjunto = TensorDataset(*(torch.tensor(a) for a in (X, y, m)))
    n_val = len(conjunto) // 10
    entrenamiento, validacion = random_split(conjunto, [len(conjunto) - n_val, n_val],
                                             generator=torch.Generator().manual_seed(0))
    cargador = DataLoader(entrenamiento, batch_size=256, shuffle=True)
    cargador_val = DataLoader(validacion, batch_size=1024)

    modelo = RedEDFA()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    programador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, factor=0.5, patience=5)
    mejor = float("inf")

    for epoca in range(1, args.epocas + 1):
        modelo.train()
        for bx, by, bm in cargador:
            optimizador.zero_grad()
            mse_enmascarado(modelo(bx), by, bm).backward()
            optimizador.step()

        modelo.eval()
        with torch.no_grad():
            errores = [((modelo(bx) - by).abs() * bm).sum() / bm.sum() for bx, by, bm in cargador_val]
        mae = float(torch.stack(errores).mean())
        programador.step(mae)
        if mae < mejor:
            mejor = mae
            torch.save(modelo.state_dict(), args.salida)
        print(f"época {epoca}: MAE validación {mae:.4f} dB (mejor {mejor:.4f})")


if __name__ == "__main__":
    main()
