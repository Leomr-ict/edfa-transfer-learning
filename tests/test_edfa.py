import numpy as np
import pytest

from edfa import datos, osa
from edfa.entrenamiento import mae_enmascarado
from edfa.modelo import cargar, predecir
from edfa.validacion import pliegues_por_forma

CARPETA_OSA = datos.RAIZ / "datos" / "osa_ejemplo"


@pytest.mark.parametrize("amp", datos.AMPLIFICADORES)
def test_matrices_tienen_formato_cosmos(amp):
    X, y, m = datos.matrices(datos.cargar_amplificador(amp))
    assert X.shape[1] == 193 and y.shape[1] == 95 and m.shape == y.shape
    assert set(np.unique(m)) <= {0.0, 1.0}


def test_se_descarta_medicion_fuera_de_rango():
    df = datos.cargar_amplificador("S3")
    assert len(df) == 269
    assert df["EDFA_input_power_total"].max() <= datos.PIN_MAXIMA_DBM


@pytest.mark.parametrize("amp", datos.AMPLIFICADORES)
def test_pliegues_por_forma_no_comparten_formas(amp):
    df = datos.cargar_amplificador(amp)
    usadas = []
    for entrenamiento, prueba in pliegues_por_forma(df):
        assert not set(df["forma"].iloc[entrenamiento]) & set(df["forma"].iloc[prueba])
        usadas += list(prueba)
    assert sorted(usadas) == list(range(len(df)))  # cada medición se prueba una vez


def test_ajuste_s3_solo_cambia_la_ganancia_de_s3():
    medido, ajustado = datos.cargar_enlace(), datos.cargar_enlace(ajuste_s3=True)
    otros = medido["Span"] != "S3"
    assert np.allclose(medido.loc[otros, "Gain_GT"], ajustado.loc[otros, "Gain_GT"])
    assert np.allclose(medido["Pin_GT"], ajustado["Pin_GT"])
    nominal = ajustado[ajustado["Folder"] == "PIN_17_NOMINAL"]
    medias = nominal.groupby("Span")["Gain_GT"].mean()
    assert medias["S3"] == pytest.approx(medias["S2"])


def test_mae_ignora_canales_apagados():
    y = np.zeros((1, 4))
    pred = np.array([[1.0, 1.0, 50.0, 50.0]])
    mascara = np.array([[1, 1, 0, 0]])
    assert mae_enmascarado(pred, y, mascara) == pytest.approx(1.0)


def test_modelo_base_reproduce_error_del_notebook():
    # El notebook original reportó 0,403 dB para el modelo base en S1.
    X, y, m = datos.matrices(datos.cargar_amplificador("S1"))
    assert mae_enmascarado(predecir(cargar(), X), y, m) == pytest.approx(0.40, abs=0.01)


def test_osa_detecta_95_canales_en_grilla_de_50_ghz():
    entrada = osa.leer_traza(CARPETA_OSA / "s1_entrada_95ch.csv")
    salida = osa.leer_traza(CARPETA_OSA / "s1_salida_95ch.csv")
    canales = osa.detectar_canales(entrada, n_canales=95)
    assert len(canales) == 95
    assert np.diff(canales).mean() == pytest.approx(0.4, abs=0.01)
    ganancia = osa.potencia_en_canales(salida, canales) - osa.potencia_en_canales(entrada, canales)
    assert 19 < ganancia.mean() < 22
