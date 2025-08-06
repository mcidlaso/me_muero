import pandas as pd
import numpy as np
from invisible_cities.core.fit_functions import fit, polynom, gauss, expo
from scipy import stats

def f_diffusion_band(df: pd.DataFrame,
      bins: np.ndarray,
      nsigma: float,
      ) -> pd.DataFrame:

    Zrms2 = df.Zrms**2
    meansy , bin_edges, _ = stats.binned_statistic(df.DT, Zrms2, bins = bins, statistic='mean')
    stds, _, _ = stats.binned_statistic(df.DT, Zrms2, bins = bins, statistic='std')
    values_up = meansy + nsigma*stds
    values_down = meansy - nsigma*stds
    bin_centers = (bin_edges[:-1]+bin_edges[1:])*0.5
    f_up = fit(polynom, bin_centers, values_up, seed = (0, 1))
    f_down = fit(polynom, bin_centers, values_down, seed = (0, 1))


    mask = (f_up.fn(df.DT) > Zrms2) & (f_down.fn(df.DT) < Zrms2)

    return df[mask]

def f_S2e(df: pd.DataFrame,
          bins:np.ndarray,
          nsigma: float,
          )->pd.DataFrame:

    meansy, bin_edges, _ = stats.binned_statistic(df.DT, df.S2e, bins = bins, statistic = 'mean')
    stds, _, _ =stats.binned_statistic(df.DT, df.S2e, bins = bins, statistic = 'std')
    values_up = meansy + nsigma*stds
    values_down = meansy - nsigma*stds
    bin_centers = (bin_edges[:-1] + bin_edges[1:])*0.5
    f_up = fit(expo, bin_centers, values_up, seed = (meansy.mean(), -10000))
    f_down= fit(expo, bin_centers, values_down, seed = (meansy.mean(), -10000))

    mask = (f_up.fn(df.DT) > df.S2e) & (f_down.fn(df.DT) < df.S2e)

    return df[mask]


def test_maria_diffusion():

    DT = np.arange(0, 2000, 0.1)
    Zrms = np.arange(0, 20, 0.001)

    d = {'Zrms': Zrms, 'DT': DT}

    df_test = pd.DataFrame(data=d)
    fun_test = f_diffusion_band(df_test, 10, 1)

    assert 0 < len(fun_test) <= len(df_test)


def test_maria_diffusion2():

    DT = np.arange(0, 2000, 0.1)
    Zrms = np.arange(0, 20, 0.001)

    d = {'Zrms': Zrms, 'DT': DT}

    df_test = pd.DataFrame(data=d)
    fun_test = f_diffusion_band(df_test, 10, 10000)

    assert np.all(fun_test.values == df_test.values)


def test_maria_diffusion3():

    DT = np.arange(0, 2000, 0.1)
    Zrms = np.arange(0, 20, 0.001)

    d = {'Zrms': Zrms, 'DT': DT}

    df_test = pd.DataFrame(data=d)
    df_test = df_test.iloc[:0]

    try:
        fun_test = f_diffusion_band(df_test, 10, 100)

    except Exception as e:

        print(f"Test diffusion3 passed, an exception ocurred: {e}")


def test_f_diffusion_band():

    DT          = np.linspace(0, 1400, 1000)
    Zrms        = np.linspace(0, 20, 1000) + [1, -1]*(len(DT)//2)

    test_matrix = np.column_stack((DT, Zrms))

    df          = pd.DataFrame(test_matrix, columns = ['DT', 'Zrms'])

    assert f_diffusion_band(df, 10, 10000).shape == df.shape

    assert f_diffusion_band(df, 10, 0).shape == (0, 2)

def test_fS2e():

    DT          = np.linspace(0, 1400, 1000)
    S2e         = 8200 * np.exp(-DT/30000) + [1, -1]*(len(DT)//2)

    test_matrix = np.column_stack((DT, S2e))

    df          = pd.DataFrame(test_matrix, columns = ['DT', 'S2e'])

    assert f_S2e(df, 10, 100).shape == df.shape

    assert f_S2e(df, 10, 0).shape == (0, 2)

test_maria_diffusion(), test_maria_diffusion2(), test_maria_diffusion3(), test_f_diffusion_band(), test_fS2e()
