import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from typing import Callable
from invisible_cities.core.fit_functions import fit, polynom, gauss

def diffusion_band(kdst         :   pd.DataFrame,
                   lower_limit  :   Callable,
                   upper_limit  :   Callable)->pd.DataFrame:

    mask        = ((kdst.Zrms**2 < upper_limit(kdst.DT)) &
                   (kdst.Zrms**2 > lower_limit(kdst.DT)))

    if kdst.empty:
        raise ValueError ('Empty DataFrame!')

    kdst_inband = kdst[mask]

    return kdst_inband


def diffusion_band_2(kdst   : pd.DataFrame,
                     bins   : int,
                     sigmas : float) -> pd.DataFrame:

    means, bins_edges, _ = binned_statistic(kdst.DT, kdst.Zrms**2, bins = bins, statistic = 'mean')

    std, _, _   = binned_statistic(kdst.DT, kdst.Zrms**2, bins = bins, statistic = 'std')
    top         = means + sigmas * std
    bottom      = means - sigmas * std

    bins_center = (bins_edges[:-1]+bins_edges[1:])/2
    top_fit     = fit(polynom, bin_centers, top,    seed = (0, 1))
    bottom_fit  = fit(polynom, bin_centers, bottom, seed = (0, 1))


    mask        = (f_up.fn(df.DT) > Zrms2) & (f_down.fn(df.DT) < Zrms2)

    return df[mask]


def test_diffusion_band():
    Zrms = np.arange(0,100, 0.01)**0.5
    DT = np.arange(0, 2000, 0.2)
    d = {'Zrms': Zrms, 'DT': DT}
    df_test = pd.DataFrame(data = d)
    lower_limit = lambda x: 0.05*x - 3
    upper_limit = lambda x: 0.05*x + 3
    fun_test = diffusion_band(df_test, lower_limit, upper_limit)

    assert fun_test.shape == df_test.shape

    assert np.all(fun_test.values == df_test.values)


def test_diffusion_band_2():

    Zrms = np.linspace(0, 10,   1000)**2
    DT   = np.linspace(0, 1400, 1000)
    df   = pd.DataFrame(np.column_stack((DT, Zrms)), columns = ['DT', 'Zrms'])

    lower_limit = lambda x: 0.05*x + 3
    upper_limit = lambda x: 0.05*x + 3


    assert diffusion_band(df, lower_limit, lower_limit).shape == (0,2)

    assert diffusion_band(df, upper_limit, lower_limit).shape == (0,2)


def test_diffusion_band_3():

    df   = pd.DataFrame(np.column_stack(([], [])), columns = ['DT', 'Zrms'])

    lower_limit = lambda x: 0.05*x + 3
    upper_limit = lambda x: 0.05*x + 3

    try:
        diffusion_band(df, lower_limit, upper_limit)

    except ValueError:

        print("Test 3 passed! Caught expected ValueError.")
        return

    except Exception as e:

        print(f"Test 3 failed! Unexpected exception: {e}")
        raise


test_diffusion_band()
test_diffusion_band_2()
test_diffusion_band_3()
