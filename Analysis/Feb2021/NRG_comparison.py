from __future__ import annotations

import scipy
import scipy.io
from dataclasses import dataclass
import numpy as np
import plotly.io as pio
from functools import lru_cache

pio.renderers.default = "browser"

# def NRG_fitter() -> Callable:
#     NRG = scipy.io.loadmat('NRGResults.mat')
#     occ = NRG["Occupation_mat"]
#     ens = np.reshape(NRG["Ens"], 401)
#     ts = np.reshape(NRG["Ts"], 70)
#     ens = np.flip(ens)
#     occ = np.flip(occ, 0)
#     interp = RectBivariateSpline(ens, np.log10(ts), occ, kx=1, ky=1)
#
#     def interpNRG(x, logt, dx=1, amp=1, center=0, lin=0, const=0):
#         ens = np.multiply(np.add(x, center), dx)
#         curr = [interp(en, logt)[0][0] for en in ens]
#         scaled_current = np.multiply(curr, amp)
#         scaled_current += const + np.multiply(lin, x)
#         return scaled_current
#
#     return interpNRG

@dataclass
class NRGData:
    ens: np.ndarray
    ts: np.ndarray
    conductance: np.ndarray
    dndt: np.ndarray
    entropy: np.ndarray
    occupation: np.ndarray
    int_dndt: np.ndarray

    @classmethod
    @lru_cache
    def from_mat(cls, path=r'D:\OneDrive\GitHub\dat_analysis\dat_analysis\resources\NRGResults.mat') -> NRGData:
        import os
        print(os.path.abspath('.'))
        data = scipy.io.loadmat(path)
        return cls(
            ens=data['Ens'].flatten(),
            ts=data['Ts'].flatten(),
            conductance=data['Conductance_mat'].T,
            dndt=data['DNDT_mat'].T,
            entropy=data['Entropy_mat'].T,
            occupation=data['Occupation_mat'].T,
            int_dndt=data['intDNDT_mat'].T,
        )




if __name__ == '__main__':
    pass





