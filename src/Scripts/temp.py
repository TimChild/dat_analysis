from src.Scripts.StandardImports import *
import numpy as np
import matplotlib.pyplot as plt
import plotly

def func(x, x1, a, b):
    return a * x + np.sin(b * x) + x1

def print_func(x, x1, a, b, d):
    print(f'x={x}, x1={x1}, a={a}, b={b}, d={d}')
    return a * x + np.sin(b * x) + x1 + d

if __name__ == '__main__':
    a = np.array([1, 2])
    b = np.array([3, 4, 5])
    c = np.ones(5)
    d = 5

    x = np.linspace(0, 10, 11)
    x1 = np.linspace(-5, 5, 11)
    # x = x / 360 * 2 * np.pi

    ag, bg, xg = CU.add_data_dims(a, b, x)
    am, bm, xm = np.meshgrid(ag, bg, xg, indexing='ij')

    x1g = CU.match_dims(x1, xg, dim=-1)
    x1m = CU.match_dims(x1, xm, dim=-1)
    dm = np.ones(x1m.shape)*d

    # x1ng = CU.match_dims(x1, x, dim=-1)
    z1 = func(xg, x1g, ag, bg)
    z2 = func(x, x1g, ag, bg)
    z3 = func(xm, x1m, am, bm)

    fig, ax = plt.subplots(1)

    np.array(list(map(print_func, np.nditer(xm), np.nditer(x1m), np.nditer(am), np.nditer(bm), np.nditer(dm)))).reshape(xm.shape)
