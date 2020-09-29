# from src.DataStandardize.ExpSpecific.Aug20 import add_temp_magy
# from typing import List, Union, Tuple
#
# from src.Scripts.StandardImports import *
# from src.Plotting.Plotly.PlotlyUtil import PlotlyViewer as PV
# from progressbar import progressbar
# import plotly.graph_objects as go
# from src.DataStandardize.ExpSpecific.Sep20 import Fixes
# from src.Plotting.Plotly import PlotlyUtil as PlU
# from src.Scripts.SquareEntropyAnalysis import *
# import src.Scripts.SquareEntropyAnalysis as EA
# import src.DatObject.Attributes.SquareEntropy as SE

import concurrent.futures
import math


import concurrent.futures


def run_concurrent(funcs, func_args=None, func_kwargs=None, which='multiprocess', max_num=10):
    which = which.lower()
    if which not in ('multiprocess', 'multithread'):
        raise ValueError('Which must be "multiprocess" or "multithread"')

    if type(funcs) != list and type(func_args) == list:
        funcs = [funcs]*len(func_args)
    if func_args is None:
        func_args = [None]*len(funcs)
    else:
        # Make sure func_args is a list of lists, (for use with list of single args)
        for i, arg in enumerate(func_args):
            if type(arg) not in [list, tuple]:
                func_args[i] = [arg]
    if func_kwargs is None:
        func_kwargs = [{}]*len(funcs)

    num_workers = len(funcs)
    if num_workers > max_num:
        num_workers = max_num

    results = {i: None for i in range(len(funcs))}

    if which == 'multithread':
        worker_maker = concurrent.futures.ThreadPoolExecutor
    elif which == 'multiprocess':
        worker_maker = concurrent.futures.ProcessPoolExecutor
    else:
        raise ValueError

    with worker_maker(max_workers=num_workers) as executor:
        future_to_result = {executor.submit(func, *f_args, **f_kwargs): i for i, (func, f_args, f_kwargs) in enumerate(zip(funcs, func_args, func_kwargs))}
        for future in concurrent.futures.as_completed(future_to_result):
            i = future_to_result[future]
            results[i] = future.result()
    return list(results.values())







PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()

    ans = run_concurrent(is_prime, PRIMES)
