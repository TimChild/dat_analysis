import pickle
import inspect
from unittest.mock import patch
from src import temp
import pandas as pd
import numpy as np
import src.DFcode.DFutil as DU



import matplotlib as mpl
import matplotlib.pyplot as plt
import src.PlottingFunctions as PF


def call_add_fig_info(fig):
    stack = PF.add_standard_fig_info(fig)
    return stack


if __name__ == '__main__':

    fig, ax = plt.subplots(1)


    PF.add_standard_fig_info(fig)
