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
import src.Scratch.temp as temp
import inspect

if __name__ == '__main__':

    fig, ax = plt.subplots(1)

    # PF.add_figtext(fig, 'How is this working? whatabout with a really long text that goes way too far for the figure how does it work then??? ')
    # PF.add_fig_info(fig)
    stack = temp.call_add_fig_info(fig)
    for f in stack:
        print(f'Index = {f.index}, Filename = {f.filename[-40:]}, Function = {f.function}, code_context = {f.code_context}, frame = {f.frame}, lineno = {f.lineno}')
