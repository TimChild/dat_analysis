import logging
import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
import abc
import datetime
from typing import List


from src import CoreUtil as CU

logger = logging.getLogger(__name__)


class Root(tk.Tk):
    def __init__(self, title='Put Title Here', min_size=(600, 400), icon=None):
        super().__init__()
        self.title(title)
        self.minsize(*min_size)
        if icon is not None:
            self.wm_iconbitmap(icon)
        self.configure(background='#511F11')


class BaseWindow(abc.ABC):
    """For shared function/setup for any Stock window"""

    def __init__(self, master: tk.Tk = None, **kwargs):
        if master is None:
            master = Root(**kwargs)
        self.master = master


class FigGridTest(BaseWindow):

    def __init__(self):
        super().__init__(master=None)

        rows = 2
        columns = 1
        [self.master.columnconfigure(i, pad=3) for i in range(rows)]
        [self.master.columnconfigure(i, pad=3) for i in range(columns)]



