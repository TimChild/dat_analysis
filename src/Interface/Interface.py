import logging
import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import abc

import src.Plotting.Mpl.PlotUtil
import src.Interface.FigAxesSubclasses as FAS
import matplotlib.colors

logger = logging.getLogger(__name__)


class Root(tk.Tk):
    def __init__(self, title='Put Title Here', min_size=(0, 0), icon=None):
        super().__init__()
        self.title(title)
        self.minsize(*min_size)
        if icon is not None:
            self.wm_iconbitmap(icon)
        self.configure(background='#AAAAAA')

    def callback(self):
        self.quit()


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
        columns = 2
        [self.master.rowconfigure(i, pad=10, weight=0) for i in range(rows)]
        [self.master.columnconfigure(i, pad=10, weight=0) for i in range(columns)]

        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=1)

        self.fig_frame = tk.Frame(self.master)
        self.fig_frame.grid(row=1, column=0, sticky=(tk.N, tk.E, tk.S, tk.W))

        self.fig_tool_frame = tk.Frame(self.master)
        self.fig_tool_frame.grid(row=0, column=0, sticky=(tk.E, tk.W), padx=10)

        self.toolbar_frame = tk.Frame(self.master)
        self.toolbar_frame.grid(row=0, rowspan=2, column=1, stick=(tk.NE, tk.SW), padx=5, pady=10)

        set_temp_background_minsize([self.fig_frame, self.fig_tool_frame, self.toolbar_frame],
                                    ['fig', 'ftool', 'mytool'])

        self.fig = None
        self.axes = None

    def set_fig(self, fig: plt.Figure):
        assert isinstance(fig, FAS.MyFigure)
        self.fig = fig
        self.axes = fig.axes
        fig_canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        fig_canvas.draw()

        toolbar = NavigationToolbar2Tk(fig_canvas, self.fig_tool_frame)  # I think it is packed as part of __init__
        # toolbar.update()  # I'm pretty sure this is called in the Nav... __init__

        fig_canvas.get_tk_widget().pack()  # I think this is necessary to make the figure show up.

    # @property
    # def title(self):
    #     return self.ax.title
    #
    # @title.setter
    # def title(self, value):
    #     self.ax.set_title(value)


def get_fig_window(num_axes):
    fig, axs = FAS.make_fig_axes(num_axes)
    window = FigGridTest()
    window.set_fig(fig)
    return window


def set_temp_background_minsize(frames, names=None):
    frame: tk.Frame
    if names is None:
        names = list(range(len(frames)))
    colors = src.Plotting.Mpl.PlotUtil.get_colors(len(frames))
    for name, frame, color in zip(names, frames, colors):
        color = matplotlib.colors.to_hex(color)
        text = tk.Text(master=frame, width=len(name), height=1)
        text.insert(tk.INSERT, f'{name}')
        text.pack()
        frame.configure(bg=color, height=50, width=50)


if __name__ == '__main__':
    a = get_fig_window(3)
    # tk.mainloop()
