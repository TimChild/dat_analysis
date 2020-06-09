from matplotlib.projections import register_projection
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.PlottingFunctions as PF


class MyFigure(plt.Figure):
    pass


class MyAxes(plt.Axes):
    """For storing additional info with axes and overriding methods"""
    name = 'MyAxes'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = None
        self.y = None

        self.xerr = None
        self.yerr = None

    def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        super().plot(self, *args, scalex=True, scaley=True, data=None, **kwargs)
        # TODO: args 0, 1, are probably x, y.. but they could be in kwargs x, y
        self.x = args[0]
        self.y = args[1]

    def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None,
                vmin=None, vmax=None, alpha=None, linewidths=None,
                verts=None, edgecolors=None, *args, plotnonfinite=False,
                **kwargs):
        super().scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None,
                vmin=None, vmax=None, alpha=None, linewidths=None,
                verts=None, edgecolors=None, *args, plotnonfinite=False,
                **kwargs)
        self.x = x
        self.y = y

    def errorbar(self, x, y, yerr=None, xerr=None,
                 fmt='', ecolor=None, elinewidth=None, capsize=None,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False, errorevery=1, capthick=None,
                 **kwargs):
        super().errorbar(x, y, yerr=None, xerr=None,
                 fmt='', ecolor=None, elinewidth=None, capsize=None,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False, errorevery=1, capthick=None,
                 **kwargs)
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr = yerr


register_projection(MyAxes)  # Registers MyAxes so that it can be called in plt.subplots etc


def make_fig_axes(num):
    fig = plt.figure(FigureClass=MyFigure)
    gs = PF.get_gridspec(fig, num, return_list=True)  # Could get actual gs here instead if want different axes
    axs = []
    for g in gs:
        axs.append(fig.add_subplot(g, projection='MyAxes'))
    return fig, axs


if __name__ == '__main__':
    # fig, ax = plt.subplots(FigureClass = MyFigure, subplot_kw=dict(projection='MyAxes'))  # How to invoke MyFigure and MyAxes
    fig, axs = make_fig_axes(5)
