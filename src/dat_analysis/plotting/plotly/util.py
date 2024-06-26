from __future__ import annotations
from typing import List, Union, Optional, TYPE_CHECKING, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import plotly.io as pio
import numbers
from itertools import product
import numpy as np
import re
import logging

from ...core_util import resample_data

if TYPE_CHECKING:
    from ...analysis_tools.data import Data


# Allows working in jupyter lab, notebook, and export to pdf
pio.renderers.default = "plotly_mimetype+notebook+pdf"

default_config = dict(
    toImageButtonOptions={
        "format": "png",  # one of png, svg, jpeg, webp
        "scale": 2,  # multiply title/legend/axis/canvas sizes by this factor
    },
)

default_layout = dict(
    template="plotly_white",
    xaxis=dict(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    ),
    yaxis=dict(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    ),
    #      modebar_add=['drawline',
    #  'drawopenpath',
    #  'drawclosedpath',
    #  'drawcircle',
    #  'drawrect',
    #  'eraseshape'
    # ]
)
_my_plotly_template = go.layout.Template(
    layout=default_layout
)  # Turn the dict options into a plotly template
pio.templates["datanalysis"] = _my_plotly_template  # Register the template
pio.templates.default = (
    "datanalysis"  # Set the newly registered template as the default from now on
)


def show_named_plotly_colours():
    """
    function to display to user the colours to match plotly's named
    css colours.

    Reference:
        #https://community.plotly.com/t/plotly-colours-list/11730/3

    Returns:
        plotly dataframe with cell colour to match named colour name

    """
    s = """
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,teal
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        """
    li = s.split(",")
    li = [l.replace("\n", "") for l in li]
    li = [l.replace(" ", "") for l in li]

    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame.from_dict({"colour": li})
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["plotly Named CSS colours"],
                    line_color="black",
                    fill_color="white",
                    align="center",
                    font=dict(color="black", size=14),
                ),
                cells=dict(
                    values=[df.colour],
                    line_color=[df.colour],
                    fill_color=[df.colour],
                    align="center",
                    font=dict(color="black", size=11),
                ),
            )
        ]
    )

    fig.show(renderer="browser")


def make_slider_figure(
    datas: List[Union[List[np.ndarray], np.ndarray]],
    xs: Union[np.ndarray, List[np.ndarray]],
    ys: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ids=None,
    titles=None,
    labels=None,
    xlabel="",
    ylabel="",
    plot_kwargs=None,
):
    """
    Get plotly figure with data in layers and a slider to change between them
    Args:
        ylabel (str): Y label
        xlabel (str): X label
        plot_kwargs (dict): Args to pass into either go.Heatmap or go.Scatter depending on 2D or 1D
        xs (Union(np.ndarray, list)): x_array for each slider position or single array to use for all
        ys (Union(np.ndarray, list)): y_array for each slider position or single array to use for all
        datas (list): list of datas for each slider position (i.e. can be list of list of arrays or just list of arrays)
        ids (list): ID for slider
        titles (list): Title at slider position
        labels (List[str]): Label for each trace per step (i.e. four lines per step, four labels)

    Returns:
        (go.Figure): plotly figure
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    assert type(datas) == list

    if ys is None:
        datas_per_step = np.atleast_2d(datas[0]).shape[0]
    else:
        datas_per_step = 1

    ids = ids if ids else range(len(datas))
    titles = titles if titles is not None else range(len(datas))
    labels = labels if labels is not None else [None] * datas_per_step

    if isinstance(xs, np.ndarray):
        xs = [xs] * len(datas)

    if ys is not None:
        if isinstance(ys, np.ndarray):
            ys = [ys] * len(datas)

    fig = go.Figure()
    if ys is not None:
        for data, x, y in zip(datas, xs, ys):
            fig.add_trace(go.Heatmap(visible=False, x=x,
                          y=y, z=data, **plot_kwargs))
            fig.data[0].visible = True
    else:
        for data, x in zip(datas, xs):
            x = np.atleast_2d(x)
            data = np.atleast_2d(data)
            if x.shape[0] == 1 and data.shape[0] != 1:
                x = np.tile(x, (data.shape[0], 1))
            for x, d, label in zip(x, data, labels):
                plot_kwargs["mode"] = plot_kwargs.pop("mode", "lines")
                fig.add_trace(
                    go.Scatter(x=x, y=d, visible=False,
                               name=label, **plot_kwargs)
                )

        for i in range(datas_per_step):
            fig.data[i].visible = True

    steps = []
    for i, (id, title) in enumerate(zip(ids, titles)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"{title}"},
                plot_kwargs,
            ],
            label=f"{id}",
        )
        for j in range(datas_per_step):
            step["args"][0]["visible"][
                i * datas_per_step + j
            ] = True  # Toggle i'th trace to visible
        steps.append(step)

    sliders = [dict(active=0, currentvalue={
                    "prefix": ""}, pad={"t": 50}, steps=steps)]
    fig.update_layout(
        sliders=sliders, title=titles[0], xaxis_title=xlabel, yaxis_title=ylabel
    )
    return fig


def add_vertical(fig, x):
    fig.update_layout(
        shapes=[dict(type="line", yref="paper", y0=0,
                     y1=1, xref="x", x0=x, x1=x)]
    )


def add_horizontal(fig, y):
    fig.update_layout(
        shapes=[dict(type="line", yref="y", y0=y,
                     y1=y, xref="paper", x0=0, x1=1)]
    )


def default_fig(rows=1, cols=1, **make_subplots_kwargs):
    if rows != 1 or cols != 1 or make_subplots_kwargs:
        fig = make_subplots(rows=rows, cols=cols, **make_subplots_kwargs)
    else:
        fig = go.Figure()
    apply_default_layout(fig)
    return fig


def apply_default_layout(fig):
    fig.update_layout(**default_layout)
    fig.update_xaxes(default_layout["xaxis"])
    fig.update_yaxes(default_layout["yaxis"])
    return fig


def heatmap(x, y, data, resample=True, **kwargs) -> go.Heatmap:
    """Shortcut to plotting heatmaps but after resampling so it doesn't take forever to plot"""
    max_pts = kwargs.pop("max_num_pnts", 300)
    if resample:
        data, x = resample_data(
            data, x=x, resample_x_only=True, max_num_pnts=max_pts)
    coloraxis = kwargs.pop("coloraxis", "coloraxis")
    hm = go.Heatmap(x=x, y=y, z=data, coloraxis=coloraxis, **kwargs)
    return hm


def error_fill(x, data, error, **kwargs):
    if isinstance(error, numbers.Number):
        error = [error] * len(x)
    x, data, error = [np.array(arr) for arr in [x, data, error]]
    upper = data + error
    lower = data - error

    # hacky way to prevent plotly autoscaling incorrectly
    upper[0] = np.nan
    lower[0] = np.nan

    fill_color = kwargs.pop("fill_color", "rgba(50, 50, 50, 0.2)")
    return go.Scatter(
        x=np.concatenate((x, x[::-1])),
        y=np.concatenate((upper, lower[::-1])),
        fill="tozeroy",
        fillcolor=fill_color,
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        **kwargs,
    )


def _figs_2d(figs):
    fig_is_2d = [False] * len(figs)
    for i, fig in enumerate(figs):
        for data in fig.data:
            if isinstance(data, go.Heatmap):
                fig_is_2d[i] = True
    return fig_is_2d


def _figs_contain_2d(figs):
    def are_all_true(l):
        return all(l)

    def are_some_true(l):
        return any(l) and not all(l)

    def are_all_false(l):
        return not all(l) and not any(l)

    fig_is_2d = _figs_2d(figs)
    if are_all_true(fig_is_2d):
        return "all"
    elif are_all_false(fig_is_2d):
        return "none"
    else:
        return "some"


def _figs_all_2d(figs):
    if _figs_contain_2d(figs) == "all":
        return True
    return False


def _figs_all_1d(figs):
    if _figs_contain_2d(figs) == "none":
        return True
    return False


def _move_2d_data(
    dest_fig: go.Figure,
    source_figs: list[go.Figure],
    fig_locations: list[tuple],
    match_colorscale: bool,
    specify_rows=None,  # If only moving a subset of figs
    specify_cols=None,  # If only moving a subset of figs
):  # , leave_legend_space=False):
    rows = max([l[0] for l in fig_locations]
               ) if specify_rows is None else specify_rows
    cols = max([l[1] for l in fig_locations]
               ) if specify_cols is None else specify_cols
    locations_axis_dict = {
        loc: i + 1 for i, loc in enumerate(get_subplot_locations(rows, cols))
    }

    if not match_colorscale:
        if cols == 1:
            xs = [1.00]
        elif cols == 2:
            xs = [0.43, 1.00]
        elif cols == 3:
            xs = [0.245, 0.625, 1.00]
        else:
            raise NotImplementedError
        # if leave_legend_space:
        #     xs = [x*0.8 for x in xs]
        if rows == 1:
            len_ = 1
            ys = [0.5]
        elif rows == 2:
            len_ = 0.4
            ys = [0.81, 0.19]
        elif rows == 3:
            len_ = 0.25
            ys = [0.89, 0.5, 0.11]
        else:
            raise NotImplementedError
        colorbar_locations = {
            (r, c): loc
            for (r, c), loc in zip(
                product(range(1, rows + 1), range(1, cols + 1)), product(ys, xs)
            )
        }

    # move data from each figure to subplots (matching colors)
    for fig, (row, col) in zip(source_figs, fig_locations):
        axis_num = locations_axis_dict[(row, col)]
        for j, data in enumerate(fig.data):
            if isinstance(data, go.Heatmap):
                if match_colorscale:
                    data.coloraxis = "coloraxis"
                else:
                    data.coloraxis = f"coloraxis{axis_num}"
            dest_fig.add_trace(data, row=row, col=col)
        if not match_colorscale:
            colorbar_location = colorbar_locations[(row, col)]
            dest_fig.update_layout(
                {f"coloraxis{axis_num}": fig.layout.coloraxis}
            )  # Copy across most info
            y, x = colorbar_location
            dest_fig.update_layout(
                {f"coloraxis{axis_num}_colorbar": dict(x=x, y=y, len=len_)}
            )  # Position the individual colorbar

        dest_fig.update_layout(
            {
                f"xaxis{axis_num}_title": fig.layout.xaxis.title,
                f"yaxis{axis_num}_title": fig.layout.yaxis.title,
            }
        )


def _move_1d_data(
    dest_fig: go.Figure,
    source_figs: list[go.Figure],
    fig_locations: list[tuple],
    match_colors: bool,
    no_legend=False,
    specify_rows=None,
    specify_cols=None,
):
    rows = max([l[0] for l in fig_locations]
               ) if specify_rows is None else specify_rows
    cols = max([l[1] for l in fig_locations]
               ) if specify_cols is None else specify_cols
    locations_axis_dict = {
        loc: i + 1 for i, loc in enumerate(get_subplot_locations(rows, cols))
    }

    # match the first figures colors if they were specified
    if (
        hasattr(source_figs[0].data[0], "line")
        and source_figs[0].data[0].line.color
        and match_colors
    ):
        colors = [d.line.color for d in source_figs[0].data]
    else:
        colors = pc.DEFAULT_PLOTLY_COLORS

    # move data from each figure to subplots (matching colors)
    for fig, (row, col) in zip(source_figs, fig_locations):
        axis_num = locations_axis_dict[(row, col)]
        showlegend = True if axis_num == 1 and not no_legend else False
        for j, data in enumerate(fig.data):
            color = colors[
                j % len(colors)
            ]  # % to cycle through colors if more data than colors
            if match_colors:
                data.update(showlegend=showlegend,
                            legendgroup=j, line_color=color)
            dest_fig.add_trace(data, row=row, col=col)
        dest_fig.update_layout(
            {
                f"xaxis{axis_num}_title": fig.layout.xaxis.title,
                f"yaxis{axis_num}_title": fig.layout.yaxis.title,
            }
        )


def _copy_annotations(dest_fig, source_figs):
    """Copy annotations to dest_fig (updating xref and yref if multiple source figs)
    Note: Does NOT copy annotations that use xref/yref = 'paper'
    """
    for i, fig in enumerate(source_figs):
        annotations = fig.layout.annotations
        for annotation in annotations:
            if annotation.xref != "paper" and annotation.yref != "paper":
                annotation.update(
                    xref=f"x{i + 1}",
                    yref=f"y{i + 1}",
                )
                dest_fig.add_annotation(annotation)


def _copy_shapes(dest_fig, source_figs):
    """Copy shapes to dest_fig (updating xref and yref if multiple source figs)"""
    for i, fig in enumerate(source_figs):
        # Plotly names axes 'x', 'x2', 'x3' etc.
        num_str = f"{i+1}" if i > 0 else ""
        shapes = fig.layout.shapes
        for shape in shapes:
            if shape.xref == "paper" and shape.yref == "paper":
                shape.update(
                    xref=f"x{num_str} domain",
                    yref=f"y{num_str} domain",
                )
            else:
                shape.update(
                    xref=shape.xref.replace("x", f"x{num_str}"),
                    yref=shape.yref.replace("y", f"y{num_str}"),
                )
                dest_fig.add_shape(shape)


def figures_to_subplots(
    figs, title=None, rows=None, cols=None, shared_data=False, **kwargs
):
    """
    Combine multiple plotly figures into a single figure with subplots where the legend and/or colorbar can be shared between them (only if all 2D or all 1D)
    """
    # set defaults
    if rows is None and cols is None:
        # if len(figs) == 1:
        #     return figs[0]
        # elif len(figs) == 2:
        #     rows, cols = 1, 2
        # elif len(figs) <= 4:
        #     rows, cols = 2, 2
        # elif len(figs) <= 6:
        #     rows, cols = 2, 3
        # elif len(figs) <= 9:
        #     rows, cols = 3, 3
        if len(figs) <= 9:
            cols = int(np.ceil(np.sqrt(len(figs))))
            rows = int(np.ceil(len(figs) / cols))
        else:
            raise NotImplementedError(f"Only implemented up to 3x3")

    if not rows:
        rows = 1 if cols else len(figs)
    if not cols:
        cols = 1

    # get single list of fig row/cols
    fig_locations = get_subplot_locations(rows, cols)
    figs_2d = _figs_2d(figs)

    if _figs_all_2d(figs):
        horizontal_spacing = 0.15 if not shared_data else None
        full_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                fig.layout.title.text if fig.layout.title.text else f"fig {i}"
                for i, fig in enumerate(figs)
            ],
            horizontal_spacing=horizontal_spacing,
            **kwargs,
        )
        _move_2d_data(
            dest_fig=full_fig,
            source_figs=figs,
            fig_locations=fig_locations,
            match_colorscale=shared_data,
        )
    elif _figs_all_1d(figs):
        full_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                fig.layout.title.text if fig.layout.title.text else f"fig {i}"
                for i, fig in enumerate(figs)
            ],
            **kwargs,
        )
        _move_1d_data(
            dest_fig=full_fig,
            source_figs=figs,
            fig_locations=fig_locations,
            match_colors=shared_data,
        )
        full_fig.update_layout(
            legend_title=figs[0].layout.legend.title
            if figs[0].layout.legend.title
            else "",
        )
    else:  # Some are 2D some are 1D  (Legends are removed, not easy to deal with...)
        horizontal_spacing = 0.15 if not shared_data else None
        full_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                fig.layout.title.text if fig.layout.title.text else f"fig {i}"
                for i, fig in enumerate(figs)
            ],
            horizontal_spacing=horizontal_spacing,
            **kwargs,
        )
        _move_2d_data(
            dest_fig=full_fig,
            source_figs=[fig for fig, is_2d in zip(
                figs, figs_2d) if is_2d is True],
            fig_locations=[
                location
                for location, is_2d in zip(fig_locations, figs_2d)
                if is_2d is True
            ],
            match_colorscale=shared_data,
            specify_rows=rows,
            specify_cols=cols,
        )
        _move_1d_data(
            dest_fig=full_fig,
            source_figs=[fig for fig, is_2d in zip(
                figs, figs_2d) if is_2d is False],
            fig_locations=[
                location
                for location, is_2d in zip(fig_locations, figs_2d)
                if is_2d is False
            ],
            match_colors=shared_data,
            no_legend=True,
            specify_rows=rows,
            specify_cols=cols,
        )

    _copy_annotations(dest_fig=full_fig, source_figs=figs)
    _copy_shapes(dest_fig=full_fig, source_figs=figs)
    apply_default_layout(full_fig)
    full_fig.update_layout(
        title=title,
        height=300 * rows,
    )
    return full_fig


def get_subplot_locations(rows, cols, invert=False):
    """Return a single list of the subplot locations in figure with rows/cols of subplots"""
    if invert:
        rows, cols = cols, rows
    return list(product(range(1, rows + 1), range(1, cols + 1)))


def fig_waterfall(fig: go.Figure, waterfall_state: bool):
    """Convert a fig to a waterfall plot (or inverse)"""
    if fig:
        fig = go.Figure(fig)
    if fig and fig.data:
        if (
            len(fig.data) == 1
            and isinstance(fig.data[0], go.Heatmap)
            and waterfall_state
        ):  # Convert from heatmap to waterfall
            hm = fig.data[0]
            fig.data = ()
            x = hm.x
            for r, y in zip(hm.z, hm.y):
                fig.add_trace(go.Scatter(x=x, y=r, name=str(y)))
            fig.update_layout(
                legend=dict(title=fig.layout.yaxis.title.text),
                yaxis_title=fig.layout.coloraxis.colorbar.title.text,
            )
        elif (
            len(fig.data) > 1
            and all([isinstance(d, go.Scatter) for d in fig.data])
            and 'xaxis2' not in fig.layout  # Subplots
            # All traces have a numeric name
            and np.all([re.search('^(?=.)[+-]?[0-9]*(?:\.[0-9]+)?$', str(t.name)) for t in fig.data])
            # Filled Scatter Traces
            and not np.any([d.fill != None for d in fig.data])
            and not waterfall_state
        ):  # Convert from waterfall to heatmap
            # Don't if there are subplots
            rows = fig.data
            fig.data = ()
            x = rows[0].x
            y = []
            for i, r in enumerate(rows):
                try:
                    name = r.name
                    v = float(name)
                except Exception as e:
                    v = i
                y.append(v)
            z = np.array([r.y for r in rows])
            fig.add_trace(go.Heatmap(x=x, y=y, z=z))
            fig.update_layout(
                yaxis_title=fig.layout.legend.title.text,
                legend=None,
                coloraxis=dict(colorbar=dict(
                    title=fig.layout.yaxis.title.text)),
            )
        else:
            pass
    return fig


def limit_max_datasize(fig: go.Figure, max_x=1000, max_y=1000, resample_x='decimate', resample_y='downsample'):
    """
    Reduces the datasize of data in figure by resampling
    Especially useful for plotting heatmaps that would otherwise contain millions of points

    Args:
        max_x: max no. datapoints per dataset in x direction
        max_y: max no. datapoints per dataset in y direction
        resample_x: 'none', 'decimate', 'bin', or 'downsample' to determine how data is resampled (default 'decimate')
        resample_y: 'none', 'bin', or 'downsample' to determine how data is resampled (default 'downsample')
    """

    def resample_data_x(data: Data):
        if data.x.shape[0] > max_x:
            bin_or_step_size = int(data.x.shape[0] / max_x)
            bin_or_step_size = 1 if bin_or_step_size < 1 else bin_or_step_size
            match resample_x:
                case 'decimate':
                    data = data.decimate(numpnts=max_x)
                case 'bin':
                    data = data.bin(bin_x=bin_or_step_size)
                case 'downsample':
                    if data.data.ndim == 2:
                        data = data[:, ::bin_or_step_size]
                    elif data.data.ndim == 1:
                        data = data[::bin_or_step_size]
                    else:
                        raise NotImplementedError
                case _:
                    pass
        return data

    def resample_data_y(data: Data):
        if data.y.shape[0] > max_y:
            bin_or_step_size = int(data.y.shape[0] / max_y)
            bin_or_step_size = 1 if bin_or_step_size < 1 else bin_or_step_size
            match resample_y:
                case 'decimate':
                    raise NotImplementedError
                case 'bin':
                    data = data.bin(bin_y=bin_or_step_size)
                case 'downsample':
                    data = data[::bin_or_step_size, :]
                case _:
                    pass
        return data

    def resample_heatmap(fdata: go.Heatmap):
        # Extract data from fig_data
        # Not ideal, but circular import otherwise
        from ...analysis_tools.data import Data
        x, y, d = [np.array(arr) for arr in [fdata.x, fdata.y, fdata.z]]
        data = Data(x=x, y=y, data=d)

        # Resample in x-dir
        data = resample_data_x(data)

        # Resample in y-dir
        data = resample_data_y(data)

        # Update fig_data
        fdata.x, fdata.y, fdata.z = data.x, data.y, data.data
        return fdata

    def resample_scatter(fdata: go.Scatter):
        # Extract data from fig_data
        # Not ideal, but circular import otherwise
        from ...analysis_tools.data import Data
        x, d = [np.array(arr) for arr in [fdata.x, fdata.y]]
        data = Data(x=x, data=d)

        # Resample in x-dir
        data = resample_data_x(data)

        # Update fig_data
        fdata.x, fdata.y = data.x, data.data
        return fdata

    def resample_error_fill(fdata: go.Scatter):
        # Not ideal, but circular import otherwise
        from ...analysis_tools.data import Data
        datas = []
        half_id = int(len(fdata.x) / 2)

        # Extract data from fig_data (two parts because error fill goes out and back and fills between)
        for direction, s_ in zip([1, -1], [np.s_[:half_id], np.s_[:half_id - 1:-1]]):
            x, d = [np.array(arr)[s_] for arr in [fdata.x, fdata.y]]
            data = Data(x=x, data=d)

            # Resample in x-dir
            data = resample_data_x(data)

            # Hacky way to prevent plotly autoscaling wrong with error fill (prevents including y=0 unnecessarily)
            data.data[0] = np.nan

            data = data[::direction]
            datas.append(data)

        # Update fig_data
        fdata.x, fdata.y = np.concatenate(
            [d.x for d in datas]), np.concatenate([d.data for d in datas])
        return fdata

    new_datas = []
    for fdata in fig.data:
        match fdata.type:
            case 'heatmap':
                new_datas.append(resample_heatmap(fdata))
            case 'scatter':
                if fdata.fill == 'tozeroy' and fdata.hoverinfo == 'skip':  # Error Fill
                    new_datas.append(resample_error_fill(fdata))
                else:
                    new_datas.append(resample_scatter(fdata))
            case _:
                logging.info(
                    f"Not implemented to downsample fig data of type {fdata.type}")
                new_datas.append(fdata)

    fig.data = new_datas
    return fig


def make_animated(fig: go.Figure, step_duration=0.1, copy=False, label_prefix='Datnum:'):
    """Adds animation to a slider figure

    Notes:
        Currently written for Heatmaps ONLY
        step_duration in s (converted to ms for plotly)
    """
    if copy:
        import copy

        fig = copy.deepcopy(fig)
    step_duration_ms = step_duration * 1000

    # Get the list of heatmaps and steps (TODO: Make work for scatter etc)
    heatmaps = [d for d in fig.data if isinstance(d, go.Heatmap)]
    steps = fig.layout.sliders[0].steps

    # Create Frames for animation and
    frames = []
    for step, heatmap in zip(steps, heatmaps):
        heatmap.update(visible=None)
        frames.append(
            go.Frame(
                name=step.label, data=heatmap, layout=dict(
                    title=step.args[1]["title"])
            )
        )
        step.update(
            {
                "args": [
                    [step.label],
                    {
                        "mode": "immediate",
                    },
                ],
                "method": "animate",
            }
        )
    fig.frames = frames

    # Doesn't need all the data now, but at least 1 is necessary!
    fig.data = [heatmaps[0]]

    # Update the sliders dict
    fig.layout.sliders[0].update(
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": label_prefix,
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
        }
    )

    # Add animate buttons to fig
    fig.update_layout(
        transition_duration=step_duration_ms,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": step_duration_ms, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )
    return fig


def get_colors(values: [np.ndarray, list[float], float], low=0, high=1, colorscale: [str, Any] = "bluered"):
    """Get list of colors corresponding to values from given colorscale
    Args:
        values: Values to get colors for
        low: Lowest value to use for colorscale (e.g. min of ALL data)
        high: Highest value to use for colorscale (e.g. max of ALL data)
        colorscale: Colorscale to use, can be anything that pc.sample_colorscale takes (e.g. list of colors etc)

    """
    values = np.asanyarray(values).flatten()
    color_val = (values - low) / (high - low)
    colors = pc.sample_colorscale(colorscale, color_val)
    if values.size == 1:
        return colors[0]
    return colors


if __name__ == "__main__":
    pass
