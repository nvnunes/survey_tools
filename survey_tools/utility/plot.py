#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import matplotlib.pyplot as plt
import numpy as np
import pyds9

class PlotException(Exception):
    pass

class StructType:
    pass

def get_plot_colours(plot_mode, use_accessible_colours = True):
    plot_colours = StructType()

    if plot_mode == 'latex' or use_accessible_colours:
        # Accessible Colours
        # see: https://towardsdatascience.com/two-simple-steps-to-create-colorblind-friendly-data-visualizations-2ed781a167ec
        # see: https://colorbrewer2.org/#type=sequential&scheme=GnBu&n=7
        plot_colours.colour1 = '#f0f9e8'
        plot_colours.colour2 = '#ccebc5'
        plot_colours.colour3 = '#a8ddb5'
        plot_colours.colour4 = '#7bccc4'
        plot_colours.colour5 = '#4eb3d3'
        plot_colours.colour6 = '#2b8cbe'
        plot_colours.colour7 = '#08589e'
    else:
        # see: https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=6
        plot_colours.colour1 = '#d53e4f'
        plot_colours.colour2 = '#fc8d59'
        plot_colours.colour3 = '#fee08b'
        plot_colours.colour4 = '#e6f598'
        plot_colours.colour5 = '#99d594'
        plot_colours.colour6 = '#3288bd'
        plot_colours.colour7 = '#800080'

    plot_colours.colours = [
        plot_colours.colour1,
        plot_colours.colour2,
        plot_colours.colour3,
        plot_colours.colour4,
        plot_colours.colour5,
        plot_colours.colour6,
        plot_colours.colour7
    ]

    plot_colours.fit_data_colour  = plot_colours.colour6
    plot_colours.fit_line_colour  = plot_colours.colour2
    plot_colours.fit_line_colour2 = plot_colours.colour4

    plot_colours.hist_colour      = plot_colours.colour6
    plot_colours.hist2_colour     = plot_colours.colour4
    plot_colours.hist3_colour     = plot_colours.colour2
    plot_colours.hist4_colour     = plot_colours.colour1
    plot_colours.hist_label_color = plot_colours.colour4

    return plot_colours

def create_plot(plot_mode='notebook', subplots=None, figsize=None, title=None, projection=None, width_ratios=None, height_ratios=None, **kwargs):
    if plot_mode == 'latex':
        if projection is not None:
            fig, ax = plt.subplots(subplot_kw=dict(projection=projection), **kwargs)
        else:
            fig, ax = plt.subplots(**kwargs)
    else:
        if subplots is None:
            fig = plt.figure(figsize=figsize, **kwargs)

            if title is not None:
                plt.title(title)

            if projection is not None:
                ax = plt.subplot(1, 1, 1, projection=projection)
            else:
                ax = plt.subplot(1, 1, 1)
        else:
            if len(kwargs) > 0:
                subplot_kw = kwargs
            else:
                subplot_kw=dict()

            if projection is not None:
                subplot_kw['projection'] = projection

            fig, ax = plt.subplots(*subplots, figsize=figsize, height_ratios=height_ratios, width_ratios=width_ratios, subplot_kw=subplot_kw)

            if title is not None:
                fig.suptitle(title)

    return fig, ax

def save_plot(filename):
    plt.savefig(filename)

def format_hist(ax, bins=None, values=None, xtickspacing=None, xforceinteger=False, ytickspacing=None, yforceinteger=False):
    if xforceinteger:
        ax.locator_params(axis='x', integer=True)
    if bins is not None and xtickspacing is not None:
        ax.set_xlim([np.floor(np.min(bins)/xtickspacing)*xtickspacing, np.ceil(np.max(bins)/xtickspacing)*xtickspacing])
    if yforceinteger:
        ax.locator_params(axis='y', integer=True)
    if values is not None and ytickspacing is not None:
        ax.set_ylim([np.floor(np.min(values)/ytickspacing)*ytickspacing, np.ceil(np.max(values)/ytickspacing)*ytickspacing])

def add_annotation(ax, text, position='topleft', border=False):
    match position:
        case 'topleft':
            xy = [0.075,0.905]
            horizontalalignment = 'left'
            verticalalignment = 'top'
        case 'topcenter':
            xy = [0.5,0.905]
            horizontalalignment = 'center'
            verticalalignment = 'top'
        case 'topright':
            xy = [0.925,0.905]
            horizontalalignment = 'right'
            verticalalignment = 'top'
        case 'bottomleft':
            xy = [0.075,0.095]
            horizontalalignment = 'left'
            verticalalignment = 'bottom'
        case 'bottomcenter':
            xy = [0.5,0.095]
            horizontalalignment = 'center'
            verticalalignment = 'bottom'
        case 'bottomright':
            xy = [0.925,0.095]
            horizontalalignment = 'right'
            verticalalignment = 'bottom'

    if border:
        bbox = dict(boxstyle="square", linewidth=1, facecolor='white')
    else:
        bbox = None

    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=horizontalalignment, va=verticalalignment, bbox=bbox)

def DS9_plot_image(image_hdul):
    ds9 = pyds9.DS9()
    ds9.set_pyfits(image_hdul) # pylint: disable=too-many-function-args
    ds9.set('scale pow')
    ds9.set('scale mode zscale')
    ds9.set('cmap grey')
    ds9.set('cmap invert yes')
    ds9.set('zoom to fit')
    return ds9
