#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import copy
from logging import warning
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist import angle_helper
import numpy as np
from astropy.table import Table
from astropy.coordinates import Longitude, Latitude, SkyCoord
import astropy.units as u
import astropy_healpix as healpix
from astropy_healpix import HEALPix
import skyproj
from survey_tools.utility.rotator import Rotator

class HealpixException(Exception):
    pass

def _get_nside(level):
    return 2**level

def get_npix(level):
    return healpix.nside_to_npix(_get_nside(level))

def get_level(npix):
    if is_npix_valid(npix):
        return int(np.log2(np.sqrt(npix/12.0)))
    return None

def get_area(level):
    return healpix.nside_to_pixel_area(_get_nside(level))

def get_resolution(level):
    return healpix.nside_to_pixel_resolution(_get_nside(level))

def is_npix_valid(npix):
    nside = np.sqrt(npix/12.0)
    return nside == int(nside)

#region SkyCoords

def get_healpix_from_skycoord(level, skycoords, frame='icrs'):
    hp = HEALPix(nside=_get_nside(level), order='nested', frame=frame)
    return hp.skycoord_to_healpix(skycoords)

def get_boundaries_skycoord(level, pixs=None, step=1, frame='icrs'):
    hp = HEALPix(nside=_get_nside(level), order='nested', frame=frame)

    if pixs is None:
        pixs = np.arange(hp.npix)

    boundaries = hp.boundaries_skycoord(pixs, step)

    if np.size(pixs) == 1:
        return boundaries[0]

    return boundaries

def filter_skycoords(skycoords, coords, frame='icrs'):
    if len(coords) == 4:
        (min_lon, max_lon, min_lat, max_lat) = coords
    else:
        raise HealpixException('Invalid number of arguments')

    if frame == 'galactic':
        lon = skycoords.l.degree
        lat = skycoords.b.degree
    else:
        lon = skycoords.ra.degree
        lat = skycoords.dec.degree

    if min_lon > max_lon:
        lon_filter = (lon >= min_lon) | (lon < max_lon)
    else:
        lon_filter = (lon >= min_lon) & (lon < max_lon)

    lat_filter = (lat >= min_lat) & (lat < max_lat)

    return lon_filter & lat_filter

def get_pixels(level, coords=(), frame='icrs'):
    pixs = np.arange(get_npix(level))
    if len(coords) == 0:
        return pixs
    else:
        skycoords = get_pixel_skycoord(level, pixs, frame=frame)
        row_filter = filter_skycoords(skycoords, coords, frame=frame)
        return pixs[row_filter]

def get_pixel_skycoord(level, pixs=None, coords=(), frame='icrs'):
    hp = HEALPix(nside=_get_nside(level), order='nested', frame=frame)
    if pixs is None:
        pixs = get_pixels(level, coords, frame=frame)
    return hp.healpix_to_skycoord(pixs)

def get_pixel_details(level, coords=(), frame='icrs'):
    pixs = get_pixels(level, coords, frame)
    skycoords = get_pixel_skycoord(level, pixs=pixs, frame=frame)
    return Table([pixs, skycoords.ra.degree, skycoords.dec.degree], names=['pix', 'ra', 'dec'])

#endregion

#region Subpixels

def get_parent_pixel(level, pix, outer_level):
    if outer_level > level:
        raise HealpixException('Outer Level must be smaller than inner level')
    return pix // 4**(level - outer_level)

def get_neighbours(level, pix):
    return healpix.neighbours(pix, _get_nside(level), order='nested')

def get_subpixel_npix(outer_level, inner_level):
    return 4**(inner_level - outer_level)

def _get_subpixels_min_max(outer_level, outer_pix, inner_level):
    inner_pixels_per_outer_pixel = get_subpixel_npix(outer_level, inner_level)
    min_pix = outer_pix * inner_pixels_per_outer_pixel
    max_pix = min_pix + inner_pixels_per_outer_pixel
    return (min_pix, max_pix)

def get_subpixel_indexes(level, pix, inner_level, outer_level):
    if level <= outer_level:
        raise HealpixException('level must be larger than outer level')
    if level > inner_level:
        raise HealpixException('level must be smaller than inner level')

    parent_outer_pix = get_parent_pixel(level, pix, outer_level)
    (outer_start_pixel, _) = _get_subpixels_min_max(outer_level, parent_outer_pix, inner_level)
    (start_pixel, stop_pixel) = _get_subpixels_min_max(level, pix, inner_level)

    return np.arange(start = start_pixel - outer_start_pixel, stop = stop_pixel - outer_start_pixel)

def get_subpixels(outer_level, outer_pix, inner_level):
    if inner_level <= outer_level:
        raise HealpixException('Inner level must be larger than outer level')

    if np.size(outer_pix) == 1:
        if isinstance(outer_pix, list) or isinstance(outer_pix, np.ndarray):
            outer_pix = outer_pix[0]
        (start_pixel, stop_pixel) = _get_subpixels_min_max(outer_level, outer_pix, inner_level)
        return np.arange(start = start_pixel, stop = stop_pixel)

    pixs = np.array([], dtype=np.int_)
    for pix in outer_pix:
        (start_pixel, stop_pixel) = _get_subpixels_min_max(outer_level, pix, inner_level)
        pixs = np.concatenate((pixs, np.arange(start = start_pixel, stop = stop_pixel)))

    return pixs

def get_subpixels_skycoord(outer_level, outer_pix, inner_level, frame='icrs'):
    pixs = get_subpixels(outer_level, outer_pix, inner_level)

    hp = HEALPix(nside=2**inner_level, order='nested', frame=frame)
    skycoords = hp.healpix_to_skycoord(pixs)

    return (pixs, skycoords)

def get_subpixels_detail(outer_level, outer_pix, inner_level, frame='icrs'):
    (pixs, skycoords) = get_subpixels_skycoord(outer_level, outer_pix, inner_level, frame=frame)

    subpixels = Table([pixs, skycoords.ra.degree, skycoords.dec.degree], names=['pix', 'ra', 'dec'])
    subpix_resolution = get_resolution(inner_level)

    return (subpixels, subpix_resolution)

#endregion

#region Plotting

def plot(values, level=None, pixs=None, skycoords=None, contour_values=None, plot_properties=None):
    _set_default_plot_properties(values, contour_values, plot_properties)

    with mpl.rc_context({
        'xtick.labelcolor': plot_properties['colors']['xtick_label'], # not supported in skyproj 1.x
        'xtick.labelsize': plot_properties['fontsize']['xtick_label'],
        'ytick.labelcolor': plot_properties['colors']['ytick_label'], # not supported in skyproj 1.x
        'ytick.labelsize': plot_properties['fontsize']['ytick_label']
    }):
        fig, ax = plt.subplots(figsize=(plot_properties['width'], plot_properties['height']), dpi=plot_properties['dpi'])

        kwargs = {
            'celestial': plot_properties['flip'],
            'galactic': plot_properties['mapcoord'] == 'G',
            'lon_0': plot_properties['rotation'],
            'gridlines': plot_properties['grid'],
            'longitude_ticks': 'symmetric' if plot_properties['grid_longitude'] == 'degrees' else 'positive',
        }

        match plot_properties['projection']:
            case 'mollweide':
                sp = skyproj.MollweideSkyproj(ax, **kwargs)
            case 'gnomonic':
                sp = skyproj.GnomonicSkyproj(ax, **kwargs)
            case _:
                sp = skyproj.Skyproj(ax, **kwargs)

        plot_properties['fig'] = fig
        plot_properties['ax'] = fig.gca() # reload ax after skyproj
        plot_properties['sp'] = sp

        _draw_map(values, level, pixs, skycoords, **plot_properties) # pylint: disable=possibly-used-before-assignment

        if contour_values is not None:
            contour_plot_properties = plot_properties.copy()
            contour_plot_properties.update({
                'cmap': plot_properties['contour_cmap'],
                'norm': plot_properties['contour_norm'],
                'vmin': plot_properties['contour_vmin'],
                'vmax': plot_properties['contour_vmax']
            })
            _draw_map(contour_values, None, None, None, plot_type='contour', **contour_plot_properties)

        _draw_grid(**plot_properties)
        _draw_cbar(**plot_properties)
        _draw_boundaries(**plot_properties)

        if plot_properties.get('milkyway', False):
            sp.draw_milky_way(
                width=plot_properties['milkyway_width'] if plot_properties.get('milkyway_width', None) is not None else 10,
                linewidth=1.5,
                color=plot_properties['colors']['milkyway'],
                linestyle='-'
            )

        if plot_properties.get('ecliptic', False):
            _draw_ecliptic(
                sp = plot_properties['sp'],
                galactic=plot_properties['galactic'],
                width=plot_properties['ecliptic_width'] if plot_properties.get('ecliptic_width', None) is not None else 10,
                linewidth=1.5,
                color=plot_properties['colors']['ecliptic'],
                linestyle='-'
            )

        if plot_properties.get('tissot', False):
            sp.tissot_indicatrices()

        _finish_plot(**plot_properties)

        return fig

GLOBE_PROJECTIONS = ['aitoff', 'hammer', 'lambert', 'mollweide']

def _update_dictionary(dict1, dict2):
    for key in dict1.keys():
        if key in dict2:
            dict1[key] = dict2[key]
    return dict1

def _set_default_plot_properties(values, contour_values, plot_properties=None):
    """Plot Parameters:
       ---------------
    galactic : bool, optional
      If True, the plot will be in galactic coordinates. Default: False
    projection :  {'aitoff', 'hammer', 'lambert', 'mollweide', 'cartesian', 'polar'}
      type of the plot
    flip : bool, optional
      Defines the longitude convention:
        True (astro) - default, east towards left, west towards right
        False (geo) - ast towards right, west towards left
    rotation : scalar, optional
      Logitudinal rotation to apply
    width: scalar, optional
      Width of the plot. Default: 10
    height: scalar, optional
      Height of the plot. Default: width * 0.63
    xsize : int, optional
      Size of the image. Default: 1000
    cmap : str, optional
      Specify the colormap. default: Viridis
    norm : {'hist', 'log', 'symlog', None}
      Color normalization:
      hist = histogram equalized color mapping.
      log = logarithmic color mapping.
      symlog = symmetric logarithmic, linear between -linthresh and linthresh.
      default: None (linear color mapping)
    contour_cmap : str, optional
      Specify the colormap for the contour. default: None
    contour_norm : see norm, optional
      Color normalization for the contour. default: None
    contour_levels : int, optional
      Number of levels for the contour. default: 10
    vmin : float
      The minimum range value
    vmax : float
      The maximum range value
    grid: bool
      show grid lines.
    cbar : bool, optional
      Show the colorbar. Default: True
    cbar_orientation : {'vertical', 'horizontal'}
      color bar orientation
    cbar_ticks : list
      custom ticks on the colorbar
    cbar_format : str, optional
      The format of the scale label. Default: '%g'
    cbar_unit : str, optional
      A text describing the unit of the data. Default: ''
    boundaries_level : int, optional
      The level of the boundaries to show. Default: None
    colors: dict, optional
      Override default colors
    fontsize:  dict, optional
      Override default fontsize of labels
"""
    if plot_properties is None:
        plot_properties = {}

    # Map Coordinates
    if plot_properties.get('mapcoord', None) is not None:
        raise HealpixException('mapcoord not supported with skyproj')
    if plot_properties.get('galactic', False):
        plot_properties['mapcoord'] = 'G'
    else:
        plot_properties['mapcoord'] = 'C'

    # Projection
    if plot_properties.get('projection', None) is None:
        plot_properties['projection'] = 'cartesian'

    if plot_properties['projection'] in ['cart', 'equi', 'equirectangular', 'rectangular', 'lacarte', 'platecarree']:
        plot_properties['projection'] = 'cartesian'

    if plot_properties.get('zoom', None) is None:
        plot_properties['zoom'] = False

    if plot_properties.get('rotation', None) is None:
        plot_properties['rotation'] = 0.0

    if plot_properties.get('flip', None) is None:
        plot_properties['flip'] = True

    # Size
    if plot_properties.get('width', None) is None:
        plot_properties['width'] = 10

    if plot_properties.get('height', None) is None:
        plot_properties['height'] = plot_properties['width']/1.618

    if plot_properties.get('dpi', None) is None:
        plot_properties['dpi'] = 100

    if plot_properties.get('xsize', None) is None:
        plot_properties['xsize'] = int(plot_properties['width'] * 0.8 * plot_properties['dpi'])

    # Grid
    if plot_properties.get('grid', None) is None:
        plot_properties['grid'] = True

    if plot_properties.get('grid_longitude', None) is None:
        plot_properties['grid_longitude'] = 'degrees'

     # Colors
    color_defaults = {
        'background': 'white',
        'badvalue': 'gray',
        'grid': (0.8,0.8,0.8) if values is not None else (0.5,0.5,0.5),
        'xtick_label': 'black',
        'ytick_label': 'black',
        'boundaries': 'red',
        'milkyway': 'black',
        'ecliptic': 'black'
    }

    if plot_properties['projection'] in GLOBE_PROJECTIONS:
        color_defaults['xtick_label'] = color_defaults['grid']

    if 'colors' in plot_properties and plot_properties['colors'] is not None:
        plot_properties['colors'] = _update_dictionary(color_defaults,  plot_properties['colors'])
    else:
        plot_properties['colors'] = color_defaults

   # Fonts
    fontsize_defaults = {
        'xlabel': 12,
        'ylabel': 12,
        'title': 14,
        'xtick_label': 10,
        'ytick_label': 10,
        'cbar_label': 12,
        'cbar_tick_label': 10,
        'boundaries_label': 12,
        'boundaries_label_small': max(8, 12.0-(plot_properties.get('boundaries_level') or 0)/2.0)
    }

    if 'fontsize' in plot_properties and plot_properties['fontsize'] is not None:
        plot_properties['fontsize'] = _update_dictionary(fontsize_defaults,  plot_properties['fontsize'])
    else:
        plot_properties['fontsize'] = fontsize_defaults

    # Color Map
    cmap = plot_properties.get('cmap', None)
    if cmap is None:
        cmap = 'viridis'

    if isinstance(cmap, str):
        cmap0 = plt.get_cmap(cmap)
    elif isinstance(cmap, mpl.colors.Colormap):
        cmap0 = cmap
    else:
        cmap0 = plt.get_cmap(mpl.rcParams['image.cmap'])

    cmap = copy.copy(cmap0)
    cmap.set_over(plot_properties['colors']['badvalue'])
    cmap.set_under(plot_properties['colors']['badvalue'])
    cmap.set_bad(plot_properties['colors']['badvalue'])

    plot_properties['cmap'] = cmap

    # Normalization
    if values is not None:
        if plot_properties.get('vmin', None) is None and plot_properties.get('cbar_ticks', None) is not None:
            plot_properties['vmin'] = np.min(plot_properties['cbar_ticks'])

        if plot_properties.get('vmax', None) is None and plot_properties.get('cbar_ticks', None) is not None:
            plot_properties['vmax'] = np.max(plot_properties['cbar_ticks'])

        plot_properties['norm'], plot_properties['vmin'], plot_properties['vmax'] = _prepare_norm(
            values,
            plot_properties.get('norm', None),
            plot_properties.get('vmin', None),
            plot_properties.get('vmax', None)
        )

    # Contours

    if contour_values is not None:
        if 'contour_cmap' not in plot_properties:
            plot_properties['contour_cmap'] = None

        plot_properties['contour_norm'], plot_properties['contour_vmin'], plot_properties['contour_vmax'] = _prepare_norm(
            contour_values,
            plot_properties.get('contour_norm', None),
            plot_properties.get('contour_vmin', None),
            plot_properties.get('contour_vmax', None)
        )

        if 'contour_levels' not in plot_properties:
            plot_properties['contour_levels'] = None

    return plot_properties

def _prepare_norm(values, norm, vmin, vmax):
    is_valid = ~np.isnan(values) & ~np.isinf(values)

    if vmin is None:
        if np.sum(is_valid) > 0:
            vmin = np.min(values[is_valid])

    if vmax is None:
        if np.sum(is_valid) > 0:
            vmax = np.max(values[is_valid])

    if norm is None:
        norm = mpl.colors.NoNorm()
    elif isinstance(norm, str):
        match norm.lower():
            case 'lin' | 'norm' | 'normalize':
                norm = mpl.colors.Normalize()
            case 'log':
                norm = mpl.colors.LogNorm()
            case 'symlog':
                norm = mpl.colors.SymLogNorm(1, linscale=0.1, clip=True, base=10) # pylint: disable=unexpected-keyword-arg
            case _:
                raise HealpixException('Unrecognized norm')

    norm.vmin = vmin
    norm.vmax = vmax
    norm.autoscale_None(values[is_valid])

    return norm, vmin, vmax

def _draw_map(values, level, pixs, skycoords,
    plot_type='mesh',
    sp=None,
    mapcoord=None,
    zoom=False,
    xsize=1000,
    cmap=None,
    norm=None,
    **kwargs # pylint: disable=unused-argument
):
    if values is None:
        return

    if plot_type not in ['mesh', 'contour']:
        raise HealpixException('Invalid map type')

    num_values = len(values)

    if level is None:
        if not is_npix_valid(num_values):
            raise HealpixException('level and pixs required when passing a partial map')
        level = get_level(num_values)

    npix = get_npix(level)

    if pixs is None:
        if num_values != npix:
            raise HealpixException('pixs must be specified when values are not a full HEALpix map')
    else:
        if num_values == npix:
            pixs = None
        elif num_values != len(pixs):
            raise HealpixException('values and pixs must be the same size')

    if pixs is not None:
        if mapcoord != 'C':
            raise HealpixException('rotating coordinates is not supported when pixs specified')

        if plot_type == 'contour':
            raise HealpixException('contour plot is not support when pixs specified')

        sp.draw_hpxpix(_get_nside(level), pixs, values, nest=True, zoom=zoom, xsize=xsize, cmap=cmap, norm=norm)
    else:
        if mapcoord != 'C':
            if skycoords is None:
                skycoords = get_pixel_skycoord(level, pixs)

            rotator = Rotator(coord=['C', mapcoord], inv=True)
            theta_cel = np.deg2rad(90 - skycoords.dec.degree)
            phi_cel = np.deg2rad(skycoords.ra.degree)
            theta, phi = rotator(theta_cel, phi_cel)
            lon = np.rad2deg(phi) % 360.0
            lat = 90 - np.rad2deg(theta)
            remap_pixs = get_healpix_from_skycoord(level, SkyCoord(lon, lat, unit=(u.degree, u.degree)))
            values = values[remap_pixs]

        if plot_type == 'contour':
            _draw_contour_map(sp, values, xsize=xsize, cmap=cmap, norm=norm, **kwargs)
        else:
            sp.draw_hpxmap(values, nest=True, zoom=False, xsize=xsize, cmap=cmap, norm=norm)

def _draw_contour_map(sp, values,
        ax = None,
        projection='cart',
        xsize=1000,
        cmap=None,
        norm=None,
        contour_levels=None,
        **kwargs # pylint: disable=unused-argument
    ):

    if projection != 'cartesian':
        warning('Contour plot only works correctly with the cartesian projection')

    extent = sp.get_extent()
    lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
    lat_range = [extent[2], extent[3]]

    aspect = 0.5
    lon, lat = np.meshgrid(
        np.linspace(lon_range[0], lon_range[1], xsize),
        np.linspace(lat_range[0], lat_range[1], int(aspect*xsize))
    )

    hp = HEALPix(nside=_get_nside(get_level(len(values))), order='nested', frame='icrs')
    pixs = hp.lonlat_to_healpix(Longitude(lon, unit=u.degree), Latitude(lat, unit=u.degree))
    values_raster = values[pixs]

    ax.contour(lon, lat, np.ma.array(values_raster, mask=np.isnan(values_raster)),
        levels=contour_levels,
        transform=ax.projection if projection != 'cartesian' else None,
        cmap=cmap,
        colors='k' if cmap is None else None,
        norm=norm,
        vmin=norm.vmin if norm is None else None,
        vmax=norm.vmax if norm is None else None
    )

def _draw_grid(
        sp=None,
        ax=None,
        zoom=False,
        grid=True,
        grid_longitude=None,
        colors=None,
        **kwargs # pylint: disable=unused-argument
):
    if not hasattr(ax, 'gridlines'): # version 1.x
        is_old_version = True
        aa = sp._aa # pylint: disable=protected-access
        gridlines = aa.gridlines
    else:
        is_old_version = False
        gridlines = ax.gridlines

    if not grid:
        if is_old_version:
            aa.axis['left'].major_ticklabels.set_visible(False)
            aa.axis['right'].major_ticklabels.set_visible(False)
            aa.axis['bottom'].major_ticklabels.set_visible(False)
            aa.axis['top'].major_ticklabels.set_visible(False)

            if sp._boundary_labels: # pylint: disable=protected-access
                for label in sp._boundary_labels: # pylint: disable=protected-access
                    label.remove()
                sp._boundary_labels = [] # pylint: disable=protected-access

        return

    gridlines.set_edgecolor(colors['grid'])

    # The following is a HACK to add support for:
    # 1. Longitude in Hours instead of Degrees
    # 2. Tick with minute and second divisions if appropriate 

    grid_helper = gridlines._grid_helper # pylint: disable=protected-access

    if is_old_version:
        n_grid_lon, n_grid_lat = sp._compute_n_grid_from_extent( # pylint: disable=protected-access
            ax.get_extent(lonlat=True)
        )
    else:
        n_grid_lon, n_grid_lat = grid_helper._compute_n_grid_from_extent( # pylint: disable=protected-access
            ax.get_extent(),
            n_grid_lon_default=6,
            n_grid_lat_default=6,
        )

    match grid_longitude:
        case 'hours':
            class WrappedFormatterHMS(angle_helper.FormatterHMS):
                def __init__(self):
                    self._formatter = skyproj.mpl_utils.WrappedFormatterDMS(180, sp._longitude_ticks) # pylint: disable=protected-access

                def __call__(self, direction, factor, values):
                    return super().__call__(direction, factor, self._formatter._wrap_values(factor, values))

            lon_locator = angle_helper.LocatorHMS(n_grid_lon, include_last=zoom)
            lon_formatter = WrappedFormatterHMS()
        case _:
            lon_locator = angle_helper.LocatorDMS(n_grid_lon, include_last=zoom)
            lon_formatter = None

    lat_locator = angle_helper.LocatorDMS(n_grid_lat, include_last=True)

    if is_old_version:
        grid_helper.update_grid_finder(grid_locator1 = lon_locator, grid_locator2 = lat_locator)
        if lon_formatter is not None:
            sp._tick_formatter1 = lon_formatter # pylint: disable=protected-access
            grid_helper.update_grid_finder(tick_formatter1 = lon_formatter)
    else:
        grid_helper._grid_locator_lon = lon_locator # pylint: disable=protected-access
        grid_helper._grid_locator_lat = lat_locator # pylint: disable=protected-access
        if lon_formatter is not None:
            grid_helper._tick_formatters['lon'] = lon_formatter # pylint: disable=protected-access

    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    grid_helper._update_grid(x1, y1, x2, y2) # pylint: disable=protected-access

    if is_old_version:
        sp._draw_aa_bounds_and_labels() # pylint: disable=protected-access

def _draw_boundaries(
        ax=None,
        sp=None,
        mapcoord='C',
        zoom=False,
        rotation=0.0,
        boundaries_level=None,
        boundaries_pixs=None,
        colors=None,
        fontsize=None,
        **kwargs # pylint: disable=unused-argument
):
    if boundaries_level is None:
        return

    if boundaries_pixs is not None and not isinstance(boundaries_pixs, list) and not isinstance(boundaries_pixs, np.ndarray):
        boundaries_pixs = [boundaries_pixs]

    xlim = np.sort(np.rad2deg(ax.get_xlim()))
    ylim = np.sort(np.rad2deg(ax.get_ylim()))

    step = max(1,2**(7-boundaries_level)) # level 0 = 128, level 1 = 64, level 2 = 32, ...
    pixs = np.arange(get_npix(boundaries_level))
    skycoords = get_pixel_skycoord(boundaries_level, pixs)
    boundaries = get_boundaries_skycoord(boundaries_level, pixs=pixs, step=step)
    boundaries_label_fontsize = fontsize['boundaries_label_small'] if not zoom else fontsize['boundaries_label']

    if mapcoord == 'G':
        boundaries_lon = boundaries.galactic.l.to(u.degree).value
        boundaries_lat = boundaries.galactic.b.to(u.degree).value
        center_lon = skycoords.galactic.l.to(u.degree).value
        center_lat = skycoords.galactic.b.to(u.degree).value
    else:
        boundaries_lon = boundaries.ra.to(u.degree).value
        boundaries_lat = boundaries.dec.to(u.degree).value
        center_lon = skycoords.ra.to(u.degree).value
        center_lat = skycoords.dec.to(u.degree).value

    xlim += rotation

    max_count = 250
    count = 0
    for i, pix in enumerate(pixs):
        vertices = np.vstack([boundaries_lon[i], boundaries_lat[i]]).transpose()

        if zoom and not np.any((vertices[:, 0] >= xlim[0]) & (vertices[:, 0] <= xlim[1]) & (vertices[:, 1] >= ylim[0]) & (vertices[:, 1] <= ylim[1])):
            continue

        count += 1
        if count > max_count:
            plt.clf()
            raise HealpixException('Too many boundaries to plot')

        sp.draw_polygon(lon=vertices[:,0], lat=vertices[:,1], edgecolor=colors['boundaries'])

        if boundaries_pixs is not None and pix not in boundaries_pixs:
            continue

        if zoom and ((center_lon[i] < xlim[0]) or (center_lon[i] > xlim[1]) or (center_lat[i] < ylim[0]) or (center_lat[i] > ylim[1])):
            continue

        ax.text(center_lon[i], center_lat[i], f"{pix}", color=colors['boundaries'], fontsize=boundaries_label_fontsize, ha='center', va='center')

def _draw_ecliptic(
        sp=None,
        galactic=False,
        width=10,
        linewidth=1.5,
        color='black',
        linestyle='-',
        **kwargs # pylint: disable=unused-argument
    ):

    elon = np.linspace(0, 360, 500)
    elat = np.zeros_like(elon)
    ec = SkyCoord(lon=elon*u.degree, lat=elat*u.degree, frame='barycentricmeanecliptic')

    if galactic:
        lon = ec.galactic.l.degree
        lat = ec.galactic.b.degree
    else:
        lon = ec.fk5.ra.degree
        lat = ec.fk5.dec.degree

    sp.plot(lon, lat, linewidth=linewidth, color=color, linestyle=linestyle, **kwargs)
    # pop any labels
    # kwargs.pop('label', None)

    if width > 0:
        for delta in [+width, -width]:
            ec = SkyCoord(lon=elon*u.degree, lat=(elat + delta)*u.degree, frame='barycentricmeanecliptic')
            if galactic:
                lon = ec.galactic.l.degree
                lat = ec.galactic.b.degree
            else:
                lon = ec.fk5.ra.degree
                lat = ec.fk5.dec.degree

            sp.plot(lon, lat, linewidth=1.0, color=color, linestyle='--', **kwargs)

def _draw_cbar(
        sp=None,
        vmin=None,
        vmax=None,
        cbar=True,
        cbar_orientation='vertical',
        cbar_ticks=None,
        cbar_format='%g',
        cbar_pad=0.03,
        cbar_shrink=0.6,
        cbar_unit='',
        fontsize=None,
        **kwargs # pylint: disable=unused-argument
    ):

    if not cbar or vmin is None or vmax is None:
        return

    cb = sp.draw_colorbar(
        ticks=cbar_ticks,
        format=cbar_format,
        fontsize=fontsize['cbar_tick_label'],
        location='bottom' if cbar_orientation == 'horizontal' else 'right',
        shrink=cbar_shrink,
        pad=cbar_pad
    )

    cb.set_label(label=cbar_unit, fontsize=fontsize['cbar_label'])

def _finish_plot(
    ax=None,
    projection='cartesian',
    title=None,
    xlabel=None,
    ylabel=None,
    fontsize=None,
    **kwargs # pylint: disable=unused-argument
):
    if title is not None:
        title_pad = 25 if projection not in GLOBE_PROJECTIONS else 10
        ax.set_title(title, pad=title_pad, fontsize=fontsize['title'])

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize['xlabel'])

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize['ylabel'])

#endregion
