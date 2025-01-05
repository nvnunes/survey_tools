#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import copy
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors
from matplotlib.patches import Polygon
import matplotlib.projections
from matplotlib.ticker import Formatter, MultipleLocator
import numpy as np
from astropy.table import Table
from astropy.coordinates import Longitude, Latitude
import astropy.units as u
import astropy_healpix as healpix
from astropy_healpix import HEALPix
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

def get_healpix_from_skycoord(level, coords, frame='icrs'):
    hp = HEALPix(nside=_get_nside(level), order='nested', frame=frame)
    return hp.skycoord_to_healpix(coords)

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

def plot(values, level=None, pixs=None, plot_properties=None):
    _set_default_plot_properties(values, plot_properties)

    fig = plt.figure(figsize=(plot_properties['width'], plot_properties['height']))
    fig.add_subplot(1, 1, 1, projection=(None if plot_properties.get('projection') == 'cartesian' else plot_properties['projection']))

    data_plot = _plot_data(values, level, pixs, **plot_properties)
    _plot_grid(**plot_properties)
    _plot_cbar(data_plot, values, **plot_properties)
    _plot_boundaries(**plot_properties)
    _finish_plot(**plot_properties)

    return fig

GEOGRAPHIC_PROJECTIONS = ['aitoff', 'hammer', 'lambert', 'mollweide']
DEFAULT_LONGITUDE_RANGE = [-180, 180]
DEFAULT_LATITUDE_RANGE = [-90, 90]

def _update_dictionary(dict1, dict2):
    for key in dict1.keys():
        if key in dict2:
            dict1[key] = dict2[key]
    return dict1

def _set_default_plot_properties(values, plot_properties=None):
    """Plot Parameters:
       ---------------
    coords : sequence of character, optional
      Either one of 'G', 'E' or 'C' to describe the coordinate system of the
      map, or a sequence of 2 of these to rotate the map from the first to the
      second coordinate system. default: 'G'
    projection :  {'aitoff', 'hammer', 'lambert', 'mollweide', 'cartesian', 'polar'}
      type of the plot
    flip : bool, optional
      Defines the convention of projection:
        False (geo) - default, east towards roght, west towards left
        True (astro) - east towards left, west towards right
    rotation : scalar, optional
      Describe the logitudinal rotation to apply
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
    vmin : float
      The minimum range value
    vmax : float
      The maximum range value
    grid: bool
      show grid lines.
    grid_labels : bool
      show longitude and latitude labels.
    grid_longitude_spacing : float
      set x axis grid spacing in degrees. Default: 60
    grid_latitude_spacing : float
      set y axis grid spacing in degrees. Default: 30
    cbar : bool, optional
      Show the colorbar. Default: True
    cbar_extend : str, optional
      Whether to extend the colorbar to mark where min or max tick is less than
      the min or max of the data. Options are 'min', 'max', 'neither', or 'both'
    cbar_orientation : {'horizontal', 'vertical'}
      color bar orientation
    cbar_ticks : list
      custom ticks on the colorbar
    cbar_format : str, optional
      The format of the scale label. Default: '%g'
    cbar_unit : str, optional
      A text describing the unit of the data. Default: ''
    cbar_show_tickmarkers : bool, optional
      Preserve tickmarkers for the full bar with labels specified by ticks
      default: None
    boundaries_level : int, optional
      The level of the boundaries to show. Default: None
    colors: dict, optional
      Override default colors
    fontname : str, optional
      Set the fontname of the text
    fontsize:  dict, optional
      Override default fontsize of labels
"""
    if plot_properties is None:
        plot_properties = {}

    # Coordinate System
    if plot_properties.get('coords', None) is None:
        plot_properties['coords'] = 'C'

    # Projection
    if plot_properties.get('projection', None) is None:
        plot_properties['projection'] = 'cartesian'

    if plot_properties['projection'] in ['cart', 'equi', 'equirectangular', 'rectangular', 'lacarte', 'platecarree']:
        plot_properties['projection'] = 'cartesian'

    if plot_properties.get('rotation', None) is None:
        plot_properties['rotation'] = 0.0

    if plot_properties.get('flip', None) is None:
        plot_properties['flip'] = False

    # Size
    if plot_properties.get('width', None) is None:
        plot_properties['width'] = 10

    if plot_properties.get('height', None) is None:
        plot_properties['height'] = plot_properties['width']*0.63

    if plot_properties.get('xsize', None) is None:
        plot_properties['xsize'] = plot_properties['width']*100

    # Color Bar
    if plot_properties.get('cbar_orientation', None) is None:
        plot_properties['cbar_orientation'] = 'vertical' if plot_properties['projection'] == 'cartesian' else 'horizontal'

    if plot_properties['projection'] in GEOGRAPHIC_PROJECTIONS:
        if plot_properties['cbar_orientation'] == 'vertical':
            shrink = 0.6
            pad = 0.03
            if plot_properties.get('cbar_ticks', None) is not None:
                lpad = 0
            else:
                lpad = -8
        else: #if plot_properties['cbar_orientation'] == 'horizontal':
            shrink = 0.6
            pad = 0.05
            if plot_properties.get('cbar_ticks', None) is not None:
                lpad = 0
            else:
                lpad = -8
    elif plot_properties['projection'] == 'polar':
        if plot_properties['cbar_orientation'] == 'vertical':
            shrink = 1
            pad = 0.01
            lpad = 0
        else: #if cbar_orientation == 'horizontal':
            shrink = 0.4
            pad = 0.01
            lpad = 0
    else: #if projection == 'cart':
        if plot_properties['cbar_orientation'] == 'vertical':
            shrink = 0.6
            pad = 0.03
            if plot_properties.get('cbar_ticks', None) is not None:
                lpad = 0
            else:
                lpad = -8
        else: #if cbar_orientation == 'horizontal':
            shrink = 0.6
            pad = 0.05
            if plot_properties.get('cbar_ticks', None) is not None:
                lpad = 0
            else:
                lpad = -8
            if plot_properties['xlabel'] is None:
                pad = pad + 0.01

    if plot_properties['cbar_orientation'] == 'vertical' and plot_properties.get('title', None) is not None:
        lpad += 8

    plot_properties['cbar_shrink'] = shrink
    plot_properties['cbar_pad'] = pad
    plot_properties['cbar_label_pad'] = lpad

     # Colors
    color_defaults = {
        'background': 'white',
        'badvalue': 'gray',
        'gridline': (0.9,0.9,0.9) if values is not None else (0.5,0.5,0.5),
        'grid_xtick_label': 'black',
        'grid_ytick_label': 'black',
        'boundaries': 'red'
    }

    if plot_properties['projection'] != 'cartesian':
        color_defaults['grid_xtick_label'] = color_defaults['gridline']

    if 'colors' in plot_properties and plot_properties['colors'] is not None:
        plot_properties['colors'] = _update_dictionary(color_defaults,  plot_properties['colors'])
    else:
        plot_properties['colors'] = color_defaults

   # Fonts
    if 'fontname' not in plot_properties:
        plot_properties['fontname'] = None

    fontsize_defaults = {
        'xlabel': 12,
        'ylabel': 12,
        'title': 14,
        'xtick_label': 12,
        'ytick_label': 12,
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
    elif isinstance(cmap, matplotlib.colors.Colormap):
        cmap0 = cmap
    else:
        cmap0 = plt.get_cmap(matplotlib.rcParams['image.cmap'])

    cmap = copy.copy(cmap0)
    cmap.set_over(plot_properties['colors']['badvalue'])
    cmap.set_under(plot_properties['colors']['badvalue'])
    cmap.set_bad(plot_properties['colors']['background'])

    plot_properties['cmap'] = cmap

    # Normalization
    if values is not None:
        is_valid = ~np.isnan(values) & ~np.isinf(values)

        if plot_properties.get('vmin', None) is None:
            if plot_properties.get('cbar_ticks', None) is not None:
                plot_properties['vmin'] = np.min(plot_properties['cbar_ticks'])
            elif np.sum(is_valid) > 0:
                plot_properties['vmin'] = np.min(values[is_valid])
            elif 'vmin' not in plot_properties:
                plot_properties['vmin'] = None

        if plot_properties.get('vmax', None) is None:
            if plot_properties.get('cbar_ticks', None) is not None:
                plot_properties['vmax'] = np.max(plot_properties['cbar_ticks'])
            elif np.sum(is_valid) > 0:
                plot_properties['vmax'] = np.max(values[is_valid])
            elif 'vmax' not in plot_properties:
                plot_properties['vmax'] = None

        norm = plot_properties.get('norm', None)
        if norm is None:
            norm = 'none'

        if isinstance(norm, str):
            match norm.lower():
                case 'lin' | 'norm' | 'normalize':
                    norm = matplotlib.colors.Normalize()
                case 'log':
                    norm = matplotlib.colors.LogNorm()
                case 'symlog':
                    norm = matplotlib.colors.SymLogNorm(1, linscale=0.1, clip=True, base=10) # pylint: disable=unexpected-keyword-arg
                case _:
                    norm = matplotlib.colors.NoNorm()

        norm.vmin = plot_properties['vmin']
        norm.vmax = plot_properties['vmax']
        norm.autoscale_None(values[is_valid])

        plot_properties['norm'] = norm

    return plot_properties

def _plot_data(values, level, pixs,
    coords=None,
    projection='cartesian',
    rotation=None,
    flip=False,
    xsize=1000,
    ysize=None,
    cmap=None,
    norm=None,
    **kwargs # pylint: disable=unused-argument
):
    if values is not None:
        npix = len(values)

        if level is None:
            if not is_npix_valid(npix):
                raise HealpixException('level and pixs required when passing a partial map')
            level = get_level(npix)

        if pixs is None:
            if npix == get_npix(level):
                pixs = None
            else:
                raise HealpixException('pixs must be specified when values are not a full HEALpix map')
        else:
            if npix != len(pixs):
                raise HealpixException('values and pixs must be the same size')

    if values is not None and pixs is not None and projection not in GEOGRAPHIC_PROJECTIONS:
        if rotation != 0.0:
            raise HealpixException('rotation not supported when pixs specified')

        symmetric = False
        method = 'closest'
        match np.array(coords)[0]:
            case 'C':
                skycoords = get_pixel_skycoord(level, pixs, frame='icrs')
                max_longitude = max(skycoords.ra.degree)
                min_longitude = min(skycoords.ra.degree)
                max_latitude = max(skycoords.dec.degree)
                min_latitude = min(skycoords.dec.degree)
            case 'G':
                skycoords = get_pixel_skycoord(level, pixs, frame='galactic')
                max_longitude = max(skycoords.l.degree)
                min_longitude = min(skycoords.l.degree)
                max_latitude = max(skycoords.b.degree)
                min_latitude = min(skycoords.b.degree)
            case 'E':
                skycoords = get_pixel_skycoord(level, pixs, frame='ecliptic')
                max_longitude = max(skycoords.lon.degree)
                min_longitude = min(skycoords.lon.degree)
                max_latitude = max(skycoords.lat.degree)
                min_latitude = min(skycoords.lat.degree)
    else:
        symmetric = True
        if pixs is not None:
            method = 'closest'
        else:
            method = 'interpolate'
        [min_longitude, max_longitude] = DEFAULT_LONGITUDE_RANGE
        [min_latitude, max_latitude] = DEFAULT_LATITUDE_RANGE

    range_lon = max_longitude - min_longitude # pylint: disable=possibly-used-before-assignment
    range_lat = max_latitude - min_latitude # pylint: disable=possibly-used-before-assignment
    mid_lon = (max_longitude + min_longitude) / 2
    mid_lat = (max_latitude + min_latitude) / 2

    if not symmetric:
        range_lon = min(360.0, range_lon * 1.1)
        range_lat = min(180.0, range_lat * 1.1)

        min_longitude = max(0.0, mid_lon-range_lon/2)
        max_longitude = min(360.0, mid_lon+range_lon/2)
        min_latitude = max(-90.0, mid_lat-range_lat/2)
        max_latitude = min(90.0, mid_lat+range_lat/2)

    if ysize is None:
        ratio = range_lon * np.cos(np.deg2rad(mid_lat)) / range_lat
        ysize = int(xsize / ratio)

    longitude = np.linspace(min_longitude, max_longitude, xsize)
    latitude = np.linspace(min_latitude, max_latitude, ysize)

    if values is not None:
        if (rotation != 0.0) or (np.size(coords) > 1 and coords[0] != coords[1]):
            phi = np.deg2rad(longitude)
            theta = np.deg2rad(90 - latitude)
            PHI, THETA = np.meshgrid(phi, theta)

            rotator = Rotator(coord=coords, rot=-rotation, inv=True)
            THETA, PHI = rotator(THETA.flatten(), PHI.flatten())
            THETA = THETA.reshape(ysize, xsize)
            PHI = PHI.reshape(ysize, xsize)

            LONGITUDE = np.rad2deg(PHI) % 360.0
            LATITUDE = 90 - np.rad2deg(THETA)
        else:
            LONGITUDE, LATITUDE = np.meshgrid(longitude, latitude)

        rotated_longitude = Longitude(LONGITUDE.flatten(), unit=u.degree)
        rotated_latitude = Latitude(LATITUDE.flatten(), unit=u.degree)

        match method:
            case 'interpolate':
                if len(values) != get_npix(level):
                    raise HealpixException('not supported')
                plot_values = healpix.interpolate_bilinear_lonlat(
                    rotated_longitude,
                    rotated_latitude,
                    values,
                    order='nested'
                ).reshape(ysize, xsize)
            case 'closest':
                grid_pix = healpix.lonlat_to_healpix(
                    rotated_longitude,
                    rotated_latitude,
                    nside=_get_nside(level),
                    order='nested'
                ).reshape(ysize, xsize)

                if pixs is None:
                    plot_values = values[grid_pix]
                else:
                    if not np.all(np.diff(pixs) >= 0):
                        sorted_indices = np.argsort(pixs)
                        pixs = pixs[sorted_indices]
                        values = values[sorted_indices]
                    pixs_indices = np.searchsorted(pixs, grid_pix)
                    plot_values = np.full(grid_pix.shape, np.nan)
                    valid_indices = np.isin(grid_pix, pixs)
                    plot_values[valid_indices] = values[pixs_indices[valid_indices]]

    if flip:
        if symmetric:
            longitude *= -1
        else:
            longitude = 360.0 - longitude

    if not symmetric or (values is None and projection not in GEOGRAPHIC_PROJECTIONS):
        ax = plt.gca()
        ax.set_xlim(np.deg2rad(np.array([min(longitude), max(longitude)])))
        ax.set_ylim(np.deg2rad(np.array([min(latitude), max(latitude)])))

    if values is None:
        return None

    return plt.pcolormesh(
        np.deg2rad(longitude),
        np.deg2rad(latitude),
        plot_values,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        shading='auto'
    )

class DegreesFormatter(Formatter):
    def __init__(self, round_to=1.0): # pylint: disable=useless-parent-delegation
        self._round_to = round_to

    def __call__(self, x, pos=None):
        degrees = np.rad2deg(x)
        degrees = round(degrees / self._round_to) * self._round_to

        d = int(degrees)
        m = int(np.round((degrees - d) * 60, 5))
        s = int(np.round((degrees - d - m/60.0) * 3600, 2))

        if m == 0 and s == 0:
            return f"{d:.0f}\N{DEGREE SIGN}"

        if s == 0:
            return f"{d:.0f}\N{DEGREE SIGN}{m:0>2.0f}\N{PRIME}"

        if s % 1 == 0:
            sec_format = '0>2.0f'
        else:
            sec_format = '0>2.1f'

        return f"{d:.0f}\N{DEGREE SIGN}{m:0>2.0f}\N{PRIME}{s:{sec_format}}\N{DOUBLE PRIME}"

class LongitudeFormatter(DegreesFormatter):
    def __init__(self, round_to=1.0): # pylint: disable=useless-parent-delegation
        super().__init__(round_to)

    def __call__(self, x, pos=None): # pylint: disable=useless-parent-delegation
        x = x % (2*np.pi)
        return super().__call__(x, pos)

class RotatedLongitudeFormatter(LongitudeFormatter):
    def __init__(self, round_to=1.0, rotation=0.0, flip=False):
        super().__init__(round_to)
        self._rotation = rotation
        self._flip = flip

    def __call__(self, x, pos=None):
        x = (x + self._rotation)*(-1 if self._flip else 1)
        return super().__call__(x, pos)

class SymmetricLongitudeFormatter(DegreesFormatter):
    def __init__(self, round_to=1.0): # pylint: disable=useless-parent-delegation
        super().__init__(round_to)

    def __call__(self, x, pos=None):
        x = (x + np.pi) % (2*np.pi) - np.pi
        return super().__call__(x, pos)

class RotatedSymmetricLongitudeFormatter(SymmetricLongitudeFormatter):
    def __init__(self, round_to=1.0, rotation=0.0, flip=False):
        super().__init__(round_to)
        self._rotation = rotation
        self._flip = flip

    def __call__(self, x, pos=None):
        x = (x + self._rotation)*(-1 if self._flip else 1)
        return super().__call__(x, pos)

class RightAscensionFormatter(Formatter):
    def __init__(self, round_to=1.0):
        self._round_to = round_to/15 # convert round_to from degrees to hours

    def __call__(self, x, pos=None):
        hours = np.rad2deg(x)/15 + (24 if x < 0 else 0)
        hours = round(hours / self._round_to) * self._round_to

        h = int(hours)
        m = int(np.round((hours - h) * 60, 5))
        s = int(np.round((hours - h - m/60.0) * 3600, 2))

        if m == 0 and s == 0:
            return f"{h:.0f}\u02B0"

        if s == 0:
            return f"{h:.0f}\u02B0{m:0>2.0f}\u1D50"

        if s % 1 == 0:
            sec_format = '0>2.0f'
        else:
            sec_format = '0>2.1f'

        return f"{h:.0f}\u02B0{m:0>2.0f}\u1D50{s:{sec_format}}\u02E2"

class RotatedRightAscensionFormatter(RightAscensionFormatter):
    def __init__(self, round_to=1.0, rotation=0.0, flip=False):
        super().__init__(round_to)
        self._rotation = rotation
        self._flip = flip

    def __call__(self, x, pos=None):
        x = (x + self._rotation)*(-1 if self._flip else 1)
        return super().__call__(x, pos)

def _round_angular_tick_spacing(ticks, is_hours=False):
    tick_spacing = np.diff(ticks)[0]
    tick_range = np.abs(ticks[0]-ticks[-1])

    if is_hours:
        tick_spacing = tick_spacing / 15.0
        tick_range = tick_range / 15.0
        factor = 15.0
    else:
        factor = 1

    if tick_spacing > 1:
        round_to_factor = 1
        if is_hours:
            round_to_options = [6,4,3,2,1]
        else:
            round_to_options = [90,60,45,30,15,10,5,4,3,2,1]
    elif tick_spacing*60 > 1:
        round_to_factor = 60
        round_to_options = [30,20,15,10,5,1]
    else:
        round_to_factor = 3600
        if tick_spacing*3600 > 1:
            round_to_options = [30,20,15,10,5,1]
        else:
            round_to_options = [0.5,0.1]

    round_to_index = np.argmin(np.abs(tick_spacing*round_to_factor - round_to_options))
    round_to = round_to_options[round_to_index] / round_to_factor

    if tick_range/round_to > 5 and round_to_index > 0:
        round_to_index = round_to_index-1
        round_to = round_to_options[round_to_index] / round_to_factor

    if tick_range/round_to < 2 and round_to_index < len(round_to_options)-1:
        round_to_index = round_to_index+1
        round_to = round_to_options[round_to_index] / round_to_factor

    return factor * round_to

def _plot_grid(
        projection='cartesian',
        rotation=0.0,
        flip=False,
        grid=True,
        grid_labels=True,
        grid_longitude_spacing=None,
        grid_latitude_spacing=None,
        grid_longitude_type=None,
        colors=None,
        fontname=None,
        fontsize=None,
        **kwargs # pylint: disable=unused-argument
):
    ax = plt.gca()

    if grid:
        plt.grid(True, color=colors['gridline'])
        ax.set_axisbelow(False) # Fix to force grid lines to be on top of the image

        if grid_longitude_spacing is None:
            grid_longitude_spacing = _round_angular_tick_spacing(np.rad2deg(ax.get_xticks()), is_hours=grid_longitude_type=='hours')

        if grid_latitude_spacing is None:
            grid_latitude_spacing = _round_angular_tick_spacing(np.rad2deg(ax.get_yticks()))

        if projection in GEOGRAPHIC_PROJECTIONS:
            ax.set_longitude_grid(grid_longitude_spacing)
            ax.set_latitude_grid(grid_latitude_spacing)
            ax.set_longitude_grid_ends(90)
        else:
            ax.xaxis.set_major_locator(MultipleLocator(np.deg2rad(grid_longitude_spacing)))
            ax.yaxis.set_major_locator(MultipleLocator(np.deg2rad(grid_latitude_spacing)))

    if grid and grid_labels:
        match grid_longitude_type:
            case 'hours':
                ax.xaxis.set_major_formatter(RotatedRightAscensionFormatter(grid_longitude_spacing, rotation=np.deg2rad(rotation), flip=flip))
            case 'degrees':
                ax.xaxis.set_major_formatter(RotatedLongitudeFormatter(grid_longitude_spacing, rotation=np.deg2rad(rotation), flip=flip))
            case _:
                ax.xaxis.set_major_formatter(RotatedSymmetricLongitudeFormatter(grid_longitude_spacing, rotation=np.deg2rad(rotation), flip=flip))

        ax.yaxis.set_major_formatter(DegreesFormatter(grid_latitude_spacing))

        ax.tick_params(axis='x', labelfontfamily=fontname, labelsize=fontsize['xtick_label'], colors=colors['grid_xtick_label'])
        ax.tick_params(axis='y', labelfontfamily=fontname, labelsize=fontsize['ytick_label'], colors=colors['grid_ytick_label'])
    else:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

def _convert_to_180_range(angles):
    return (angles + 180.0) % 360.0 - 180.0

def _plot_boundaries(
        coords='C',
        rotation=0.0,
        flip=False,
        boundaries_level=None,
        boundaries_pixs=None,
        colors=None,
        fontname=None,
        fontsize=None,
        **kwargs # pylint: disable=unused-argument
):
    if boundaries_level is None:
        return

    if boundaries_pixs is not None and not isinstance(boundaries_pixs, list) and not isinstance(boundaries_pixs, np.ndarray):
        boundaries_pixs = [boundaries_pixs]

    ax = plt.gca()
    xlim = np.sort(np.rad2deg(ax.get_xlim()))
    ylim = np.sort(np.rad2deg(ax.get_ylim()))
    xrange = np.abs(xlim[0]-xlim[1])
    yrange = np.abs(ylim[0]-ylim[1])
    full_range = xrange >= 360.0 and yrange >= 180.0
    symmetric = full_range

    step = max(1,2**(7-boundaries_level)) # level 0 = 128, level 1 = 64, level 2 = 32, ...
    pixs = np.arange(get_npix(boundaries_level))
    skycoords = get_pixel_skycoord(boundaries_level, pixs)
    boundaries = get_boundaries_skycoord(boundaries_level, pixs=pixs, step=step)

    if coords == 'G' or coords[1] == 'G':
        boundaries_lon = boundaries.galactic.l.to(u.degree).value + rotation
        boundaries_lat = boundaries.galactic.b.to(u.degree).value
        center_lon = skycoords.galactic.l.to(u.degree).value + rotation
        center_lat = skycoords.galactic.b.to(u.degree).value
    elif coords == 'E' or coords[1] == 'E':
        boundaries_lon = boundaries.ecliptic.lon.to(u.degree).value + rotation
        boundaries_lat = boundaries.ecliptic.lat.to(u.degree).value
        center_lon = skycoords.ecliptic.lon.to(u.degree).value + rotation
        center_lat = skycoords.ecliptic.lat.to(u.degree).value
    else:
        boundaries_lon = boundaries.ra.to(u.degree).value + rotation
        boundaries_lat = boundaries.dec.to(u.degree).value
        center_lon = skycoords.ra.to(u.degree).value + rotation
        center_lat = skycoords.dec.to(u.degree).value

    if symmetric:
        boundaries_lon = _convert_to_180_range(boundaries_lon)
        center_lon = _convert_to_180_range(center_lon)
    else:
        boundaries_lon = boundaries_lon % 360.0
        center_lon = center_lon % 360.0

    count = 0
    max_count = 250

    for i, pix in enumerate(pixs):
        vertices = np.vstack([boundaries_lon[i], boundaries_lat[i]]).transpose()

        if symmetric and min(vertices[:,0]) == -180 and max(vertices[:,0]) > 0: # fix unnecessary wrap around
            vertices[vertices[:,0] == -180,0] = 180

        if flip:
            if symmetric:
                vertices[:,0] *= -1
                center_lon[i] *= -1
            else:
                vertices[:,0] = 360.0 - vertices[:,0]
                center_lon[i] = 360.0 - center_lon[i]

        if not full_range and not np.any((vertices[:, 0] >= xlim[0]) & (vertices[:, 0] <= xlim[1]) & (vertices[:, 1] >= ylim[0]) & (vertices[:, 1] <= ylim[1])):
            continue

        count += 1
        if count > max_count:
            plt.clf()
            raise HealpixException('Too many boundaries to plot')

        if max(vertices[:,0]) - min(vertices[:,0]) < 0.95*xrange: # only draw boundaries that do not wrap around
            ax.add_patch(Polygon(np.deg2rad(vertices), closed=True, edgecolor=colors['boundaries'], facecolor='none', lw=0.5, zorder=10))

        if full_range or np.all((center_lon[i] >= xlim[0]) & (center_lon[i] <= xlim[1]) & (center_lat[i] >= ylim[0]) & (center_lat[i] <= ylim[1])):
            if boundaries_pixs is None or pix in boundaries_pixs:
                boundaries_label_fontsize = fontsize['boundaries_label_small'] if full_range else fontsize['boundaries_label']
                ax.text(np.deg2rad(center_lon[i]), np.deg2rad(center_lat[i]), f"{pix}", color=colors['boundaries'], fontsize=boundaries_label_fontsize, fontname=fontname, ha='center', va='center')

def _plot_cbar(data_plot, values,
        vmin=None,
        vmax=None,
        cbar=True,
        cbar_orientation='horizontal',
        cbar_tick_direction='out',
        cbar_ticks=None,
        cbar_extend=None,
        cbar_format='%g',
        cbar_label_pad=0.0,
        cbar_pad=0.0,
        cbar_show_tickmarkers=False,
        cbar_shrink=1.0,
        cbar_unit='',
        cbar_vertical_tick_rotation=90,
        fontname=None,
        fontsize=None,
        **kwargs # pylint: disable=unused-argument
    ):

    if not cbar or vmin is None or vmax is None:
        return

    fig = plt.gcf()

    if cbar_ticks is None:
        cbar_ticks = [vmin, vmax]

    if cbar_extend is None:
        cbar_extend = 'neither'
        if vmin > np.min(values):
            cbar_extend = 'min'
        if vmax < np.max(values):
            cbar_extend = 'max'
        if vmin > np.min(values) and vmax < np.max(values):
            cbar_extend = 'both'

    cb = fig.colorbar(
        data_plot,
        orientation=cbar_orientation,
        shrink=cbar_shrink,
        pad=cbar_pad,
        ticks=(None if cbar_show_tickmarkers else cbar_ticks),
        extend=cbar_extend,
    )

    # Hide all tickslabels not in tick variable. Do not delete tick-markers
    if cbar_show_tickmarkers:
        ticks = list(set(cb.get_ticks()) | set(cbar_ticks))
        ticks = np.sort(ticks)
        ticks = ticks[ticks >= vmin]
        ticks = ticks[ticks <= vmax]
        labels = [cbar_format % tick if tick in cbar_ticks else '' for tick in ticks]

        try:
            cb.set_ticks(ticks, labels)
        except TypeError:
            cb.set_ticks(ticks)
            cb.set_ticklabels(labels)
    else:
        labels = [cbar_format % tick for tick in cbar_ticks]

    if cbar_orientation == 'horizontal':
        cb.ax.set_xticklabels(labels, fontname=fontname)

        cb.ax.xaxis.set_label_text(
            cbar_unit, fontsize=fontsize['cbar_label'], fontname=fontname
        )
        cb.ax.tick_params(
            axis='x',
            labelsize=fontsize['cbar_tick_label'],
            direction=cbar_tick_direction
        )
        cb.ax.xaxis.labelpad = cbar_label_pad
    if cbar_orientation == 'vertical':
        cb.ax.set_yticklabels(
            labels,
            rotation=cbar_vertical_tick_rotation,
            va='center',
            fontname=fontname
        )

        cb.ax.yaxis.set_label_text(
            cbar_unit,
            fontsize=fontsize['cbar_label'],
            rotation=90,
            fontname=fontname
        )
        cb.ax.tick_params(
            axis='y',
            labelsize=fontsize['cbar_tick_label'],
            direction=cbar_tick_direction
        )
        cb.ax.yaxis.labelpad = cbar_label_pad

    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor('face')

def _finish_plot(
    grid=False,
    grid_labels=False,
    title=None,
    xlabel=None,
    ylabel=None,
    fontname=None,
    fontsize=None,
    **kwargs # pylint: disable=unused-argument
):
    left = 0.02
    right = 0.98
    top = 0.95
    bottom = 0.05

    if grid and grid_labels:
        left += 0.02

    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    ax = plt.gca()

    if title is not None:
        ax.set_title(title, fontsize=fontsize['title'], fontname=fontname)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize['xlabel'], fontname=fontname)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize['ylabel'], fontname=fontname)

#endregion
