#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
from astropy.table import Table
import astropy_healpix as healpix
from astropy_healpix import HEALPix

class HealpixException(Exception):
    pass

def get_npix(level):
    return healpix.nside_to_npix(2**level)

def get_area(level):
    return healpix.nside_to_pixel_area(2**level)

def get_resolution(level):
    return healpix.nside_to_pixel_resolution(2**level)

def get_healpix(level, coords, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)
    return hp.skycoord_to_healpix(coords)

def get_boundaries(level, pix, step=1, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)
    boundaries = hp.boundaries_skycoord(pix, step)
    if np.size(pix) == 1:
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
    hp = HEALPix(nside=2**level, order='nested', frame=frame)
    if pixs is None:
        pixs = get_pixels(level, coords, frame=frame)
    return hp.healpix_to_skycoord(pixs)

def get_pixel_details(level, coords=(), frame='icrs'):
    pixs = get_pixels(level, coords, frame)
    skycoords = get_pixel_skycoord(level, pixs=pixs, frame=frame)
    return Table([pixs, skycoords.ra.degree, skycoords.dec.degree], names=['pix', 'ra', 'dec'])

def get_parent_pix(level, pix, outer_level):
    if outer_level > level:
        raise HealpixException("Outer Level must be smaller than inner level")
    return pix // 4**(level - outer_level)

def get_subpixel_npix(outer_level, inner_level):
    return 4**(inner_level - outer_level)

def _get_subpixels_min_max(outer_level, outer_pix, inner_level):
    inner_pixels_per_outer_pixel = get_subpixel_npix(outer_level, inner_level)
    min_pix = outer_pix * inner_pixels_per_outer_pixel
    max_pix = min_pix + inner_pixels_per_outer_pixel
    return (min_pix, max_pix)

def get_subpixels(outer_level, outer_pix, inner_level):
    if inner_level <= outer_level:
        raise HealpixException("Inner level must be larger than outer level")

    if np.size(outer_pix) == 1:
        (start_pixel, stop_pixel) = _get_subpixels_min_max(outer_level, outer_pix, inner_level)
        return np.arange(start = start_pixel, stop = stop_pixel)

    pixs = np.array([])
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
