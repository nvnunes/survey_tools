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

def get_skycoord(level, pix, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)
    return hp.healpix_to_skycoord(pix)

def get_boundaries(level, pix, step=1, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)
    boundaries = hp.boundaries_skycoord(pix, step)
    if len(pix) == 1:
        return boundaries[0]
    else:
        return boundaries

def get_pixels(level, frame='icrs'):
    skycoords = get_pixels_skycoord(level, frame)

    pixs = np.arange(len(skycoords))
    pixels = Table([pixs, skycoords.ra.degree, skycoords.dec.degree], names=['pix', 'ra', 'dec'])

    return pixels

def get_pixels_skycoord(level, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)

    pixs = np.arange(hp.npix)
    skycoords = hp.healpix_to_skycoord(pixs)

    return skycoords

def get_subpixels_min_max(outer_level, outer_pix, inner_level):
    inner_pixels_per_outer_pixel = 4**(inner_level - outer_level)
    min_pix = outer_pix * inner_pixels_per_outer_pixel
    max_pix = min_pix + inner_pixels_per_outer_pixel
    return (min_pix, max_pix)

def get_subpixels(outer_level, outer_pix, inner_level, frame='icrs'):
    (pixs, skycoords, subpix_resolution) = get_subpixels_skycoord(outer_level, outer_pix, inner_level, frame=frame)

    subpixels = Table([pixs, skycoords.ra.degree, skycoords.dec.degree], names=['pix', 'ra', 'dec'])

    return (subpixels, subpix_resolution)

def get_subpixels_skycoord(outer_level, outer_pix, inner_level, frame='icrs'):
    pixs = _get_subpixels_within_outer_pixel(outer_level, outer_pix, inner_level)

    hp = HEALPix(nside=2**inner_level, order='nested', frame=frame)
    skycoords = hp.healpix_to_skycoord(pixs)

    subpix_resolution = get_resolution(inner_level)

    return (pixs, skycoords, subpix_resolution)

def _get_subpixels_within_outer_pixel(outer_level, outer_pix, inner_level):
    if inner_level <= outer_level:
        raise HealpixException("Inner Level must be larger than outer level")

    (start_pixel, stop_pixel) = get_subpixels_min_max(outer_level, outer_pix, inner_level)

    subpixels = np.arange(start = start_pixel, stop = stop_pixel)

    return subpixels
