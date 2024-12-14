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

def get_resolution(level):
    return healpix.nside_to_pixel_resolution(2**level)

def get_skycoord(level, pix, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)
    return hp.healpix_to_skycoord(pix)

def get_boundaries(level, pix, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)
    return hp.boundaries_skycoord(pix, 1)[0]

def get_pixels(level, frame='icrs'):
    (skycoords, pix_resolution) = get_pixels_skycoord(level, frame)

    pixs = np.arange(len(skycoords))
    pixels = Table([pixs, skycoords.ra.degree, skycoords.dec.degree], names=['pix', 'ra', 'dec'])

    return (pixels, pix_resolution)

def get_pixels_skycoord(level, frame='icrs'):
    hp = HEALPix(nside=2**level, order='nested', frame=frame)

    pixs = np.arange(hp.npix)
    skycoords = hp.healpix_to_skycoord(pixs)

    pix_resolution = hp.pixel_resolution

    return (skycoords, pix_resolution)

def get_subpixels(outer_level, outer_pix, inner_level, frame='icrs', sort=False):
    (pixs, skycoords, subpix_resolution) = get_subpixels_skycoord(outer_level, outer_pix, inner_level, frame=frame, sort=sort)

    subpixels = Table([pixs, skycoords.ra.degree, skycoords.dec.degree], names=['pix', 'ra', 'dec'])

    return (subpixels, subpix_resolution)

def get_subpixels_skycoord(outer_level, outer_pix, inner_level, frame='icrs', sort=False):
    pixs = _get_subpixels_within_outer_pixel(outer_level, outer_pix, inner_level, sort=sort)

    hp = HEALPix(nside=2**inner_level, order='nested', frame=frame)
    skycoords = hp.healpix_to_skycoord(pixs)

    subpix_resolution = get_resolution(inner_level)

    return (pixs, skycoords, subpix_resolution)

def _get_subpixels_within_outer_pixel(outer_level, outer_pix, inner_level, sort=False):
    if inner_level <= outer_level:
        raise HealpixException("Inner Level must be larger than outer level")

    factor = 2**(inner_level - outer_level)
    inner_pixels_per_outer_pixel = factor**2
    start_pixel = outer_pix * inner_pixels_per_outer_pixel

    subpixels = np.arange(start = start_pixel, stop = start_pixel + inner_pixels_per_outer_pixel)

    if sort:
        return np.sort(subpixels)

    return subpixels
