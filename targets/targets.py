#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from astropy.coordinates import SkyCoord
from astropy import units as u
import survey_tools.gaia as gaia

class StructType:
    pass

SEARCH_DISTANCE_FACTOR = 4.0

def _query_stars(ra, dec, min_mag=-2.0, max_mag=25.0, max_sep=60.0, wfs_band='R', epoch=None):
    stars = gaia.get_stars_by_ra_dec_distance(ra, dec, max_sep/3600*SEARCH_DISTANCE_FACTOR)

    if len(stars) == 0:
        return stars

    if epoch is not None:
        (stars['obs_ra'], stars['obs_dec']) = gaia.apply_proper_motion(stars, epoch=epoch)
        star_coords = SkyCoord(ra=stars['obs_ra'], dec=stars['obs_dec'], unit=(u.degree, u.degree))
    else:
        star_coords = SkyCoord(ra=stars['gaia_ra'], dec=stars['gaia_dec'], unit=(u.degree, u.degree))

    target_coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))
    stars['sep'] = target_coords.separation(star_coords).arcsec

    mag_field = f"gaia_{wfs_band}"
    mag_filter = (stars[mag_field] >= min_mag) & (stars[mag_field] < max_mag)
    sep_filter = stars['sep'] <= max_sep

    return stars[mag_filter & sep_filter].copy()

def count_nearby_stars(ra, dec, min_mag=-2.0, max_mag=25.0, max_sep=60.0, wfs_band='R', epoch=None):
    stars = _query_stars(ra, dec, min_mag=min_mag, max_mag=max_mag, max_sep=max_sep, wfs_band=wfs_band, epoch=epoch)
    return len(stars)

def find_nearby_stars(ra, dec, min_mag=-2.0, max_mag=25.0, max_sep=60.0, wfs_band='R', epoch=None):
    stars = _query_stars(ra, dec, min_mag=min_mag, max_mag=max_mag, max_sep=max_sep, wfs_band=wfs_band, epoch=epoch)
    if len(stars) > 0:
        stars.sort('sep')
    return stars
