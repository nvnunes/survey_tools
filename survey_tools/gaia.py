#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import os
from contextlib import redirect_stderr, redirect_stdout
import numpy as np
import astropy.units as u
from astroquery.gaia import Gaia, GaiaClass
import requests
from survey_tools import healpix

class GaiaException(Exception):
    pass

def _get_bounding_box(min_ra, max_ra, min_dec, max_dec):
    ra = (min_ra + max_ra)/2
    dec = (min_dec + max_dec)/2
    width = abs(min_ra - max_ra)
    height = abs(min_dec - max_dec)

    if ra < 0:
        ra += 360

    return ra, dec, width, height

def _get_healpix_circle(level, pix):
    coord = healpix.get_pixel_skycoord(level, pix)
    radius = healpix.get_resolution(level)/2 # resolution is a width so like diameter
    radius_multiple = 4 # to ensure we get all stars in the pixel

    ra = coord.ra.degree
    dec = coord.dec.degree
    radius = radius_multiple*radius.to(u.degree).value

    return ra, dec, radius

def _get_region_clause(ra, dec, radius=None, box=()):
    if len(box) > 0:
        # BOX is deprecated so when the above doesn't work, try the following instead:
        # POLYGON('ICRS', {min_ra},{min_dec},{max_ra},{min_dec},{max_ra},{max_dec},{min_ra},{max_dec})
        width = box[0]
        height = box[1]
        return f"BOX('ICRS',{ra},{dec},{width},{height})"

    if radius is None:
        raise ValueError("Either radius or box must be provided")

    return f"CIRCLE('ICRS',{ra},{dec},{radius})"

def _get_level_clause(level, pix):
    if level is None or pix is None:
        return ''

    return f"AND gaia_healpix_index({level}, SOURCE_ID) = {pix}"

def _count_stars(ra, dec, radius=None, box=(), level=None, pix=None, verbose=False):
    region_clause = _get_region_clause(ra, dec, radius=radius, box=box)
    level_clause = _get_level_clause(level, pix)

    query = f"""
SELECT COUNT(*)
  FROM gaiadr3.gaia_source
 WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), {region_clause})
       {level_clause}
   AND in_qso_candidates = '0'
   AND in_galaxy_candidates = '0'
    """

    try:
        if verbose:
            job = GaiaClass(verbose=True).launch_job_async(query, verbose=True)
        else:
            with open(os.devnull, 'w') as fnull: # pylint: disable=unspecified-encoding
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    job = Gaia.launch_job_async(query)
    except requests.exceptions.HTTPError as e: # pylint: disable=no-member
        if str(e) == 'OK':
            raise GaiaException('Archive is likely down for maintenance') from e
        raise e

    results = job.get_results()

    return results['COUNT_ALL'].filled().value[0]

def _get_stars(ra, dec, radius=None, box=(), level=None, pix=None, verbose=False):
    region_clause = _get_region_clause(ra, dec, radius=radius, box=box)
    level_clause = _get_level_clause(level, pix)

    query = f"""
SELECT SOURCE_ID AS gaia_id
     , ra AS gaia_ra
     , dec AS gaia_dec
     , phot_g_mean_mag AS gaia_G
     , phot_bp_mean_mag AS gaia_BP
     , phot_rp_mean_mag AS gaia_RP
     , ref_epoch AS gaia_ref_epoch
     , pmra AS gaia_pmra
     , pmdec AS gaia_pmdec
     , non_single_star AS gaia_non_single_star
  FROM gaiadr3.gaia_source
 WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), {region_clause})
       {level_clause}
   AND in_qso_candidates = '0'
   AND in_galaxy_candidates = '0'
 ORDER BY SOURCE_ID
    """

    try:
        if verbose:
            job = GaiaClass(verbose=True).launch_job_async(query, verbose=True)
        else:
            with open(os.devnull, 'w') as fnull: # pylint: disable=unspecified-encoding
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    job = Gaia.launch_job_async(query)
    except requests.exceptions.HTTPError as e: # pylint: disable=no-member
        if str(e) == 'OK':
            raise GaiaException('Archive is likely down for maintenance') from e
        raise e

    results = job.get_results()

    if len(results) > 0:
        # see: https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T9
        mag_diff = results['gaia_BP'] - results['gaia_RP']
        G_minus_R_mag = (-0.02275) + (0.3961)*mag_diff + (-0.1243)*np.power(mag_diff, 2) + (-0.01396)*np.power(mag_diff, 3) + (0.003775)*np.power(mag_diff, 4)
        results['gaia_R'] = results['gaia_G'] - G_minus_R_mag

    return results

def count_stars_by_ra_dec_distance(ra, dec, radius, verbose=False):
    return _count_stars(ra, dec, radius=radius, verbose=verbose)

def get_stars_by_ra_dec_distance(ra, dec, radius, verbose=False):
    return _get_stars(ra, dec, radius=radius, verbose=verbose)

def count_stars_by_ra_dec_range(min_ra, max_ra, min_dec, max_dec, verbose=False):
    ra, dec, width, height = _get_bounding_box(min_ra, max_ra, min_dec, max_dec)
    return _count_stars(ra, dec, box=(width, height), verbose=verbose)

def get_stars_by_ra_dec_range(min_ra, max_ra, min_dec, max_dec, verbose=False):
    ra, dec, width, height = _get_bounding_box(min_ra, max_ra, min_dec, max_dec)
    return _get_stars(ra, dec, box=(width, height), verbose=verbose)

def count_stars_by_healpix(level, pix, verbose=False):
    # Querying Gaia is MUCH faster when using RA/Dec query conditions as well
    ra, dec, radius = _get_healpix_circle(level, pix)
    return _count_stars(ra, dec, radius=radius, level=level, pix=pix, verbose=verbose)

def get_stars_by_healpix(level, pix, verbose=False):
    # Querying Gaia is MUCH faster when using RA/Dec query conditions as well
    ra, dec, radius = _get_healpix_circle(level, pix)
    return _get_stars(ra, dec, radius=radius, level=level, pix=pix, verbose=verbose)

def apply_proper_motion(stars, dT=None, epoch=None):
    if dT is None and epoch is None:
        raise GaiaException('Either dT or epoch must be provided')

    if dT is None:
        dT = epoch - stars['gaia_ref_epoch']

    pm_ra  = stars['gaia_pmra'] / np.cos(stars['gaia_dec']/180.0*np.pi) # mas[ra]/year
    pm_dec = stars['gaia_pmdec']                                        # mas/year

    if np.ma.is_masked(pm_ra):
        pm_ra.fill_value = 0.0
        pm_ra = pm_ra.filled()

    if np.ma.is_masked(pm_dec):
        pm_dec.fill_value = 0.0
        pm_dec = pm_dec.filled()

    epoch_ra  = stars['gaia_ra']  + dT * pm_ra  / 1000 / 3600
    epoch_dec = stars['gaia_dec'] + dT * pm_dec / 1000 / 3600

    return (epoch_ra, epoch_dec)
