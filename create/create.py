#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import astropy.units as u
from survey_tools import catalog, match

def load_3DHST(field):
    catalog_params_3DHST = catalog.get_params('3D-HST', field, 'F160W')
    catalog_3DHST = catalog.read(catalog_params_3DHST, force_tables=True)
    catalog_region_3DHST = catalog.get_catalog_region(catalog_3DHST, added_distance=0.5*u.degree)
    return (catalog_params_3DHST, catalog_3DHST, catalog_region_3DHST)

def load_UVISTA():
    catalog_params_UVISTA = catalog.get_params('UVISTA', 'COSMOS', 'K')
    catalog_UVISTA = catalog.read(catalog_params_UVISTA)
    catalog_region_UVISTA = catalog.get_catalog_region(catalog_UVISTA)
    return (catalog_params_UVISTA, catalog_UVISTA, catalog_region_UVISTA)

def cross_match_3DHST(catalog_master, field):
    catalog_params_3DHST = catalog.get_params('3D-HST', field, 'F160W')
    catalog_3DHST = catalog.read(catalog_params_3DHST)
    matches_3DHST = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_3DHST)
    match.append_cross_matched_data(matches_3DHST, catalog_master, catalog_3DHST)
    return (catalog_3DHST, matches_3DHST)

def cross_match_ZCB(catalog_master):
    catalog_params_ZCB = catalog.get_params('ZCOSMOS-BRIGHT')
    catalog_ZCB = catalog.read(catalog_params_ZCB)
    if catalog_master.catalog == 'UVISTA':
        matches_ZCB = match.cross_match_catalogs_by_id_ra_dec(catalog_master, catalog_ZCB, id_field1='z_spec_id')
    else:
        matches_ZCB = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_ZCB)
    match.append_cross_matched_data(matches_ZCB, catalog_master, catalog_ZCB)
    return (catalog_ZCB, matches_ZCB)

def cross_match_ZCD(catalog_master):
    catalog_params_ZCD = catalog.get_params('ZCOSMOS-DEEP')
    catalog_ZCD = catalog.read(catalog_params_ZCD)
    matches_ZCD = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_ZCD)
    match.append_cross_matched_data(matches_ZCD, catalog_master, catalog_ZCD)
    return (catalog_ZCD, matches_ZCD)

def cross_match_VUDS(catalog_master, field):
    catalog_params_VUDS = catalog.get_params('VUDS', field)
    catalog_VUDS = catalog.read(catalog_params_VUDS)
    matches_VUDS = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_VUDS)
    match.append_cross_matched_data(matches_VUDS, catalog_master, catalog_VUDS)
    return (catalog_VUDS, matches_VUDS)

def cross_match_Casey(catalog_master, field):
    catalog_params_Casey = catalog.get_params('Casey', field)
    catalog_Casey = catalog.read(catalog_params_Casey)
    matches_Casey = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_Casey)
    match.append_cross_matched_data(matches_Casey, catalog_master, catalog_Casey)
    return (catalog_Casey, matches_Casey)

def cross_match_DEMOS(catalog_master):
    catalog_params_DEIMOS = catalog.get_params('DEIMOS')
    catalog_DEIMOS = catalog.read(catalog_params_DEIMOS)
    matches_DEIMOS = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_DEIMOS)
    match.append_cross_matched_data(matches_DEIMOS, catalog_master, catalog_DEIMOS)
    return (catalog_DEIMOS, matches_DEIMOS)

def cross_match_MOSDEF(catalog_master, field):
    catalog_params_MOSDEF = catalog.get_params('MOSDEF', field)
    catalog_MOSDEF = catalog.read(catalog_params_MOSDEF)
    if catalog_master.catalog == '3DHST':
        matches_MOSDEF = match.cross_match_catalogs_by_id(catalog_master, catalog_MOSDEF, id_field2='ID_V4')
    else:
        matches_MOSDEF = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_MOSDEF)
    match.append_cross_matched_data(matches_MOSDEF, catalog_master, catalog_MOSDEF)
    return (catalog_MOSDEF, matches_MOSDEF)

def cross_match_FMOS(catalog_master):
    catalog_params_FMOS = catalog.get_params('FMOS')
    catalog_FMOS = catalog.read(catalog_params_FMOS)
    matches_FMOS = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_FMOS)
    match.append_cross_matched_data(matches_FMOS, catalog_master, catalog_FMOS)
    return (catalog_FMOS, matches_FMOS)

def cross_match_KMOS3D(catalog_master, field):
    catalog_params_KMOS3D = catalog.get_params('KMOS3D', field)
    catalog_KMOS3D = catalog.read(catalog_params_KMOS3D)
    if catalog_master.catalog == '3DHST':
        matches_KMOS3D = match.cross_match_catalogs_by_id(catalog_master, catalog_KMOS3D)
    else:
        matches_KMOS3D = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_KMOS3D)

    match.append_cross_matched_data(matches_KMOS3D, catalog_master, catalog_KMOS3D)
    return (catalog_KMOS3D, matches_KMOS3D)

def cross_match_C3R2(catalog_master, field):
    catalog_params_C3R2 = catalog.get_params('C3R2', field)
    catalog_C3R2 = catalog.read(catalog_params_C3R2)
    matches_C3R2 = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_C3R2)
    match.append_cross_matched_data(matches_C3R2, catalog_master, catalog_C3R2)
    return (catalog_C3R2, matches_C3R2)

def cross_match_LEGAC(catalog_master):
    catalog_params_LEGAC = catalog.get_params('LEGAC')
    catalog_LEGAC = catalog.read(catalog_params_LEGAC)
    if catalog_master.catalog == 'UVISTA':
        matches_LEGAC = match.cross_match_catalogs_by_id_ra_dec(catalog_master, catalog_LEGAC, id_field2='ID')
    else:
        matches_LEGAC = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_LEGAC)
    match.append_cross_matched_data(matches_LEGAC, catalog_master, catalog_LEGAC)
    return (catalog_LEGAC, matches_LEGAC)

def cross_match_HSCSSP(catalog_master, field):
    catalog_params_HSCSSP = catalog.get_params('HSCSSP', field)
    catalog_HSCSSP = catalog.read(catalog_params_HSCSSP)
    matches_HSCSSP = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_HSCSSP)
    match.append_cross_matched_data(matches_HSCSSP, catalog_master, catalog_HSCSSP)
    return (catalog_HSCSSP, matches_HSCSSP)

def cross_match_DESI(catalog_master, field, region, region_wcs):
    catalog_params_DESI = catalog.get_params('DESI', field)
    catalog_DESI = catalog.read(catalog_params_DESI, region=region, region_wcs=region_wcs)
    matches_DESI = match.cross_match_catalogs_by_ra_dec(catalog_master, catalog_DESI)
    match.append_cross_matched_data(matches_DESI, catalog_master, catalog_DESI)
    return (catalog_DESI, matches_DESI)

def consolidate_3DHST(catalog_3DHST):
    catalog.consolidate_best_data(catalog_3DHST)

def consolidate_UVISTA(catalog_UVISTA):
    catalog.consolidate_best_data(catalog_UVISTA)

def save_3DHSTPLUS(catalog_3DHST, save_format):
    catalog_params_3DHSTPLUS = catalog.get_params('3D-HST-PLUS')
    catalog.save_matched_catalog(catalog_3DHST, catalog_params_3DHSTPLUS, save_format=save_format, new_name='3D-HST-PLUS')

def save_UVISTAPLUS(catalog_UVISTA, save_format):
    catalog_params_UVISTAPLUS = catalog.get_params('UVISTA-PLUS')
    catalog.save_matched_catalog(catalog_UVISTA, catalog_params_UVISTAPLUS, save_format=save_format, new_name='UVISTA-PLUS')
