#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
import astropy.units as u
from survey_tools import catalog, match

save_ascii = True

fields = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S', 'UDS']
for i in np.arange(len(fields)):
    field = fields[i]

    # Load 3DHST
    print(f"Loading 3D-HST catalog {field} field...", flush=True)
    catalog_params_3DHST = catalog.get_params('3D-HST', field, 'F160W')
    catalog_3DHST = catalog.read(catalog_params_3DHST, force_tables=True)
    catalog_region_3DHST = catalog.get_catalog_region(catalog_3DHST, added_distance=0.5*u.degree)

    # Match zCOSMOS-Bright
    if field == 'COSMOS':
        print('  Matching zCOSMOS-Bright...', flush=True)
        catalog_params_ZCB = catalog.get_params('ZCOSMOS-BRIGHT')
        catalog_ZCB = catalog.read(catalog_params_ZCB)
        matches_ZCB = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_ZCB)
        match.append_cross_matched_data(matches_ZCB, catalog_3DHST, catalog_ZCB)
        catalog.close(catalog_ZCB)
        print(f"    Matched = {matches_ZCB.count} sources", flush=True)
        del catalog_params_ZCB, catalog_ZCB, matches_ZCB

    # Match zCOSMOS-Deep
    if field == 'COSMOS':
        print('  Matching zCOSMOS-Deep...', flush=True)
        catalog_params_ZCD = catalog.get_params('ZCOSMOS-DEEP')
        catalog_ZCD = catalog.read(catalog_params_ZCD)
        matches_ZCD = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_ZCD)
        match.append_cross_matched_data(matches_ZCD, catalog_3DHST, catalog_ZCD)
        catalog.close(catalog_ZCD)
        print(f"    Matched = {matches_ZCD.count} sources", flush=True)
        del catalog_params_ZCD, catalog_ZCD, matches_ZCD

    # Match VUDS
    if field == 'COSMOS' or field == 'GOODS-S':
        print('  Matching VUDS...', flush=True)
        catalog_params_VUDS = catalog.get_params('VUDS', field)
        catalog_VUDS = catalog.read(catalog_params_VUDS)
        matches_VUDS = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_VUDS)
        match.append_cross_matched_data(matches_VUDS, catalog_3DHST, catalog_VUDS)
        catalog.close(catalog_VUDS)
        print(f"    Matched = {matches_VUDS.count} sources", flush=True)
        del catalog_params_VUDS, catalog_VUDS, matches_VUDS

    # Match Casey DSFG
    if field == 'COSMOS' or field == 'GOODS-N' or field == 'UDS':
        print('  Matching Casey DSFG...', flush=True)
        catalog_params_Casey = catalog.get_params('Casey', field)
        catalog_Casey = catalog.read(catalog_params_Casey)
        matches_Casey = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_Casey)
        match.append_cross_matched_data(matches_Casey, catalog_3DHST, catalog_Casey)
        catalog.close(catalog_Casey)
        print(f"    Matched = {matches_Casey.count} sources", flush=True)
        del catalog_params_Casey, catalog_Casey, matches_Casey

    # Match DEIMOS 10K
    if field == 'COSMOS':
        print('  Matching DEIMOS 10K...', flush=True)
        catalog_params_DEIMOS = catalog.get_params('DEIMOS')
        catalog_DEIMOS = catalog.read(catalog_params_DEIMOS)
        matches_DEIMOS = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_DEIMOS)
        match.append_cross_matched_data(matches_DEIMOS, catalog_3DHST, catalog_DEIMOS)
        catalog.close(catalog_DEIMOS)
        print(f"    Matched = {matches_DEIMOS.count} sources", flush=True)
        del catalog_params_DEIMOS, catalog_DEIMOS, matches_DEIMOS

    # Match MOSDEF
    if field == 'AEGIS' or field == 'COSMOS' or field == 'GOODS-N' or field == 'GOODS-S' or field == 'UDS':
        print('  Matching MOSDEF...', flush=True)
        catalog_params_MOSDEF = catalog.get_params('MOSDEF', field)
        catalog_MOSDEF = catalog.read(catalog_params_MOSDEF)
        matches_MOSDEF = match.cross_match_catalogs_by_id(catalog_3DHST, catalog_MOSDEF, id_field2='ID_V4')
        match.append_cross_matched_data(matches_MOSDEF, catalog_3DHST, catalog_MOSDEF)
        catalog.close(catalog_MOSDEF)
        print(f"    Matched = {matches_MOSDEF.count} sources", flush=True)
        del catalog_params_MOSDEF, catalog_MOSDEF, matches_MOSDEF

    # Match FMOS
    if field == 'COSMOS':
        print('  Matching FMOS...', flush=True)
        catalog_params_FMOS = catalog.get_params('FMOS')
        catalog_FMOS = catalog.read(catalog_params_FMOS)
        matches_FMOS = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_FMOS)
        match.append_cross_matched_data(matches_FMOS, catalog_3DHST, catalog_FMOS)
        catalog.close(catalog_FMOS)
        print(f"    Matched = {matches_FMOS.count} sources", flush=True)
        del catalog_params_FMOS, catalog_FMOS, matches_FMOS

    # Match KMOS3D
    if field == 'COSMOS' or field == 'GOODS-S' or field == 'UDS':
        print('  Matching KMOS3D...', flush=True)
        catalog_params_KMOS3D = catalog.get_params('KMOS3D', field)
        catalog_KMOS3D = catalog.read(catalog_params_KMOS3D)
        matches_KMOS3D = match.cross_match_catalogs_by_id(catalog_3DHST, catalog_KMOS3D)
        match.append_cross_matched_data(matches_KMOS3D, catalog_3DHST, catalog_KMOS3D)
        catalog.close(catalog_KMOS3D)
        print(f"    Matched = {matches_KMOS3D.count} sources", flush=True)
        del catalog_params_KMOS3D, catalog_KMOS3D, matches_KMOS3D

    # Match C3R2
    if field == 'COSMOS':
        print('  Matching C3R2...', flush=True)
        catalog_params_C3R2 = catalog.get_params('C3R2', field)
        catalog_C3R2 = catalog.read(catalog_params_C3R2)
        matches_C3R2 = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_C3R2)
        match.append_cross_matched_data(matches_C3R2, catalog_3DHST, catalog_C3R2)
        catalog.close(catalog_C3R2)
        print(f"    Matched = {matches_C3R2.count} sources", flush=True)
        del catalog_params_C3R2, catalog_C3R2, matches_C3R2

    # Match Lega-C
    if field == 'COSMOS':
        print('  Matching LEGA-C...', flush=True)
        catalog_params_LEGAC = catalog.get_params('LEGAC')
        catalog_LEGAC = catalog.read(catalog_params_LEGAC)
        matches_LEGAC = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_LEGAC)
        match.append_cross_matched_data(matches_LEGAC, catalog_3DHST, catalog_LEGAC)
        catalog.close(catalog_LEGAC)
        print(f"    Matched = {matches_LEGAC.count} sources", flush=True)
        del catalog_params_LEGAC, catalog_LEGAC, matches_LEGAC

    # Match HELP
    if field == 'AEGIS' or field == 'COSMOS':
        print('  Matching HELP...', flush=True)
        catalog_params_HELP = catalog.get_params('HELP', field)
        catalog_HELP = catalog.read(catalog_params_HELP)
        matches_HELP = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_HELP)
        match.append_cross_matched_data(matches_HELP, catalog_3DHST, catalog_HELP)
        catalog.close(catalog_HELP)
        print(f"    Matched = {matches_HELP.count} sources", flush=True)
        del catalog_params_HELP, catalog_HELP, matches_HELP

    # Match DESI
    print('  Matching DESI...', flush=True)
    catalog_params_DESI = catalog.get_params('DESI', field)
    catalog_DESI = catalog.read(catalog_params_DESI, region=catalog_region_3DHST, region_wcs=catalog_params_3DHST.wcs)
    matches_DESI = match.cross_match_catalogs_by_ra_dec(catalog_3DHST, catalog_DESI)
    match.append_cross_matched_data(matches_DESI, catalog_3DHST, catalog_DESI)
    catalog.close(catalog_DESI)
    print(f"    Matched = {matches_DESI.count} sources", flush=True)
    del catalog_params_DESI, catalog_DESI, matches_DESI

    # Save 3D-HST Matched
    print('  Saving 3D-HST PLUS catalog...', flush=True)
    catalog.consolidate_best_data(catalog_3DHST)
    catalog_params_3DHSTPLUS = catalog.get_params('3D-HST-PLUS')
    catalog.save_matched_catalog(catalog_3DHST, catalog_params_3DHSTPLUS, save_format='pickel', new_name='3D-HST-PLUS')
    if save_ascii:
        catalog.save_matched_catalog(catalog_3DHST, catalog_params_3DHSTPLUS, save_format='ascii', new_name='3D-HST-PLUS')
    print('    done', flush=True)
