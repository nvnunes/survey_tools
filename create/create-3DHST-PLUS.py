#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
from survey_tools import catalog
import create

save_ascii = True

fields = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S', 'UDS']
for i in np.arange(len(fields)):
    field = fields[i]

    # Load 3DHST
    print(f"Loading 3D-HST catalog {field} field...", flush=True)
    (catalog_params_3DHST, catalog_3DHST, catalog_region_3DHST) = create.load_3DHST(field)

    # Match zCOSMOS-Bright
    if field == 'COSMOS':
        print('  Matching zCOSMOS-Bright...', flush=True)
        (catalog_ZCB, matches_ZCB) = create.cross_match_ZCB(catalog_3DHST)
        print(f"    Matched = {matches_ZCB.count} sources", flush=True)
        catalog_ZCB.close()
        del catalog_ZCB, matches_ZCB

    # Match zCOSMOS-Deep
    if field == 'COSMOS':
        print('  Matching zCOSMOS-Deep...', flush=True)
        (catalog_ZCD, matches_ZCD) = create.cross_match_ZCD(catalog_3DHST)
        print(f"    Matched = {matches_ZCD.count} sources", flush=True)
        catalog_ZCD.close()
        del catalog_ZCD, matches_ZCD

    # Match VUDS
    if field == 'COSMOS' or field == 'GOODS-S':
        print('  Matching VUDS...', flush=True)
        (catalog_VUDS, matches_VUDS) = create.cross_match_VUDS(catalog_3DHST, field)
        print(f"    Matched = {matches_VUDS.count} sources", flush=True)
        catalog_VUDS.close()
        del catalog_VUDS, matches_VUDS

    # Match Casey DSFG
    if field == 'COSMOS' or field == 'GOODS-N' or field == 'UDS':
        print('  Matching Casey DSFG...', flush=True)
        (catalog_Casey, matches_Casey) = create.cross_match_Casey(catalog_3DHST, field)
        print(f"    Matched = {matches_Casey.count} sources", flush=True)
        catalog_Casey.close()
        del catalog_Casey, matches_Casey

    # Match DEIMOS 10K
    if field == 'COSMOS':
        print('  Matching DEIMOS 10K...', flush=True)
        (catalog_DEIMOS, matches_DEIMOS) = create.cross_match_DEMOS(catalog_3DHST)
        print(f"    Matched = {matches_DEIMOS.count} sources", flush=True)
        catalog_DEIMOS.close()
        del catalog_DEIMOS, matches_DEIMOS

    # Match MOSDEF
    if field == 'AEGIS' or field == 'COSMOS' or field == 'GOODS-N' or field == 'GOODS-S' or field == 'UDS':
        print('  Matching MOSDEF...', flush=True)
        (catalog_MOSDEF, matches_MOSDEF) = create.cross_match_MOSDEF(catalog_3DHST, field)
        print(f"    Matched = {matches_MOSDEF.count} sources", flush=True)
        catalog_MOSDEF.close()
        del catalog_MOSDEF, matches_MOSDEF

    # Match FMOS
    if field == 'COSMOS':
        print('  Matching FMOS...', flush=True)
        (catalog_FMOS, matches_FMOS) = create.cross_match_FMOS(catalog_3DHST)
        print(f"    Matched = {matches_FMOS.count} sources", flush=True)
        catalog_FMOS.close()
        del catalog_FMOS, matches_FMOS

    # Match KMOS3D
    if field == 'COSMOS' or field == 'GOODS-S' or field == 'UDS':
        print('  Matching KMOS3D...', flush=True)
        (catalog_KMOS3D, matches_KMOS3D) = create.cross_match_KMOS3D(catalog_3DHST, field)
        print(f"    Matched = {matches_KMOS3D.count} sources", flush=True)
        catalog_KMOS3D.close()
        del catalog_KMOS3D, matches_KMOS3D

    # Match C3R2
    if field == 'COSMOS':
        print('  Matching C3R2...', flush=True)
        (catalog_C3R2, matches_C3R2) = create.cross_match_C3R2(catalog_3DHST, field)
        print(f"    Matched = {matches_C3R2.count} sources", flush=True)
        catalog_C3R2.close()
        del catalog_C3R2, matches_C3R2

    # Match Lega-C
    if field == 'COSMOS':
        print('  Matching LEGA-C...', flush=True)
        (catalog_LEGAC, matches_LEGAC) = create.cross_match_LEGAC(catalog_3DHST)
        print(f"    Matched = {matches_LEGAC.count} sources", flush=True)
        catalog_LEGAC.close()
        del catalog_LEGAC, matches_LEGAC

    # Match HSCSSP
    if field == 'AEGIS' or field == 'COSMOS':
        print('  Matching HSCSSP...', flush=True)
        (catalog_HSCSSP, matches_HSCSSP) = create.cross_match_HSCSSP(catalog_3DHST, field)
        print(f"    Matched = {matches_HSCSSP.count} sources", flush=True)
        catalog_HSCSSP.close()
        del catalog_HSCSSP, matches_HSCSSP

    # Match DESI
    print('  Matching DESI...', flush=True)
    (catalog_DESI, matches_DESI) = create.cross_match_DESI(catalog_3DHST, field, catalog_region_3DHST, catalog_params_3DHST.wcs)
    print(f"    Matched = {matches_DESI.count} sources", flush=True)
    catalog_DESI.close()
    del catalog_DESI, matches_DESI

    # Save 3D-HST Matched
    print('  Saving 3D-HST PLUS catalog...', flush=True)
    create.consolidate_3DHST(catalog_3DHST)
    create.save_3DHSTPLUS(catalog_3DHST, 'pickel')
    if save_ascii:
        create.save_3DHSTPLUS(catalog_3DHST, 'ascii')
    print('    done', flush=True)
