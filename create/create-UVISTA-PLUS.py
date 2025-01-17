#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from survey_tools import catalog
import create

save_ascii = True

# Load UltraVISTA
print('Loading UltraVISTA catalog...', flush=True)
(catalog_params_UVISTA, catalog_UVISTA, catalog_region_UVISTA) = create.load_UVISTA()

# Match zCOSMOS-Bright
print('Matching zCOSMOS-Bright...', flush=True)
(catalog_ZCB, matches_ZCB) = create.cross_match_ZCB(catalog_UVISTA)
print(f"  Matched = {matches_ZCB.count} sources", flush=True)
catalog_ZCB.close()
del catalog_ZCB, matches_ZCB

# Match zCOSMOS-Deep
print('Matching zCOSMOS-Deep...', flush=True)
(catalog_ZCD, matches_ZCD) = create.cross_match_ZCD(catalog_UVISTA)
print(f"  Matched = {matches_ZCD.count} sources", flush=True)
catalog_ZCD.close()
del catalog_ZCD, matches_ZCD

# Match 3D-HST
print('Matching 3D-HST...', flush=True)
(catalog_3DHST, matches_3DHST) = create.cross_match_3DHST(catalog_UVISTA, 'COSMOS')
print(f"  Matched = {matches_3DHST.count} sources", flush=True)
catalog_3DHST.close()
del catalog_3DHST, matches_3DHST

# Match VUDS
print('Matching VUDS...', flush=True)
(catalog_VUDS, matches_VUDS) = create.cross_match_VUDS(catalog_UVISTA, 'COSMOS')
print(f"  Matched = {matches_VUDS.count} sources", flush=True)
catalog_VUDS.close()
del catalog_VUDS, matches_VUDS

# Match Casey DSFG
print('Matching Casey DSFG...', flush=True)
(catalog_Casey, matches_Casey) = create.cross_match_Casey(catalog_UVISTA, 'COSMOS')
print(f"  Matched = {matches_Casey.count} sources", flush=True)
catalog_Casey.close()
del catalog_Casey, matches_Casey

# Match DEIMOS 10K
print('Matching DEIMOS 10K...', flush=True)
(catalog_DEIMOS, matches_DEIMOS) = create.cross_match_DEMOS(catalog_UVISTA)
print(f"  Matched = {matches_DEIMOS.count} sources", flush=True)
catalog_DEIMOS.close()
del catalog_DEIMOS, matches_DEIMOS

# Match MOSDEF
print('Matching MOSDEF...', flush=True)
(catalog_MOSDEF, matches_MOSDEF) = create.cross_match_MOSDEF(catalog_UVISTA, 'COSMOS')
print(f"  Matched = {matches_MOSDEF.count} sources", flush=True)
catalog_MOSDEF.close()
del catalog_MOSDEF, matches_MOSDEF

# Match FMOS
print('Matching FMOS...', flush=True)
(catalog_FMOS, matches_FMOS) = create.cross_match_FMOS(catalog_UVISTA)
print(f"  Matched = {matches_FMOS.count} sources", flush=True)
catalog_FMOS.close()
del catalog_FMOS, matches_FMOS

# Match KMOS3D
print('Matching KMOS3D...', flush=True)
(catalog_KMOS3D, matches_KMOS3D) = create.cross_match_KMOS3D(catalog_UVISTA, 'COSMOS')
print(f"  Matched = {matches_KMOS3D.count} sources", flush=True)
catalog_KMOS3D.close()
del catalog_KMOS3D, matches_KMOS3D

# Match C3R2
print('Matching C3R2...', flush=True)
(catalog_C3R2, matches_C3R2) = create.cross_match_C3R2(catalog_UVISTA, 'COSMOS')
print(f"  Matched = {matches_C3R2.count} sources", flush=True)
catalog_C3R2.close()
del catalog_C3R2, matches_C3R2

# Match Lega-C
print('Matching LEGA-C...', flush=True)
(catalog_LEGAC, matches_LEGAC) = create.cross_match_LEGAC(catalog_UVISTA)
print(f"  Matched = {matches_LEGAC.count} sources", flush=True)
catalog_LEGAC.close()
del catalog_LEGAC, matches_LEGAC

# Match HSCSSP
print('Matching HSCSSP...', flush=True)
(catalog_HSCSSP, matches_HSCSSP) = create.cross_match_HSCSSP(catalog_UVISTA, 'COSMOS')
print(f"  Matched = {matches_HSCSSP.count} sources", flush=True)
catalog_HSCSSP.close()
del catalog_HSCSSP, matches_HSCSSP

# Match DESI
print('Matching DESI...', flush=True)
(catalog_DESI, matches_DESI) = create.cross_match_DESI(catalog_UVISTA, 'COSMOS', catalog_region_UVISTA, catalog_params_UVISTA.wcs)
print(f"  Matched = {matches_DESI.count} sources", flush=True)
catalog_DESI.close()
del catalog_DESI, matches_DESI

# Save UltraVISTA Matched
print('Saving UltraVISTA PLUS catalog...', flush=True)
create.consolidate_UVISTA(catalog_UVISTA)
create.save_UVISTAPLUS(catalog_UVISTA, 'pickel')
if save_ascii:
    create.save_UVISTAPLUS(catalog_UVISTA, 'ascii')
print('  done', flush=True)
