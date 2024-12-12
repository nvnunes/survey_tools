#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from survey_tools import catalog, match

save_ascii = True

# Load UltraVISTA
print('Loading UltraVISTA catalog...', flush=True)
catalog_params_UVISTA = catalog.get_params('UVISTA', 'COSMOS', 'K')
catalog_UVISTA = catalog.read(catalog_params_UVISTA)
catalog_region_UVISTA = catalog.get_catalog_region(catalog_UVISTA)

# Match zCOSMOS-Bright
print('Matching zCOSMOS-Bright...', flush=True)
catalog_params_ZCB = catalog.get_params('ZCOSMOS-BRIGHT')
catalog_ZCB = catalog.read(catalog_params_ZCB)
matches_ZCB = match.cross_match_catalogs_by_id_ra_dec(catalog_UVISTA, catalog_ZCB, id_field1='z_spec_id')
match.append_cross_matched_data(matches_ZCB, catalog_UVISTA, catalog_ZCB)
catalog.close(catalog_ZCB)
print(f"  Matched = {matches_ZCB.count} sources", flush=True)
del catalog_params_ZCB, catalog_ZCB, matches_ZCB

# Match zCOSMOS-Deep
print('Matching zCOSMOS-Deep...', flush=True)
catalog_params_ZCD = catalog.get_params('ZCOSMOS-DEEP')
catalog_ZCD = catalog.read(catalog_params_ZCD)
matches_ZCD = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_ZCD)
match.append_cross_matched_data(matches_ZCD, catalog_UVISTA, catalog_ZCD)
catalog.close(catalog_ZCD)
print(f"  Matched = {matches_ZCD.count} sources", flush=True)
del catalog_params_ZCD, catalog_ZCD, matches_ZCD

# Match 3D-HST
print('Matching 3D-HST...', flush=True)
catalog_params_3DHST = catalog.get_params('3D-HST', 'COSMOS', 'F160W')
catalog_3DHST = catalog.read(catalog_params_3DHST)
matches_3DHST = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_3DHST)
match.append_cross_matched_data(matches_3DHST, catalog_UVISTA, catalog_3DHST)
catalog.close(catalog_3DHST)
print(f"  Matched = {matches_3DHST.count} sources", flush=True)
del catalog_params_3DHST, catalog_3DHST, matches_3DHST

# Match VUDS
print('Matching VUDS...', flush=True)
catalog_params_VUDS = catalog.get_params('VUDS', 'COSMOS')
catalog_VUDS = catalog.read(catalog_params_VUDS)
matches_VUDS = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_VUDS)
match.append_cross_matched_data(matches_VUDS, catalog_UVISTA, catalog_VUDS)
catalog.close(catalog_VUDS)
print(f"  Matched = {matches_VUDS.count} sources", flush=True)
del catalog_params_VUDS, catalog_VUDS, matches_VUDS

# Match Casey DSFG
print('Matching Casey DSFG...', flush=True)
catalog_params_Casey = catalog.get_params('Casey', 'COSMOS')
catalog_Casey = catalog.read(catalog_params_Casey)
matches_Casey = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_Casey)
match.append_cross_matched_data(matches_Casey, catalog_UVISTA, catalog_Casey)
catalog.close(catalog_Casey)
print(f"  Matched = {matches_Casey.count} sources", flush=True)
del catalog_params_Casey, catalog_Casey, matches_Casey

# Match DEIMOS 10K
print('Matching DEIMOS 10K...', flush=True)
catalog_params_DEIMOS = catalog.get_params('DEIMOS')
catalog_DEIMOS = catalog.read(catalog_params_DEIMOS)
matches_DEIMOS = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_DEIMOS)
match.append_cross_matched_data(matches_DEIMOS, catalog_UVISTA, catalog_DEIMOS)
catalog.close(catalog_DEIMOS)
print(f"  Matched = {matches_DEIMOS.count} sources", flush=True)
del catalog_params_DEIMOS, catalog_DEIMOS, matches_DEIMOS

# Match MOSDEF
print('Matching MOSDEF...', flush=True)
catalog_params_MOSDEF = catalog.get_params('MOSDEF', 'COSMOS')
catalog_MOSDEF = catalog.read(catalog_params_MOSDEF)
matches_MOSDEF = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_MOSDEF)
match.append_cross_matched_data(matches_MOSDEF, catalog_UVISTA, catalog_MOSDEF)
catalog.close(catalog_MOSDEF)
print(f"  Matched = {matches_MOSDEF.count} sources", flush=True)
del catalog_params_MOSDEF, catalog_MOSDEF, matches_MOSDEF

# Match FMOS
print('Matching FMOS...', flush=True)
catalog_params_FMOS = catalog.get_params('FMOS')
catalog_FMOS = catalog.read(catalog_params_FMOS)
matches_FMOS = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_FMOS)
match.append_cross_matched_data(matches_FMOS, catalog_UVISTA, catalog_FMOS)
catalog.close(catalog_FMOS)
print(f"  Matched = {matches_FMOS.count} sources", flush=True)
del catalog_params_FMOS, catalog_FMOS, matches_FMOS

# Match KMOS3D
print('Matching KMOS3D...', flush=True)
catalog_params_KMOS3D = catalog.get_params('KMOS3D', 'COSMOS')
catalog_KMOS3D = catalog.read(catalog_params_KMOS3D)
matches_KMOS3D = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_KMOS3D)
match.append_cross_matched_data(matches_KMOS3D, catalog_UVISTA, catalog_KMOS3D)
catalog.close(catalog_KMOS3D)
print(f"  Matched = {matches_KMOS3D.count} sources", flush=True)
del catalog_params_KMOS3D, catalog_KMOS3D, matches_KMOS3D

# Match C3R2
print('Matching C3R2...', flush=True)
catalog_params_C3R2 = catalog.get_params('C3R2', 'COSMOS')
catalog_C3R2 = catalog.read(catalog_params_C3R2)
matches_C3R2 = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_C3R2)
match.append_cross_matched_data(matches_C3R2, catalog_UVISTA, catalog_C3R2)
catalog.close(catalog_C3R2)
print(f"  Matched = {matches_C3R2.count} sources", flush=True)
del catalog_params_C3R2, catalog_C3R2, matches_C3R2

# Match Lega-C
print('Matching LEGA-C...', flush=True)
catalog_params_LEGAC = catalog.get_params('LEGAC')
catalog_LEGAC = catalog.read(catalog_params_LEGAC)
matches_LEGAC = match.cross_match_catalogs_by_id_ra_dec(catalog_UVISTA, catalog_LEGAC, id_field2='ID')
match.append_cross_matched_data(matches_LEGAC, catalog_UVISTA, catalog_LEGAC)
catalog.close(catalog_LEGAC)
print(f"  Matched = {matches_LEGAC.count} sources", flush=True)
del catalog_params_LEGAC, catalog_LEGAC, matches_LEGAC

# Match HELP
print('Matching HELP...', flush=True)
catalog_params_HELP = catalog.get_params('HELP', 'COSMOS')
catalog_HELP = catalog.read(catalog_params_HELP)
matches_HELP = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_HELP)
match.append_cross_matched_data(matches_HELP, catalog_UVISTA, catalog_HELP)
catalog.close(catalog_HELP)
print(f"  Matched = {matches_HELP.count} sources", flush=True)
del catalog_params_HELP, catalog_HELP, matches_HELP

# Match DESI
print('Matching DESI...', flush=True)
catalog_params_DESI = catalog.get_params('DESI', 'COSMOS')
catalog_DESI = catalog.read(catalog_params_DESI, region=catalog_region_UVISTA, region_wcs=catalog_params_UVISTA.wcs)
matches_DESI = match.cross_match_catalogs_by_ra_dec(catalog_UVISTA, catalog_DESI)
match.append_cross_matched_data(matches_DESI, catalog_UVISTA, catalog_DESI)
catalog.close(catalog_DESI)
print(f"  Matched = {matches_DESI.count} sources", flush=True)
del catalog_params_DESI, catalog_DESI, matches_DESI

# Save UltraVISTA Matched
print('Saving UltraVISTA PLUS catalog...', flush=True)
catalog.consolidate_best_data(catalog_UVISTA)
catalog_params_UVISTAPLUS = catalog.get_params('UVISTA-PLUS')
catalog.save_matched_catalog(catalog_UVISTA, catalog_params_UVISTAPLUS, save_format='pickel', new_name='UVISTA-PLUS')
if save_ascii:
    catalog.save_matched_catalog(catalog_UVISTA, catalog_params_UVISTAPLUS, save_format='ascii', new_name='UVISTA-PLUS')
print('  done', flush=True)
