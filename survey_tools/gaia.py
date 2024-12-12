#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
from astroquery.gaia import Gaia

def get_stars(min_ra, max_ra, min_dec, max_dec):
    ra = (min_ra + max_ra)/2
    dec = (min_dec + max_dec)/2
    width = abs(min_ra - max_ra)
    height = abs(min_dec - max_dec)

    job = Gaia.launch_job_async("SELECT SOURCE_ID AS gaia_id,"
                                "       ra AS gaia_ra,"
                                "       dec AS gaia_dec,"
                                "       phot_g_mean_mag AS gaia_G,"
                                "       phot_bp_mean_mag AS gaia_BP,"
                                "       phot_rp_mean_mag AS gaia_RP,"
                                "       ref_epoch AS gaia_ref_epoch,"
                                "       pmra AS gaia_pmra,"
                                "       pmdec AS gaia_pmdec,"
                                "       non_single_star AS gaia_is_multiple"
                                "  FROM gaiadr3.gaia_source"
                               f" WHERE CONTAINS(POINT('ICRS', ra, dec), BOX('ICRS',{ra},{dec},{width},{height})) = 1"
                                "   AND in_qso_candidates = '0'"
                                "   AND in_galaxy_candidates = '0'"
                                "   AND in_andromeda_survey = '0'"
                                " ORDER BY SOURCE_ID"
                            , verbose=False)
    #                       BOX is deprecated so when the above doesn't work, try the following instead:
    #                          f" WHERE CONTAINS(POINT('ICRS', ra, dec), POLYGON({min_ra},{min_dec},{max_ra},{min_dec},{max_ra},{max_dec},{min_ra},{max_dec})) = TRUE"

    results = job.get_results()

    if len(results) > 0:
        # see: https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T9
        mag_diff = results['gaia_BP'] - results['gaia_RP']
        results['G_minus_R_mag'] = (-0.02275) + (0.3961)*mag_diff + (-0.1243)*np.power(mag_diff, 2) + (-0.01396)*np.power(mag_diff, 3) + (0.003775)*np.power(mag_diff, 4)
        results['R_mag'] = results['gaia_G'] - results['G_minus_R_mag']

    return results
