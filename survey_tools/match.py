#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
from survey_tools import catalog
from survey_tools.utility import table

class StructType:
    pass

class MatchException(Exception):
    pass

def cross_match_catalogs_by_id(cat1, cat2, id_field1=None, id_field2=None, assume_unique=False):
    if cat2.count == 0:
        matches = StructType()
        matches.count = 0
        return matches

    if id_field1 is None:
        id_field1 = catalog.get_id_field(cat1)
    if id_field2 is None:
        id_field2 = catalog.get_id_field(cat2)

    _, cat1_idx, cat2_idx = np.intersect1d(cat1.sources[id_field1], cat2.sources[id_field2], assume_unique=assume_unique, return_indices=True)

    matches = StructType()
    matches.idx1 = cat1_idx
    matches.idx2 = cat2_idx
    matches.seps = np.zeros(len(cat1_idx))
    matches.dist = np.zeros(len(cat1_idx))
    matches.count = len(matches.idx1)

    unmatched_filter = np.ones((len(cat2.sources)), dtype=np.bool)
    unmatched_filter[cat2_idx] = False

    matches.unmatched_idx2 = np.arange(len(cat2.sources))[unmatched_filter]
    matches.unmatched_num = np.zeros((len(matches.unmatched_idx2)), dtype=np.int_)

    return matches

def multi_match_coords(coords1, coords2, max_sep):
    N = len(coords1)

    tmp_idx1s, tmp_idx2s, tmp_seps, tmp_dist = search_around_sky(coords1, coords2, max_sep, storekdtree='kdtree')
    sorting_indexes = np.argsort(tmp_idx1s)
    tmp_idx1s = tmp_idx1s[sorting_indexes]
    tmp_idx2s = tmp_idx2s[sorting_indexes]
    tmp_seps  = tmp_seps[sorting_indexes]
    tmp_dist  = tmp_dist[sorting_indexes]

    idx2s = [np.array([], dtype=np.int_) for i in np.arange(N)]
    seps = [np.array([]) for i in np.arange(N)]
    dist = [np.array([]) for i in np.arange(N)]

    if len(tmp_idx1s) > 0:
        j = 0
        for i in np.arange(N):
            if i < tmp_idx1s[j]:
                continue
            while j < len(tmp_idx1s) and tmp_idx1s[j] == i:
                idx2s[i] = np.append(idx2s[i], tmp_idx2s[j])
                seps[i] = np.append(seps[i], tmp_seps[j].to_value(u.arcsec))
                dist[i] = np.append(dist[i], tmp_dist[j].to_value(u.Mpc))
                j += 1
            if j == len(tmp_idx1s):
                break

    return idx2s, seps, dist

def cross_match_catalogs_by_ra_dec(cat1, cat2, max_sep=None, mode=None):
    if cat2.count == 0:
        matches = StructType()
        matches.count = 0
        return matches

    if max_sep is None:
        max_sep = 1.0*u.arcsec
    elif not isinstance(max_sep, u.Quantity):
        max_sep = max_sep * u.arcsec

    if mode is None or mode.lower() != 'closest':
        mode = 'multi'
    else:
        mode = 'closest'

    z1, _ = catalog.get_redshift_any(cat1)
    z2, _ = catalog.get_redshift_any(cat2)

    cat1_filter = z1 > 0.0
    cat2_filter = z2 > 0.0

    cat1_index_map = np.arange(cat1.count)[cat1_filter]
    cat2_index_map = np.arange(cat2.count)[cat2_filter]

    ra1  = cat1.sources[catalog.get_ra_field(cat1)][cat1_filter]
    dec1 = cat1.sources[catalog.get_dec_field(cat1)][cat1_filter]
    ra2  = cat2.sources[catalog.get_ra_field(cat2)][cat2_filter]
    dec2 = cat2.sources[catalog.get_dec_field(cat2)][cat2_filter]

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DA1 = cosmo.angular_diameter_distance(z1[cat1_filter])
    DA2 = cosmo.angular_diameter_distance(z2[cat2_filter])

    cat1_coords = SkyCoord(ra=ra1, dec=dec1, distance=DA1, frame=cat1.frame, unit=(u.degree, u.degree, u.Mpc))
    cat2_coords = SkyCoord(ra=ra2, dec=dec2, distance=DA2, frame=cat2.frame, unit=(u.degree, u.degree, u.Mpc))

    if mode == 'closest':
        idx2, seps, dist = match_coordinates_sky(cat1_coords, cat2_coords, nthneighbor=1)
        seps = np.array([s.to_value(u.arcsec) for s in seps])
        dist = np.array([d.to_value(u.Mpc) for d in dist])
        match_filter = seps <= max_sep
    else:
        # NOTE: In the current implementation [of search_around_sky], the return values are always
        # sorted in the same order as the coords1 (so idx1 is in ascending order). This is considered
        # an implementation detail, though, so it could change in a future release.

        cat1_idx2s, cat1_seps, cat1_dist = multi_match_coords(cat1_coords, cat2_coords, max_sep)
        cat2_idx1s, _        , cat2_dist = multi_match_coords(cat2_coords, cat1_coords, max_sep)

        N = len(cat1_coords)
        idx2 = -1 * np.ones((N), dtype=np.int_)
        seps = np.zeros((N))
        dist = np.zeros((N))

        for i in np.arange(N):
            # Case 1: No matches on Cat1 side
            if len(cat1_idx2s[i]) == 0:
                continue

            match1_closest_idx = np.argmin(cat1_dist[i])
            closest_idx2 = cat1_idx2s[i][match1_closest_idx]

            # Case 2: No matches on Cat2 side for closest on Cat1 side (can happen when dist==0.0)
            if len(cat2_idx1s[closest_idx2]) == 0:
                raise MatchException('Cat1 closest points to a Cat2 that does not point back')

            match2_closest_idx = np.argmin(cat2_dist[closest_idx2])

            # Case 3: Closest (or only) match on Cat1 side is closest (or only) match on Cat2 side
            if cat2_idx1s[closest_idx2][match2_closest_idx] == i:
                idx2[i] = closest_idx2
                seps[i] = cat1_seps[i][match1_closest_idx]
                dist[i] = cat1_dist[i][match1_closest_idx]
                continue

            # Case 4: Only one match on both sides but not consistent
            if len(cat1_idx2s[i]) == 1 and len(cat2_idx1s[closest_idx2]) == 1:
                raise MatchException('Only one match in Cat1 and Cat2 but not consistent')

            # Case 5: Multiple on both sides, Cat1 not in list of Cat2
            matching_idx = np.argmax(cat2_idx1s[closest_idx2] == i)
            if cat2_idx1s[closest_idx2][matching_idx] != i:
                raise MatchException('Multiple on both sides but Cat1 is not in Cat2 list')

            # Case 6: Single Cat1 and multiple Cat2 (including Cat1)
            if len(cat1_idx2s[i]) == 1 and len(cat2_idx1s[closest_idx2]) > 1 \
            or len(cat1_idx2s[i]) > 1 and len(cat2_idx1s[closest_idx2]) == 1:
                # Will only get here with a multi-match where "i" is not the closest match
                # So it is ok to SKIP Cat1 in this case
                continue

            # Case 7: Multiple on both sides including each other
            # SKIP: Multiple sources within 1" which isn't viable.
            continue

        match_filter = idx2 >= 0

    matches = StructType()
    matches.idx1 = cat1_index_map[match_filter]
    matches.idx2 = cat2_index_map[idx2[match_filter]]
    matches.seps = seps[match_filter]
    matches.dist = dist[match_filter]
    matches.count = len(matches.idx1)

    unmatched_filter = np.ones((len(cat2_coords)), dtype=np.bool)
    unmatched_filter[idx2[match_filter]] = False
    idx1s, _, _ = multi_match_coords(cat2_coords[unmatched_filter], cat1_coords, max_sep)

    matches.unmatched_idx2 = np.hstack([np.arange(cat2.count)[~cat2_filter], cat2_index_map[unmatched_filter]])
    matches.unmatched_num = np.hstack([np.zeros((np.sum(~cat2_filter)), dtype=np.int_), np.array([len(a) for a in idx1s])])

    return matches

def cross_match_catalogs_by_id_ra_dec(cat1, cat2, id_field1=None, id_field2=None, sky_max_sep=None, sky_mode=None):
    if cat2.count == 0:
        matches = StructType()
        matches.count = 0
        return matches

    if id_field1 is None:
        id_field1 = catalog.get_id_field(cat1)

    if id_field2 is None:
        id_field2 = catalog.get_id_field(cat2)

    matches_by_id     = cross_match_catalogs_by_id(cat1, cat2, id_field1, id_field2)
    matches_by_ra_dec = cross_match_catalogs_by_ra_dec(cat1, cat2, max_sep=sky_max_sep, mode=sky_mode)

    dup_matched_filter = np.in1d(matches_by_ra_dec.idx1, matches_by_id.idx1)

    matches = StructType()
    matches.idx1 = np.hstack([matches_by_id.idx1, matches_by_ra_dec.idx1[~dup_matched_filter]])
    matches.idx2 = np.hstack([matches_by_id.idx2, matches_by_ra_dec.idx2[~dup_matched_filter]])
    matches.seps = np.hstack([matches_by_id.seps, matches_by_ra_dec.seps[~dup_matched_filter]])
    matches.dist = np.hstack([matches_by_id.dist, matches_by_ra_dec.dist[~dup_matched_filter]])
    matches.count = len(matches.idx1)

    dup_unmatched_filter = np.in1d(matches_by_ra_dec.unmatched_idx2, matches.idx2)

    matches.unmatched_idx2 = matches_by_ra_dec.unmatched_idx2[~dup_unmatched_filter]
    matches.unmatched_num = matches_by_ra_dec.unmatched_num[~dup_unmatched_filter]

    return matches

def append_cross_matched_data(matches, cat1, cat2):
    if matches.count == 0:
        return

    if not np.any(np.array(cat1.all_sources) == cat2.source):
        cat1.all_sources.append(cat2.source)
        cat1.all_sources_date.append(cat2.date)

    # Step 1: Initialize
    match_id  = -1 * np.ones((cat1.count, 1), dtype=np.int_)
    match_num = -1 * np.ones((cat1.count, 1), dtype=np.int_)
    match_sep = -1 * np.ones((cat1.count, 1))
    match_dist = -1 * np.ones((cat1.count, 1))
    match_dz  = -1 * np.ones((cat1.count, 1))

    z_any = -99.0 * np.ones((cat1.count, 1))
    z_any_flag = 99 * np.ones((cat1.count, 1), dtype=np.int_)

    has_use_phot = catalog.get_has_use_phot(cat2)
    if has_use_phot:
        use_phot_field = catalog.get_use_phot_field(cat2)
        use_phot = np.zeros((cat1.count), dtype=np.int_)

    has_star_flag = catalog.get_has_star_flag(cat2)
    if has_star_flag:
        star_flag_field = catalog.get_star_flag_field(cat2)
        star_flag = np.zeros((cat1.count), dtype=np.int_)

    has_flux = catalog.get_has_flux(cat2)
    if has_flux:
        flux_field = catalog.get_flux_field(cat2)
        flux_zero_point = catalog.get_flux_zero_point(cat2)
        mag = np.zeros((cat1.count))

    has_flux_radius = catalog.get_has_flux_radius(cat2)
    if has_flux_radius:
        flux_radius_field = catalog.get_flux_radius_field(cat2)
        flux_radius_factor = catalog.get_flux_radius_factor(cat2)
        flux_radius = np.zeros((cat1.count))

    has_spp = catalog.get_has_spp(cat2)
    if has_spp:
        if not table.has_field(cat1, 'spp'):
            cat1.spp = Table()
            cat1.spp.add_column(cat1.sources[catalog.get_id_field(cat1)])

        spp = -99.0 * np.ones((cat1.count, 5)) # mass, sfr, Av, chi2, z_spp
        spp_flags = 99 * np.ones((cat1.count))

    has_lines = catalog.get_has_lines(cat2)
    if has_lines:
        if not table.has_field(cat1, 'lines'):
            cat1.lines = Table()
            cat1.lines.add_column(cat1.sources[catalog.get_id_field(cat1)])

        line_names = catalog.get_line_names()
        num_lines = len(line_names)
        lines = np.zeros((cat1.count, num_lines, 3)) # flux, flux_error, fwhm
        line_flags = 99 * np.ones((cat1.count, num_lines))

    # Step 2: Process matches
    for i in np.arange(len(matches.idx1)):
        idx1 = matches.idx1[i]
        idx2 = matches.idx2[i]

        id2         = get_source_id(cat2, idx2)
        z1, _       = catalog.get_redshift_any(cat1, idx1)
        z2, z2_flag = catalog.get_redshift_any(cat2, idx2)

        match_id[idx1]  = id2
        match_num[idx1] = 1
        match_sep[idx1] = matches.seps[i]
        match_dist[idx1] = matches.dist[i]
        match_dz[idx1]  = z2 - z1

        z_any[idx1] = z2
        z_any_flag[idx1] = z2_flag

        if has_use_phot:
            if use_phot_field is True:
                use_phot[idx1] = 1
            else:
                use_phot[idx1] = cat2.sources[use_phot_field][idx2]

        if has_star_flag:
            if star_flag_field is True:
                star_flag[idx1] = 1
            else:
                star_flag[idx1] = cat2.sources[star_flag_field][idx2]

        if has_flux:
            if cat2.sources[flux_field][idx2] > 0.0:
                mag[idx1] = -2.5*np.log10(cat2.sources[flux_field][idx2]) + flux_zero_point

        if has_flux_radius:
            flux_radius[idx1] = cat2.sources[flux_radius_field][idx2] * flux_radius_factor

        if has_spp:
            spp[idx1,:], spp_flags[idx1], spp_names = catalog.get_spp(cat2, idx2)

        if has_lines:
            lines[idx1,:,:], line_flags[idx1,:], _ = catalog.get_lines(cat2, idx2)

    # Step 3: Handle missing sources from Cat2
    num_missing_sources = len(matches.unmatched_idx2)
    if num_missing_sources > 0:
        add_missing_sources(cat1, cat2, matches.unmatched_idx2)

        match_id = np.vstack([match_id, cat2.sources[catalog.get_id_field(cat2)][matches.unmatched_idx2].reshape((-1,1))])
        match_num = np.vstack([match_num, matches.unmatched_num.reshape((-1,1))])
        match_sep = np.vstack([match_sep, np.zeros((num_missing_sources, 1))])
        match_dist = np.vstack([match_dist, np.zeros((num_missing_sources, 1))])
        match_dz = np.vstack([match_dz, np.zeros((num_missing_sources, 1))])

        missing_z_any, missing_z_any_flag = catalog.get_redshift_any(cat2, matches.unmatched_idx2)
        z_any = np.vstack([z_any, np.array(missing_z_any.reshape((-1,1)))])
        z_any_flag = np.vstack([z_any_flag, np.array(missing_z_any_flag.reshape((-1,1)))])

        if has_use_phot:
            if use_phot_field is True:
                missing_use_phot = np.ones((len(matches.unmatched_idx2)), dtype=np.int_)
            else:
                missing_use_phot = cat2.sources[use_phot_field][matches.unmatched_idx2]
            use_phot = np.vstack([use_phot.reshape((-1,1)), missing_use_phot.reshape((-1,1))])

        if has_star_flag:
            if star_flag_field is True:
                missing_star_flag = np.ones((len(matches.unmatched_idx2)), dtype=np.int_)
            else:
                missing_star_flag = cat2.sources[star_flag_field][matches.unmatched_idx2]
            star_flag = np.vstack([star_flag.reshape((-1,1)), missing_star_flag.reshape((-1,1))])

        if has_flux:
            missing_mag = np.zeros((len(matches.unmatched_idx2)))
            value_filter = cat2.sources[flux_field][matches.unmatched_idx2] > 0.0
            missing_mag[value_filter] = -2.5*np.log10(cat2.sources[flux_field][matches.unmatched_idx2][value_filter]) + flux_zero_point
            mag = np.vstack([mag.reshape((-1,1)), missing_mag.reshape((-1,1))])

        if has_flux_radius:
            missing_flux_radius = cat2.sources[flux_radius_field][matches.unmatched_idx2] * flux_radius_factor
            flux_radius = np.vstack([flux_radius.reshape((-1,1)), missing_flux_radius.reshape((-1,1))])

        if has_spp:
            missing_spp, missing_spp_flags, _ = catalog.get_spp(cat2, matches.unmatched_idx2)
            spp = np.vstack([spp, missing_spp])
            spp_flags = np.vstack([spp_flags.reshape((-1,1)), missing_spp_flags.reshape((-1,1))])

        if has_lines:
            missing_lines, missing_line_flags, _ = catalog.get_lines(cat2, matches.unmatched_idx2)
            lines = np.vstack([lines, missing_lines])
            line_flags = np.vstack([line_flags, missing_line_flags])

    # Step 4: Add extra columns to Cat1
    table.add_fields(cat1.redshift, f"z_{cat2.source}", z_any)
    table.add_fields(cat1.redshift, f"z_{cat2.source}_flag", z_any_flag)

    if has_use_phot:
        table.add_fields(cat1.sources, f"use_phot_{cat2.source}", use_phot)

    if has_star_flag:
        table.add_fields(cat1.sources, f"star_flag_{cat2.source}", star_flag)

    if has_flux:
        table.add_fields(cat1.sources, f"mag_{cat2.source}", mag)

    if has_flux_radius:
        table.add_fields(cat1.sources, f"flux_radius_{cat2.source}", flux_radius)

    if has_spp:
        for i in np.arange(len(spp_names)):
            table.add_fields(cat1.spp, f"{spp_names[i]}_{cat2.source}", spp[:,i])

        table.add_fields(cat1.spp, f"flag_{cat2.source}", spp_flags)

    if has_lines:
        for i in np.arange(num_lines):
            table.add_fields(cat1.lines, [f"f_{line_names[i]}_{cat2.source}", f"e_{line_names[i]}_{cat2.source}", f"fwhm_{line_names[i]}_{cat2.source}"], lines[:,i,:])
            table.add_fields(cat1.lines, f"flag_{line_names[i]}_{cat2.source}", line_flags[:,i])

    # Step 5: Add to idmap
    if not table.has_field(cat1, 'idmap'):
        cat1.idmap = Table()
        cat1.idmap.add_column(np.array(cat1.sources[catalog.get_id_field(cat1)]), name=catalog.get_id_field(cat1, table_name='idmap'))

    table.add_fields(cat1.idmap, cat2.source, match_id)
    table.add_fields(cat1.idmap, f"{cat2.source}_num", match_num)
    table.add_fields(cat1.idmap, f"{cat2.source}_sep", match_sep)
    table.add_fields(cat1.idmap, f"{cat2.source}_dist", match_dist)
    table.add_fields(cat1.idmap, f"{cat2.source}_dz", match_dz)

def get_source_id(catalog_data, idx_or_filter = None):
    if idx_or_filter is None:
        N = catalog_data.count
        idx_or_filter = np.ones((N), dtype=np.bool)
    elif np.size(idx_or_filter) == 1:
        if idx_or_filter == -1:
            return -1
        N = 1
    elif issubclass(idx_or_filter.dtype.type, np.integer):
        N = len(idx_or_filter)
    else:
        N = np.sum(idx_or_filter)

    field_name = catalog.get_id_field(catalog_data)
    return catalog_data.sources[field_name][idx_or_filter]

# pylint: disable=redefined-builtin
def add_missing_sources(cat1, cat2, idx_or_filter):
    if table.has_field(cat1, 'redshift') and not table.has_field(cat1.redshift, 'z_merged'):
        table.add_fields(cat1.redshift, ['z_merged', 'z_merged_flag'], [-99.0, 99])

    if table.has_field(cat2.sources, 'index'):
        id = cat2.sources['index'][idx_or_filter]
    else:
        id = cat2.sources[catalog.get_id_field(cat2)][idx_or_filter].astype(np.int_)

    ra = cat2.sources[catalog.get_ra_field(cat2)][idx_or_filter]
    dec = cat2.sources[catalog.get_dec_field(cat2)][idx_or_filter]
    z, z_flag = catalog.get_redshift_any(cat2, idx_or_filter)

    offset = 10000000*(len(cat1.all_sources)-1)
    id += offset

    cat1.sources = table.add_rows(cat1, cat1.sources, [catalog.get_id_field(cat1), catalog.get_ra_field(cat1), catalog.get_dec_field(cat1)], np.array([id, ra, dec]), default_value_func = get_default_value)

    if table.has_field(cat1, 'redshift'):
        cat1.redshift = table.add_rows(cat1, cat1.redshift, [catalog.get_id_field(cat1, 'redshift'), 'z_merged', 'z_merged_flag'], np.array([id, z, z_flag]), default_value_func = get_default_value)
    if table.has_field(cat1, 'spp'):
        cat1.spp = table.add_rows(cat1, cat1.spp, catalog.get_id_field(cat1, 'spp'), id, default_value_func = get_default_value)
    if table.has_field(cat1, 'lines'):
        cat1.lines = table.add_rows(cat1, cat1.lines, catalog.get_id_field(cat1, 'lines'), id, default_value_func = get_default_value)
    if table.has_field(cat1, 'clumps'):
        cat1.clumps = table.add_rows(cat1, cat1.clumps, catalog.get_id_field(cat1, 'clumps'), id, default_value_func = get_default_value)
    if table.has_field(cat1, 'idmap'):
        cat1.idmap = table.add_rows(cat1, cat1.idmap, catalog.get_id_field(cat1, 'idmap'), id, default_value=-1)

    cat1.count += len(id)

def get_default_value(column_name, catalog_data):
    if 'flag_' in column_name or '_flag' in column_name: # Must come first to prevent fall-through
        return 99
    elif column_name == catalog.get_x_field(catalog_data) or column_name == catalog.get_y_field(catalog_data):
        return -1
    elif column_name == 'z' or 'z_' in column_name or 'lmass' in column_name or 'lsfr' in column_name or 'Av' in column_name or 'av' in column_name or 'chi2' in column_name:
        return -99
    elif column_name == 'UV_frac_clump':
        return -1.0
    else:
        return None
