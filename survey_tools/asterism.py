#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import time
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
import astropy.units as u
from shapely.geometry import Point, Polygon

class StructType:
    pass

class AsterismException(Exception):
    pass

#region Find

def find_asterisms(star_data, id_field = 'id', ra_field = 'ra', dec_field = 'dec', mag_field = 'mag', min_stars = 1, max_stars = 1, min_separation = 0.0, max_separation = 60.0, max_1ngs_distance = 30.0, verbose=False):
    star_catalog = SkyCoord(ra=star_data[ra_field], dec=star_data[dec_field], unit=(u.degree, u.degree))

    # Find stars that are close to each other
    star_idx1s, star_idx2s, seps, _ = search_around_sky(star_catalog, star_catalog, max_separation*u.arcsec)
    sorting_indexes = np.argsort(star_idx1s)
    star_idx1s = star_idx1s[sorting_indexes]
    star_idx2s = star_idx2s[sorting_indexes]
    seps = seps[sorting_indexes]

    # Since we don't know ahead of time how many asterisms can be made from the close
    # stars, we cannot pre-allocate space for them. Instead, use a buffering strategy
    # and at the end merge the buffers and create an astropy.Table using them.
    buffer = _create_asterisms_buffer()
    added_keys = {}

    N = len(star_idx1s)
    start_time = time.time()
    close_idxs = []
    close_seps = []
    for i in np.arange(N):
        if verbose and N > 10000 and (i+1) % 10000 == 0:
            print(f"{i+1}/{N}: {time.time() - start_time:.2f}s", flush=True)
            start_time = time.time()

        if min_stars <= 1 and star_idx1s[i] == star_idx2s[i]:
            # Add single star asterism
            asterism_star_indexes, _ = _sort_star_indexes(star_idx1s[i])
            _add_asterism(buffer, asterism_star_indexes, star_data, id_field, ra_field, dec_field, mag_field, max_separation, max_1ngs_distance)

        if max_stars == 1:
            continue

        if seps[i] > min_separation*u.arcsec: # also eliminates match with self
            close_idxs.append(star_idx2s[i])
            close_seps.append(seps[i])

        if len(close_idxs) > 0 and (i+1 == N or star_idx1s[i+1] != star_idx1s[i]):
            close_idxs = np.array(close_idxs)
            close_seps = np.array([s.to(u.arcsec).value for s in close_seps])

            # Sort by separation (not strickly necessary, but forces asterisms to be added in a deterministic order)
            sorted_close_idxs = close_idxs[np.argsort(close_seps)]

            # Add multiple star asterisms
            for j in np.arange(len(sorted_close_idxs)):
                asterism_star_indexes, asterism_key = _sort_star_indexes(star_idx1s[i], sorted_close_idxs[j])
                if min_stars <= 2 and asterism_key not in added_keys:
                    _add_asterism(buffer, asterism_star_indexes, star_data, id_field, ra_field, dec_field, mag_field, max_separation, max_1ngs_distance)
                    added_keys[asterism_key] = True

                if max_stars == 2:
                    continue

                for k in np.arange(j+1, len(sorted_close_idxs)):
                    asterism_star_indexes, asterism_key = _sort_star_indexes(star_idx1s[i], sorted_close_idxs[j], sorted_close_idxs[k])
                    if asterism_key not in added_keys:
                        _add_asterism(buffer, asterism_star_indexes, star_data, id_field, ra_field, dec_field, mag_field, max_separation, max_1ngs_distance)
                        added_keys[asterism_key] = True

            close_idxs = []
            close_seps = []

    if verbose and N > 10000:
        print(f"{N}/{N}: {time.time() - start_time:.2f}s", flush=True)

    # Merge buffered arrays
    _merge_asterism_buffers(buffer)

    # Create table using buffer
    asterisms = Table([
        np.arange(buffer.N)+1, buffer.key, buffer.ra, buffer.dec, buffer.num_stars,
        buffer.star1_idx, buffer.star1_id, buffer.star1_ra, buffer.star1_dec, buffer.star1_mag,
        buffer.star2_idx, buffer.star2_id, buffer.star2_ra, buffer.star2_dec, buffer.star2_mag,
        buffer.star3_idx, buffer.star3_id, buffer.star3_ra, buffer.star3_dec, buffer.star3_mag,
        buffer.radius, buffer.area, buffer.relarea, buffer.separation, buffer.relsep
    ], names=[
        'id', 'key', 'ra', 'dec', 'num_stars',
        'star1_idx', 'star1_id', 'star1_ra', 'star1_dec', 'star1_mag',
        'star2_idx', 'star2_id', 'star2_ra', 'star2_dec', 'star2_mag',
        'star3_idx', 'star3_id', 'star3_ra', 'star3_dec', 'star3_mag',
        'radius', 'area', 'relarea', 'separation', 'relsep'
    ])

    return asterisms

# pylint: disable=possibly-used-before-assignment
def _add_asterism(buffer, star_indexes, star_data, id_field, ra_field, dec_field, mag_field, max_separation, max_1ngs_distance):
    # area of equilateral triangle that fits within circle of radius max_separation/4 (i.e. middle of FOV)
    optimal_star_area = 3.0/4.0*np.sqrt(3.0)*np.power(max_separation/4, 2)

    num_stars = len(star_indexes)

    if num_stars == 0:
        raise IndexError('Minimum of 1 star required')

    if num_stars > 3:
        raise IndexError('Maximum of 3 stars supported')

    id1  = star_data[id_field][star_indexes[0]]
    ra1  = star_data[ra_field][star_indexes[0]]
    dec1 = star_data[dec_field][star_indexes[0]]
    m1   = star_data[mag_field][star_indexes[0]]
    ra1_rad = np.deg2rad(ra1)
    dec1_rad = np.deg2rad(dec1)
    star_ids = [id1]

    if num_stars >= 2:
        id2  = star_data[id_field][star_indexes[1]]
        ra2  = star_data[ra_field][star_indexes[1]]
        dec2 = star_data[dec_field][star_indexes[1]]
        m2   = star_data[mag_field][star_indexes[1]]
        ra2_rad  = np.deg2rad(ra2)
        dec2_rad = np.deg2rad(dec2)
        l1   = np.rad2deg(np.sqrt(((ra1_rad-ra2_rad)*np.cos(dec1_rad))**2 + (dec1_rad-dec2_rad)**2)) * 3600.0 # arcsec
        star_ids.append(id2)

    if num_stars == 3:
        id3  = star_data[id_field][star_indexes[2]]
        ra3  = star_data[ra_field][star_indexes[2]]
        dec3 = star_data[dec_field][star_indexes[2]]
        m3   = star_data[mag_field][star_indexes[2]]
        ra3_rad  = np.deg2rad(ra3)
        dec3_rad = np.deg2rad(dec3)
        l2   = np.rad2deg(np.sqrt(((ra2_rad-ra3_rad)*np.cos(dec2_rad))**2 + (dec2_rad-dec3_rad)**2)) * 3600.0 # arcsec
        l3   = np.rad2deg(np.sqrt(((ra3_rad-ra1_rad)*np.cos(dec3_rad))**2 + (dec3_rad-dec1_rad)**2)) * 3600.0 # arcsec
        star_ids.append(id3)

    star_ids = np.sort(star_ids)
    key = '-'.join([str(i) for i in star_ids])

    if num_stars == 1:
        centre_ra = ra1_rad
        centre_dec = dec1_rad
        separation = max_1ngs_distance*2
        radius = max_1ngs_distance
        area = 0.0
    elif num_stars == 2:
        centre_ra = np.mean([ra1_rad, ra2_rad])
        centre_dec = np.mean([dec1_rad, dec2_rad])
        separation = l1
        radius = l1/2
        area = 0.0
    else:
        # triangle centre
        centre_ra, centre_dec = get_triangle_incenter(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra3_rad, dec3_rad) # radians
        #centre_ra, centre_dec = get_triangle_centroid(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra3_rad, dec3_rad) # radians
        #centre_ra, centre_dec = get_triangle_circumcenter(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra3_rad, dec3_rad) # radians

        # check that star distances from centre are acceptable
        d1 = np.sqrt(((ra1_rad-centre_ra)*np.cos(centre_dec))**2 + (dec1_rad-centre_dec)**2) * 180.0 / np.pi * 3600.0 # arcsec
        d2 = np.sqrt(((ra2_rad-centre_ra)*np.cos(centre_dec))**2 + (dec2_rad-centre_dec)**2) * 180.0 / np.pi * 3600.0 # arcsec
        d3 = np.sqrt(((ra3_rad-centre_ra)*np.cos(centre_dec))**2 + (dec3_rad-centre_dec)**2) * 180.0 / np.pi * 3600.0 # arcsec

        if np.max([d1, d2, d3]) > max_separation/2:
            longest_length = np.max([l1, l2, l3])

            if l1 == longest_length:
                centre_ra, centre_dec = get_midpoint(ra1_rad, dec1_rad, ra2_rad, dec2_rad)
            elif l2 == longest_length:
                centre_ra, centre_dec = get_midpoint(ra2_rad, dec2_rad, ra3_rad, dec3_rad)
            else:
                centre_ra, centre_dec = get_midpoint(ra3_rad, dec3_rad, ra1_rad, dec1_rad)

            # check that star distances from centre are acceptable
            d1 = np.sqrt(((ra1_rad-centre_ra)*np.cos(centre_dec))**2 + (dec1_rad-centre_dec)**2) * 180.0 / np.pi * 3600.0 # arcsec
            d2 = np.sqrt(((ra2_rad-centre_ra)*np.cos(centre_dec))**2 + (dec2_rad-centre_dec)**2) * 180.0 / np.pi * 3600.0 # arcsec
            d3 = np.sqrt(((ra3_rad-centre_ra)*np.cos(centre_dec))**2 + (dec3_rad-centre_dec)**2) * 180.0 / np.pi * 3600.0 # arcsec

            if np.max([d1, d2, d3]) > max_separation/2:
                return

        mean_mag = np.mean([m1, m2, m3])
        max_diff = np.max(np.abs([m1, m2, m3] - mean_mag))
        num_neg_diff = np.sum([m1, m2, m3] - mean_mag < 0)

        if max_diff < 1.0:
            # Case 1: all stars are similar in magnitude
            # (use furthest star separation)
            separation = max(l1,l2,l3)
        elif num_neg_diff == 2:
            # Case 2: two stars are brighter than the third
            # (use distance between brightest stars)
            if m3 > m1 and m3 > m2:
                separation = l1
            elif m1 > m2 and m1 > m3:
                separation = l2
            else:
                separation = l3
        else:
            # Case 3: one star is much brighter than the others
            # (use distance from brightest to furthest other star)
            if m1 < m2 and m1 < m3:
                separation = max(l1,l3)
            elif m2 < m1 and m2 < m3:
                separation = max(l1,l2)
            else:
                separation = max(l2,l3)

        area = get_triangle_angular_area(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra3_rad, dec3_rad, centre_ra, centre_dec) * (180.0 / np.pi * 3600.0)**2 # arcsec^2

        radius = area / ((l1+l2+l3)/2) # see: https://en.wikibooks.org/wiki/Trigonometry/Circles_and_Triangles/The_Incircle#Calculating_the_radius
        # radius = (l1*l2*l3)/(4*area) # see: https://artofproblemsolving.com/wiki/index.php/Circumradius

    relative_separation = separation / max_separation
    relative_area = area / optimal_star_area

    if buffer.idx == -1 or (buffer.idx+1) >= buffer.max_size:
        _increment_asterisms_buffer(buffer)

    buffer.N += 1
    buffer.idx += 1
    buffer.key       [-1][buffer.idx] = key
    buffer.ra        [-1][buffer.idx] = np.rad2deg(centre_ra)
    buffer.dec       [-1][buffer.idx] = np.rad2deg(centre_dec)
    buffer.num_stars [-1][buffer.idx] = num_stars
    buffer.star1_idx [-1][buffer.idx] = star_indexes[0]
    buffer.star1_id  [-1][buffer.idx] = id1
    buffer.star1_ra  [-1][buffer.idx] = ra1
    buffer.star1_dec [-1][buffer.idx] = dec1
    buffer.star1_mag [-1][buffer.idx] = m1
    buffer.star2_idx [-1][buffer.idx] = star_indexes[1] if num_stars >= 2 else -1
    buffer.star2_id  [-1][buffer.idx] = id2 if num_stars >= 2 else -1
    buffer.star2_ra  [-1][buffer.idx] = ra2 if num_stars >= 2 else -1
    buffer.star2_dec [-1][buffer.idx] = dec2 if num_stars >= 2 else -1
    buffer.star2_mag [-1][buffer.idx] = m2 if num_stars >= 2 else -1
    buffer.star3_idx [-1][buffer.idx] = star_indexes[2] if num_stars >= 3 else -1
    buffer.star3_id  [-1][buffer.idx] = id3 if num_stars >= 3 else -1
    buffer.star3_ra  [-1][buffer.idx] = ra3 if num_stars >= 3 else -1
    buffer.star3_dec [-1][buffer.idx] = dec3 if num_stars >= 3 else -1
    buffer.star3_mag [-1][buffer.idx] = m3 if num_stars >= 3 else -1
    buffer.radius    [-1][buffer.idx] = radius
    buffer.area      [-1][buffer.idx] = area
    buffer.relarea   [-1][buffer.idx] = relative_area
    buffer.separation[-1][buffer.idx] = separation
    buffer.relsep    [-1][buffer.idx] = relative_separation

def _create_asterisms_buffer():
    buffer = StructType()
    buffer.N = 0
    buffer.max_size = 10000
    buffer.idx = -1
    buffer.key        = []
    buffer.ra         = []
    buffer.dec        = []
    buffer.num_stars  = []
    buffer.star1_idx  = []
    buffer.star1_id   = []
    buffer.star1_ra   = []
    buffer.star1_dec  = []
    buffer.star1_mag  = []
    buffer.star2_idx  = []
    buffer.star2_id   = []
    buffer.star2_ra   = []
    buffer.star2_dec  = []
    buffer.star2_mag  = []
    buffer.star3_idx  = []
    buffer.star3_id   = []
    buffer.star3_ra   = []
    buffer.star3_dec  = []
    buffer.star3_mag  = []
    buffer.radius     = []
    buffer.area       = []
    buffer.relarea    = []
    buffer.separation = []
    buffer.relsep     = []
    return buffer

def _increment_asterisms_buffer(buffer):
    buffer.key       .append(np.zeros((buffer.max_size), dtype='<U20'   ))
    buffer.ra        .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.dec       .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.num_stars .append(np.zeros((buffer.max_size), dtype=np.int_  ))
    buffer.star1_idx .append(np.zeros((buffer.max_size), dtype=np.int_  ))
    buffer.star1_id  .append(np.zeros((buffer.max_size), dtype=np.int_  ))
    buffer.star1_ra  .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star1_dec .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star1_mag .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star2_idx .append(np.zeros((buffer.max_size), dtype=np.int_  ))
    buffer.star2_id  .append(np.zeros((buffer.max_size), dtype=np.int_  ))
    buffer.star2_ra  .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star2_dec .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star2_mag .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star3_idx .append(np.zeros((buffer.max_size), dtype=np.int_  ))
    buffer.star3_id  .append(np.zeros((buffer.max_size), dtype=np.int_  ))
    buffer.star3_ra  .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star3_dec .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.star3_mag .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.radius    .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.area      .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.relarea   .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.separation.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.relsep    .append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.idx = -1

def _merge_asterism_buffers(buffer):
    fields = dir(buffer)
    for i in np.arange(len(fields)):
        if '__' in fields[i]:
            continue

        field = getattr(buffer, fields[i])
        if isinstance(field, list) and len(field) > 0:
            if len(field[-1]) == 0:
                field[-1] = []
            elif (buffer.idx+1) < len(field[-1]):
                field[-1].resize((buffer.idx+1), refcheck=False)

            merged_field = np.hstack(field)

            if len(merged_field) != buffer.N:
                raise AsterismException('Merged buffer not the expected length')

            setattr(buffer, fields[i], merged_field)

#endregion

#region Utilities

def _sort_star_indexes(star1_idx, star2_idx = None, star3_idx = None):
    if star3_idx is not None:
        star_indexes = [star1_idx, star2_idx, star3_idx]
    elif star2_idx is not None:
        star_indexes = [star1_idx, star2_idx]
    else:
        star_indexes = [star1_idx]

    star_indexes = np.sort(star_indexes)
    asterism_key = '-'.join([str(i) for i in star_indexes])

    return star_indexes, asterism_key

def _process_flat_2points(ra1, dec1, ra2, dec2, flat_func):
    ra0 = np.mean([ra1, ra2])
    dec0 = np.mean([dec1, dec2])

    ux, uy = flat_func(
        (ra1-ra0)*np.cos(dec0), (dec1-dec0),
        (ra2-ra0)*np.cos(dec0), (dec2-dec0)
    )

    return ux/np.cos(dec0) + ra0, uy + dec0

def _get_midpoint_flat(ax, ay, bx, by):
    ux = (ax + bx) / 2
    uy = (ay + by) / 2
    return ux, uy

def get_midpoint(ra1, dec1, ra2, dec2):
    return _process_flat_2points(ra1, dec1, ra2, dec2, _get_midpoint_flat)

def _process_flat_3points(ra1, dec1, ra2, dec2, ra3, dec3, flat_func):
    ra0 = np.mean([ra1, ra2, ra3])
    dec0 = np.mean([dec1, dec2, dec3])

    ux, uy = flat_func(
        (ra1-ra0)*np.cos(dec0), (dec1-dec0),
        (ra2-ra0)*np.cos(dec0), (dec2-dec0),
        (ra3-ra0)*np.cos(dec0), (dec3-dec0)
    )

    return ux/np.cos(dec0) + ra0, uy + dec0

def _get_triangle_centroid_flat(ax, ay, bx, by, cx, cy):
    ux = (ax + bx + cx) / 3
    uy = (ay + by + cy) / 3
    return ux, uy

def get_triangle_centroid(ra1, dec1, ra2, dec2, ra3, dec3):
    return _process_flat_3points(ra1, dec1, ra2, dec2, ra3, dec3, _get_triangle_centroid_flat)

def _get_triangle_circumcenter_flat(ax, ay, bx, by, cx, cy):
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy

def get_triangle_circumcenter(ra1, dec1, ra2, dec2, ra3, dec3):
    return _process_flat_3points(ra1, dec1, ra2, dec2, ra3, dec3, _get_triangle_circumcenter_flat)

def _get_triangle_incenter_flat(ax, ay, bx, by, cx, cy):
    d1 = np.sqrt((bx-ax)**2 + (by-ay)**2)
    d2 = np.sqrt((cx-bx)**2 + (cy-by)**2)
    d3 = np.sqrt((ax-cx)**2 + (ay-cy)**2)
    p  = d1 + d2 + d3
    ux = (d1*ax + d2*bx + d3*cx)/p
    uy = (d1*ay + d2*by + d3*cy)/p
    return ux, uy

def get_triangle_incenter(ra1, dec1, ra2, dec2, ra3, dec3):
    return _process_flat_3points(ra1, dec1, ra2, dec2, ra3, dec3, _get_triangle_incenter_flat)

def get_triangle_angular_area(ra1, dec1, ra2, dec2, ra3, dec3, centre_ra, centre_dec):
    dra1  = (ra1-centre_ra) * np.cos(centre_dec)
    ddec1 = dec1-centre_dec
    dra2  = (ra2-centre_ra) * np.cos(centre_dec)
    ddec2 = dec2-centre_dec
    dra3  = (ra3-centre_ra) * np.cos(centre_dec)
    ddec3 = dec3-centre_dec
    area  = 0.5 * np.abs(dra1*(ddec2-ddec3)+dra2*(ddec3-ddec1)+dra3*(ddec1-ddec2))
    return area

def get_perpendicular_distance_from_line(ra, dec, ra1, dec1, ra2, dec2, centre_ra, centre_dec):
    # perpendicular distance from line connecting stars [2 stars]
    # see: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    x0 = (ra   - centre_ra ) * np.cos(centre_dec)
    y0 =  dec  - centre_dec
    x1 = (ra1  - centre_ra ) * np.cos(centre_dec)
    y1 =  dec1 - centre_dec
    x2 = (ra2  - centre_ra ) * np.cos(centre_dec)
    y2 =  dec2 - centre_dec
    perpendicular_distance = np.abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))/np.sqrt((x2-x1)**2+(y2-y1)**2) * 180.0 / np.pi * 3600 # arcsec
    return perpendicular_distance

def is_close_to_line(ra, dec, ra1, dec1, ra2, dec2, centre_ra, centre_dec, params):
    line_distance = np.sqrt(((ra1-ra2)*np.cos(centre_dec))**2 + (dec1-dec2)**2) * 180.0 / np.pi * 3600.0 # arcsec
    distance1 = np.sqrt(((ra-ra1)*np.cos(centre_dec))**2 + (dec-dec1)**2) * 180.0 / np.pi * 3600.0 # arcsec
    distance2 = np.sqrt(((ra-ra2)*np.cos(centre_dec))**2 + (dec-dec2)**2) * 180.0 / np.pi * 3600.0 # arcsec

    perpendicular_distance = get_perpendicular_distance_from_line(ra, dec, ra1, dec1, ra2, dec2, centre_ra, centre_dec)

    return perpendicular_distance < params.ao_theta0 * params.ao_asterism_distance_multiple \
       and distance1 < line_distance \
       and distance2 < line_distance

def is_inside_triangle(ra, dec, ra1, dec1, ra2, dec2, ra3, dec3, centre_ra, centre_dec):
    area  = get_triangle_angular_area(ra1, dec1, ra2, dec2, ra3, dec3, centre_ra, centre_dec) # radians^2
    area1 = get_triangle_angular_area(ra , dec , ra2, dec2, ra3, dec3, centre_ra, centre_dec) # radians^2
    area2 = get_triangle_angular_area(ra1, dec1, ra , dec , ra3, dec3, centre_ra, centre_dec) # radians^2
    area3 = get_triangle_angular_area(ra1, dec1, ra2, dec2, ra , dec , centre_ra, centre_dec) # radians^2
    return area1 + area2 + area3 < area*1.001

def count_shared_stars(id1a, id2a, id3a, id1b, id2b, id3b):
    num_shared = 0

    if id1a == id1b and id1a is not None and id1b is not None:
        num_shared += 1
    if id1a == id2b and id1a is not None and id2b is not None:
        num_shared += 1
    if id1a == id3b and id1a is not None and id3b is not None:
        num_shared += 1
    if id2a == id1b and id2a is not None and id1b is not None:
        num_shared += 1
    if id2a == id2b and id2a is not None and id2b is not None:
        num_shared += 1
    if id2a == id3b and id2a is not None and id3b is not None:
        num_shared += 1
    if id3a == id1b and id3a is not None and id1b is not None:
        num_shared += 1
    if id3a == id2b and id3a is not None and id2b is not None:
        num_shared += 1
    if id3a == id3b and id3a is not None and id3b is not None:
        num_shared += 1

    return num_shared

def get_triangle_intersecting_area(ra1a, dec1a, ra2a, dec2a, ra3a, dec3a, ra1b, dec1b, ra2b, dec2b, ra3b, dec3b):
    mid_ra = np.mean([ra1a, ra2a, ra3a, ra1b, ra2b, ra3b])
    mid_dec = np.mean([dec1a, dec2a, dec3a, dec1b, dec2b, dec3b])

    x1a = (ra1a - mid_ra) * np.cos(mid_dec)
    y1a = dec1a - mid_dec
    x2a = (ra2a - mid_ra) * np.cos(mid_dec)
    y2a = dec2a - mid_dec
    x3a = (ra3a - mid_ra) * np.cos(mid_dec)
    y3a = dec3a - mid_dec

    x1b = (ra1b - mid_ra) * np.cos(mid_dec)
    y1b = dec1b - mid_dec
    x2b = (ra2b - mid_ra) * np.cos(mid_dec)
    y2b = dec2b - mid_dec
    x3b = (ra3b - mid_ra) * np.cos(mid_dec)
    y3b = dec3b - mid_dec

    triangle1 = Polygon([(x1a, y1a), (x2a, y2a), (x3a, y3a)])
    triangle2 = Polygon([(x1b, y1b), (x2b, y2b), (x3b, y3b)])

    intersection = triangle1.intersection(triangle2)
    if intersection.is_empty:
        return 0.0
    return intersection.area # radians^2

def get_circle_intersecting_area(ra1, dec1, r1, ra2, dec2, r2):
    mid_ra = np.mean([ra1, ra2])
    mid_dec = np.mean([dec1, dec2])

    x1 = (ra1 - mid_ra) * np.cos(mid_dec)
    y1 = dec1 - mid_dec
    x2 = (ra2 - mid_ra) * np.cos(mid_dec)
    y2 = dec2 - mid_dec

    circle1 = Point(x1, y1).buffer(r1)
    circle2 = Point(x2, y2).buffer(r2)

    intersection = circle1.intersection(circle2)
    if intersection.is_empty:
        return 0.0

    return intersection.area # radians^2

def get_circle_overlap_area(R1, R2, d):
    if d >= R1 + R2:
        return 0  # No overlap
    elif d <= abs(R1 - R2):
        return np.pi * min(R1, R2) ** 2  # One circle completely inside the other

    term1 = R1**2 * np.arccos((d**2 + R1**2 - R2**2) / (2 * d * R1))
    term2 = R2**2 * np.arccos((d**2 + R2**2 - R1**2) / (2 * d * R2))
    term3 = 0.5 * np.sqrt((-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2))

    return term1 + term2 - term3

#endregion
