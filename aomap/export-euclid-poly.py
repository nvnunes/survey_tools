#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import astropy.units as u
from mocpy import MOC
from astropy.table import Table
import numpy as np

####################################################################################################
# EWS
####################################################################################################

num_years = 6
mocs = []
for i in range(1, num_years+1):
    filename = f"../data//euclid/rsd2024a-footprint-equ-13-year{i}-MOC.fits"
    mocs.append(MOC.from_fits(filename))
moc = mocs[0].union(*mocs[1:])

level = 9
moc = moc.degrade_to_order(level)
moc_boundaries = moc.get_boundaries()

ews_boundaries = Table(names=['ra', 'dec', 'poly'], dtype=[np.float64, np.float64, np.int_])
for poly, border_coords in enumerate(moc_boundaries):
    for i in range(len(border_coords.ra)):
        ews_boundaries.add_row([border_coords[i].ra.to(u.deg).value, border_coords[i].dec.to(u.deg).value, poly+1])

np.savetxt('../data/surveys/ews-poly.txt', ews_boundaries, delimiter=' ', fmt=['%3.6f','%3.6f','%5d'])

####################################################################################################
# EDF
####################################################################################################

fields = [
    ['north', 'EuclidMOC_EDFN_rsd2024c_depth13_atLeast2visitsPlanned'],
    ['south', 'EuclidMOC_EDFS_rsd2024c_depth13_atLeast2visitsPlanned'],
    ['fornax', 'EuclidMOC_EDFF_rsd2024c_depth13_atLeast2visitsPlanned']
]

for field in fields:
    filename = f"../data//euclid/{field[1]}.fits"
    moc = MOC.from_fits(filename)

    level = 9
    moc = moc.degrade_to_order(level)
    moc_boundaries = moc.get_boundaries()

    edf_boundaries = Table(names=['ra', 'dec', 'poly'], dtype=[np.float64, np.float64, np.int_])
    for poly, border_coords in enumerate(moc_boundaries):
        for i in range(len(border_coords.ra)):
            edf_boundaries.add_row([border_coords[i].ra.to(u.deg).value, border_coords[i].dec.to(u.deg).value, poly+1])

    np.savetxt(f"../data/surveys/edf-{field[0]}-poly.txt", edf_boundaries, delimiter=' ', fmt=['%3.6f','%3.6f','%5d'])
