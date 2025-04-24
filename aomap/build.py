#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import aomap
import argparse

####################################################################################
# Build Modes:
#     build  : continue building inner pixel data and calculating outer pixel values
#              (supports incremental building)
#     rebuild: rebuilds inner pixel data and recalculates outer pixel values
#     recalc : only recalculates outer pixel values using existing inner pixel data
####################################################################################

parser = argparse.ArgumentParser(description="Build AO map data.")
parser.add_argument('mode', nargs='?', default='build', choices=['build', 'rebuild', 'recalc'], help="Build mode: 'build', 'rebuild', or 'recalc'. Default is 'build'.")
parser.add_argument('--verbose', action='store_true', help="Enable verbose output.")

args = parser.parse_args()
mode = args.mode
verbose = args.verbose

config = aomap.read_config('config.yaml')
aomap.build_inner(config, mode=mode, verbose=verbose)
aomap.append_asterism_dust(config, mode=mode, verbose=verbose)
aomap.append_asterism_counts(config, mode=mode, verbose=verbose)
aomap.build_data(config, mode=mode, verbose=verbose)
aomap.build_survey_extent(config, verbose=verbose)
