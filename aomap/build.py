#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import aomap
import argparse
import os

####################################################################################
# Build Modes:
#     build  : only do missing/incomplete work; promote downstream stages if inputs changed
#     rebuild: rebuilds everything unconditionally
#     recalc : recomputes everything downstream from existing source data
####################################################################################

parser = argparse.ArgumentParser(description="Build AO map data.")
parser.add_argument('mode', nargs='?', default='build', choices=['build', 'rebuild', 'recalc'], help="Build mode: 'build', 'rebuild', or 'recalc'. Default is 'build'.")
parser.add_argument('--verbose', action='store_true', help="Enable verbose output.")

args = parser.parse_args()
mode = args.mode
verbose = args.verbose

os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = aomap.read_config('config.yaml')

did_work = aomap.build_inner(config, mode=mode, force_reload_gaia=(mode == 'rebuild'), verbose=verbose)
if mode == 'build' and did_work:
    mode = 'recalc'

did_work = aomap.append_asterism_stats(config, mode=mode, verbose=verbose)
if mode == 'build' and did_work:
    mode = 'recalc'

did_work = aomap.build_data(config, mode=mode, verbose=verbose)
if mode == 'build' and did_work:
    mode = 'recalc'

aomap.build_survey_extent(config, mode=mode, verbose=verbose)
