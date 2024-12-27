#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from aomap import Maps

####################################################################################
# Build Modes:
#     build  : continue building inner pixel data and calculating outer pixel values
#              (supports incremental building)
#     rebuild: rebuilds inner pixel data and recalculates outer pixel values
#     recalc : only recalculates outer pixel values using existing inner pixel data
####################################################################################

config_filename = 'config.yaml'
mode = 'build' # build, rebuild, recalc
verbose = True

built = Maps.build(config_filename, mode=mode, verbose=verbose)
if built:
    Maps.build_extra_map_levels(config_filename, verbose=verbose)
