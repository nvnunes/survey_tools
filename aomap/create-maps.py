#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from aomap import Maps

####################################################################################
# Build Modes:
#     build  : iterate over pixels in outer pixel FITS file that are NaN and
#              so incremental building is supported
#     rebuild: recreates outer pixel FITS files, builds all inner pixel data
#              and recalculates outer pixel values
#     recalc : recreates outer pixel FITS files and uses existing inner pixel
#              tables (inner.fits) to recalculate outer pixel values
#     reload : recreates outer pixel FITS files and reads outer pixel values from
#              previously computed outer pixel files (values.pkl)
####################################################################################

mode = 'build' # build, rebuild, recalc, reload

maps = Maps.load('config.yaml', mode=mode, verbose=True)
maps.build(mode=mode, verbose=True)
maps.close()
