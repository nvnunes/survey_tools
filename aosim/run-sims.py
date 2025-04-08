#!/usr/bin/env python
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-options.name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import argparse
from configparser import ConfigParser
import cProfile
import os
import sys
import time
import astropy.units as u
from joblib import Parallel, delayed
import numpy as np
import tessellate
import simulate

class StructType:
    pass

def warmup(n):
    from tiptop.tiptop import gpuEnabled # pylint: disable=import-outside-toplevel
    if gpuEnabled:
        print(f"Process {n} started: GPU enabled")
    else:
        print(f"Process {n} started: GPU disabled")

def run(options, i, r, theta):
    if options.num_sims > 1:
        sim_name = f"{options.name}_{i}"
    else:
        sim_name = options.name

    sim_start_time = time.time()

    if options.save_fits:
        sim_results, simulation = simulate.run_simulation(
            sim_name, options.ini_filename, options.wavelength, options.zenith_angle, options.seeing, r, theta,
            ee_size=options.ee_size, opt_type=options.opt_type, output_path=options.output_path,
            do_plot=False, verbose=options.verbosity>2, return_simulation=True
        )

        simulate.save_results_fits(options.output_path, options.name, sim_results, simulation)
    else:
        sim_results = simulate.run_simulation(
            sim_name, options.ini_filename, options.wavelength, options.zenith_angle, options.seeing, r, theta,
            ee_size=options.ee_size, opt_type=options.opt_type, output_path=options.output_path,
            do_plot=False, verbose=options.verbosity>2, return_simulation=False
        )

    if options.verbosity > 0 or options.num_sims == 1:
        elapsed_time = time.time() - sim_start_time
        num_digits = len(str(options.num_sims))
        if options.num_sims == 1:
            print(f"Simulation Time: {elapsed_time:.1f} sec")
        else:
            print(f"{str(i).rjust(num_digits)}/{options.num_sims}: Simulation Time: {elapsed_time:.1f} sec")

        if options.save_fits:
            print(f"Output File    : {sim_name}.fits")

    return sim_results

def main():
    ################################################################################
    # Read configuration
    ################################################################################

    debug = False

    options = StructType()
    if debug or os.getenv("DEBUG", "0") == "1" or bool(sys.gettrace()):
        options.test_case = 'MOAO' # 'GNAO', 'MOAO', 'MOAO-serial'
        options.ini_filename = 'GIRMOS.ini'
        options.grid_mode = 'square'
        options.wavelength = 1.650 * u.micron
        options.zenith_angle = 30.0 * u.deg
        options.seeing = 0.6 * u.arcsec
        options.ee_size = 100.0 * u.mas
        options.save_fits = False
        options.num_threads = 1
        options.profile = False
        options.verbosity = True
    else:
        parser = argparse.ArgumentParser(description="Run TIPTOP simulations.")
        parser.add_argument('--ini', type=str, default='GIRMOS.ini', help="INI file to use (default: 'GIRMOS.ini').")
        parser.add_argument('--grid', type=str, default='square', choices=['square','hex','origin'], help="Grid to use for science points (default: 'square').")
        parser.add_argument('--wavelength', type=float, default=None, help="Override the wavelength in microns from the INI file.")
        parser.add_argument('--zenith', type=float, default=None, help="Override the Zenith Angle in degrees from the INI file.")
        parser.add_argument('--seeing', type=float, default=None, help="Override the seeing in arcsec from the INI file.")
        parser.add_argument('--ee_size', type=float, default=100.0, help="Ensquared Energy size in mas (default: 100.0).")
        parser.add_argument('--save_fits', action='store_true', help="Save results in FITS format.")
        parser.add_argument('--threads', type=int, default=1, help="Number of threads to use (default: 1).")
        parser.add_argument('--profile', action='store_true', help="Enable profiling.")
        parser.add_argument('--load_test', action='store_true', help="Load test data instead of running a new simulation.")
        parser.add_argument('--verbose', type=int, nargs='?', const=1, default=0, help="Enable verbose output (default: 0). Use a number to set verbosity level.")
        parser.add_argument('test_case', type=str, choices=['GNAO', 'MOAO', 'MOAO-serial'], help="Specify the test case to run.")

        args = parser.parse_args()
        options.test_case = args.test_case
        options.ini_filename = args.ini
        options.grid_mode = args.grid
        options.wavelength = args.wavelength
        if options.wavelength is not None:
            options.wavelength = options.wavelength * u.micron
        options.zenith_angle = args.zenith
        if options.zenith_angle is not None:
            options.zenith_angle = options.zenith_angle * u.deg
        options.seeing = args.seeing
        if options.seeing is not None:
            options.seeing = options.seeing * u.arcsec
        options.ee_size = args.ee_size * u.mas
        options.save_fits = args.save_fits
        options.num_threads = args.threads
        options.profile = args.profile
        options.load_test = args.load_test
        options.verbosity = args.verbose

    if '-one' in options.test_case:
        options.grid_mode = 'origin'
        options.test_case = options.test_case.replace('-one', '')

    if '/' not in options.ini_filename:
        options.ini_filename = os.path.join(os.path.dirname(__file__), options.ini_filename)
    if not os.path.isfile(options.ini_filename):
        raise FileNotFoundError(f"INI file not found: {options.ini_filename}")

    if options.zenith_angle is None or options.seeing is None or options.wavelength is None:
        config = ConfigParser()
        config.optionxform = str # so keys are case-sensitive
        config.read(options.ini_filename)

        if options.wavelength is None:
            options.wavelength = float(config['sources_science']['Wavelength'].strip('[]')) * u.m
        if options.zenith_angle is None:
            options.zenith_angle = float(config['telescope']['ZenithAngle'].strip('[]')) * u.deg
        if options.seeing is None:
            options.seeing = float(config['atmosphere']['Seeing'].strip('[]')) * u.arcsec

    match options.test_case:
        case 'GNAO':
            options.opt_type = 'normal'
        case 'MOAO-serial':
            options.opt_type = 'normal'
        case 'MOAO':
            options.opt_type = 'MOAO'

    ################################################################################
    # Create a grid of points in the focal plane
    ################################################################################

    fov = 120.0 # arcsec
    radius = fov / 2.0

    match options.grid_mode:
        case 'square':
            N = 11
            x = np.linspace(-radius, radius, N)
            y = np.linspace(-radius, radius, N)
            grid_x, grid_y = np.meshgrid(x, y)
            r = np.sqrt(grid_x**2 + grid_y**2).flatten()
            theta = np.rad2deg(np.arctan2(grid_y, grid_x)).flatten()
        case 'hex':
            num_rings = 5
            points, _ = tessellate.tessellate_circle_with_hexagons(num_rings, radius)
            r, theta = tessellate.to_polar_coordinates(points)
            theta = np.rad2deg(theta)
        case _: # 'origin'
            r = np.array([0.0])
            theta = np.array([0.0])

    num_points = len(r)

    ################################################################################
    # Prepare paths
    ################################################################################

    options.name = f"{options.test_case}_{options.wavelength.to(u.nm).value:.0f}nm_{options.zenith_angle.to(u.deg).value:.0f}deg_{options.seeing.to(u.mas).value:.0f}mas_{options.grid_mode}{num_points}"

    options.output_path = '/home/keita/projects/TIPTOP_Output'
    if not os.path.isdir(options.output_path):
        options.output_path = '/Volumes/TIPTOP_Output'

    ################################################################################
    # Run simulation(s)
    ################################################################################

    if options.test_case == 'MOAO-serial' or options.load_test:
        options.num_sims = num_points
    else:
        options.num_sims = 1

    if options.num_sims == 1:
        if options.profile:
            pr = cProfile.profile()
            pr.enable()

        results = run(options, 0, r, theta)

        if options.profile:
            pr.disable()
            pr.dump_stats(f"{options.output_path}/{options.name}.prof")
    else:
        if options.num_threads > 1:
            parallel_pool = Parallel(n_jobs=options.num_threads, backend="loky")
            parallel_pool(delayed(warmup)(n) for n in range(options.num_threads))

        overall_start_time = time.time()

        if options.num_threads <= 1:
            all_results = []
            if options.load_test:
                for i in range(options.num_sims):
                    all_results.append(run(options, i, r, theta))
            else:
                for i in range(options.num_sims):
                    all_results.append(run(options, i, r[i], theta[i]))
        else:
            if options.load_test:
                all_results = parallel_pool(delayed(run)(options, i, r, theta) for i in range(options.num_sims))
            else:
                all_results = parallel_pool(delayed(run)(options, i, r[i], theta[i]) for i in range(options.num_sims))

        elapsed_time = time.time() - overall_start_time
        print(f"Overall Time: {elapsed_time:.1f} sec\n")

        if not options.load_test:
            results = all_results[0]
            results.r = np.array([result.r for result in all_results])
            results.theta = np.array([result.theta for result in all_results])
            results.psfs = np.concatenate([result.psfs for result in all_results], axis=0)
            results.sr = np.concatenate([result.sr for result in all_results], axis=0)
            results.fwhm = np.concatenate([result.fwhm for result in all_results], axis=0)
            results.ee = np.concatenate([result.ee for result in all_results], axis=0)

    if not options.load_test:
        results.grid_mode = options.grid_mode

        simulate.save_results(options.output_path, options.name, results)

main()

# TODO:
# 5. Pass pointing details NGS ra/dec, NGS mag
# 6. Save results of many sims in summary table
