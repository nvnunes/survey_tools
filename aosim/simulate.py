#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from configparser import ConfigParser
import os
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tiptop.baseSimulation import baseSimulation # type: ignore # pylint: disable=import-error
from tiptop.tiptop import gpuSelect # type: ignore # pylint: disable=import-error
import tessellate

class StructType:
    pass

class SimulateException(Exception):
    pass

def _create_tmp_config(base_config_filename, tmp_path, wavelength, zenith_angle, seeing, zd=0.0, az=0.0):
    if not isinstance(zd, (list, np.ndarray)):
        zd = [zd]
    if not isinstance(az, (list, np.ndarray)):
        az = [az]

    config = ConfigParser()
    config.optionxform = str # so keys are case-sensitive
    config.read(base_config_filename)

    config['telescope']['ZenithAngle'] = str(zenith_angle.to(u.deg).value)
    config['atmosphere']['Seeing'] = str(seeing.to(u.arcsec).value)
    config['sources_science']['Wavelength'] = f"[{wavelength.to(u.m).value:.3e}]"
    config['sources_science']['Zenith'] = '[' + ','.join(f"{n:.1f}" for n in zd) + ']'
    config['sources_science']['Azimuth'] = '[' + ','.join(f"{n:.1f}" for n in az) + ']'

    tmp_config_filename = f"{tmp_path}/tiptop.ini"
    with open(tmp_config_filename, 'w', encoding='utf-8') as configfile:
        config.write(configfile)

    return tmp_config_filename, config

def run_simulation(config_filename, wavelength, zenith_angle, seeing, zd=None, az=None, points=None, cells=None, do_plot=False, verbose=False):
    output_path = '../output'

    if (zd is not None or az is not None) and points is not None:
        raise SimulateException("You cannot specify both 'zd/az' and 'points'.")

    if points is not None:
        zd, az = tessellate.to_polar_coordinates(points)
        az = np.rad2deg(az)

    tmp_config_filename, config = _create_tmp_config(config_filename, output_path, wavelength, zenith_angle, seeing, zd, az)

    gpuIndex            = 0     # Target GPU index where the simulation will be run
    path2param          = os.path.dirname(tmp_config_filename)
    parametersFile      = os.path.splitext(os.path.basename(tmp_config_filename))[0]
    outputDir           = output_path
    outputFile          = parametersFile
    doConvolve          = True  # if you want to use the natural convolution operation set to True
    getHoErrorBreakDown = False # If you want HO error breakdown set this to True.
    ensquaredEnergy     = False # If you want ensquared energy instead of encircled energy set this to True.
    eeRadiusInMas       = 100   # used together with returnMetrics, radius used for the computation of the encirlced energy (if ensquaredEnergy is selected, this is half the side of the square)

    results = StructType()
    results.wavelength = wavelength
    results.seeing = seeing
    results.zenith_angle = zenith_angle
    results.fov = float(config['telescope']['TechnicalFoV'])
    results.zd = zd
    results.az = az
    results.points = points
    results.cells = cells
    results.NGS_zd = np.fromstring(config['sources_LO']['Zenith' ].strip('[]'), dtype=np.float64, sep=',')
    results.NGS_az = np.fromstring(config['sources_LO']['Azimuth'].strip('[]'), dtype=np.float64, sep=',')
    results.LGS_zd = np.fromstring(config['sources_HO']['Zenith' ].strip('[]'), dtype=np.float64, sep=',')
    results.LGS_az = np.fromstring(config['sources_HO']['Azimuth'].strip('[]'), dtype=np.float64, sep=',')
    results.ee_radius = eeRadiusInMas

    gpuSelect(gpuIndex)

    simulation = baseSimulation(
        path2param, parametersFile, outputDir, outputFile,
        doConvolve=doConvolve, getHoErrorBreakDown=getHoErrorBreakDown, ensquaredEnergy=ensquaredEnergy, eeRadiusInMas=eeRadiusInMas,
        doPlot=do_plot, verbose=verbose
    )

    simulation.doOverallSimulation()

    results.res_HO = simulation.HO_res
    if simulation.LOisOn:
        results.res_LO = simulation.LO_res
    else:
        results.res_LO = None

    simulation.computeMetrics()

    results.sr = simulation.sr
    results.fwhm = simulation.fwhm
    results.ee = simulation.ee

    # Save the PSF profile in a json file
    # simulation.savePSFprofileJSON()

    # Save PSD in the output fits
    # simulation.savePSDs = True
    # simulation.addSrAndFwhm = True
    # simulation.saveResults()

    return results

def plot_fov(results, bkg='EE', plot_fwhm_points=False):
    radius = results.fov/2

    match bkg:
        case 'EE':
            bkg_label = f"EE [{results.ee_radius} mas]"
            vmin = 0.0
            vmax = 1.0
        case 'SR':
            bkg_label = "SR"
            vmin = 0.0
            vmax = 1.0
        case 'FWHM':
            bkg_label = 'FWHM [mas]'
            vmin = 100.0 # min(results.fwhm)
            vmax = 300.0 # max(results.fwhm)

    _, ax = plt.subplots(figsize=(6,6))


    # Plot colormesh of EE
    if results.cells is not None:
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        for i, poly in enumerate(results.cells):
            x, y = poly.exterior.xy
            match bkg:
                case 'EE':
                    value = results.ee[i]
                case 'SR':
                    value = results.sr[i]
                case 'FWHM':
                    value = results.fwhm[i]
            ax.fill(x, y, color=cmap(norm(value)))

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=bkg_label)
    else:
        if results.points is not None:
            x = results.points[:, 0]
            y = results.points[:, 1]
        else:
            x = results.zd * np.cos(np.deg2rad(results.az))
            y = results.zd * np.sin(np.deg2rad(results.az))

        match bkg:
            case 'EE':
                values = results.ee
            case 'SR':
                values = results.sr
            case 'FWHM':
                values = results.fwhm

        X, Y = np.meshgrid(x, y)
        VALUES = griddata((x, y), values, (X, Y), method='linear')

        im = ax.pcolormesh(X, Y, VALUES, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=bkg_label)

    # Plot the circle boundary
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    # Plot the NGS
    for i, (zd, az) in enumerate(zip(results.NGS_zd, results.NGS_az)):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        circle = plt.Circle((x, y), 2, fill=True, color='red', linewidth=1)
        ax.add_patch(circle)

    # Plot the LGS
    for i, (zd, az) in enumerate(zip(results.LGS_zd, results.LGS_az)):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        circle = plt.Circle((x, y), 4, fill=True, color='orange', linewidth=1)
        ax.add_patch(circle)

    # Plot FWHM at each point
    if plot_fwhm_points:
        for i, (zd, az) in enumerate(zip(results.zd, results.az)):
            x = zd * np.cos(np.deg2rad(az))
            y = zd * np.sin(np.deg2rad(az))
            fwhm = results.fwhm[i] / 1000.0 # mas -> arcsec
            circle = plt.Circle((x, y), fwhm/2, fill=False, color='red', linewidth=1)
            ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(-radius*1.1, radius*1.1)
    ax.set_ylim(-radius*1.1, radius*1.1)

    plt.title(f"AO Performance (N={len(results.zd)})")
    plt.show()
