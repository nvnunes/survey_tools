#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from configparser import ConfigParser
import os
import pickle
import astropy.units as u
from matplotlib import rc
import matplotlib.pyplot as plt
from mat73 import loadmat
import numpy as np
from scipy.interpolate import griddata, Rbf
from tiptop import __version__ as __tiptop_version__
from tiptop.tiptop import baseSimulation
import stats
import tessellate

rc("text", usetex=False)

class SimulateException(Exception):
    pass

# ------------------------------------------------------------------------
# MOAO:
# ------------------------------------------------------------------------
# 1. LTAO HO: run HO calcs optimizing on fixed set of points (grid within 85") to get NGS LTAO PSDs
# 2. MOAO HO: run HO calcs optimizing on individual science positions to get MOAO PSDs
# 3. LTAO LO: compute NGS PSFs and stats using NGS LTAO PSDs
# 4. MOAO LO: compute Ctot using NGS LTAO PSFs and science positions
# 5. MOAO LO: compute final PSFs using MOAO PSDs and Ctot
#
# ------------------------------------------------------------------------
# TIPTOP Algorithm:
# ------------------------------------------------------------------------
#
# self.cartSciencePointingCoords: (x,y) coords of science points <-- ini > sources_science > Zenith/Azimuth
# self.cartNGSCoords_field:       (x,y) coords of NGS <-- ini > sources_LO > Zenith/Azimuth
# self.LO_fluxes_field:           ini > sensor_LO > NumberPhotons
# self.LO_freqs_field:            ini > RTC > SensorFrameRate_LO
# self.NGS_fluxes_field:          self.LO_fluxes_field * self.LO_freqs_field
#
# [HO: COMMON]
# sim.fao.initComputations() with Stage 2 config
#   Tomographic reconstruction to get self.PSD for each science point optimizing on each science point (MOAO)
#
# [HO: COMMON]
# sim.fao.initComputations() with Stage 1 config
#   Tomographic reconstruction to get self.PSD for each NGS location optimizing on fixed set of points (LTAO)
#
# [LO: NGS SPECIFIC]
# self.ngsPSF()
#   psdNGS <-- self.PSD at NGS positions
#   nSA: ini > sensor_LO > NumberLenslets
#   pf = FourierUtils.pistonFilter(2*self.tel_radius/nSAi,k)
#   psdNGS[i] = psdNGS[i] * pf
#
#   psfLE_NGS = psdSetToPsfSet(psdNGS, maskLO,
#                              self.LO_wvl, nLO, self.sx, self.grid_diameter,
#                              self.freq_range, self.dk, nPixPSFLO,
#                              self.wvlMax, overSampLO,
#                              opdMap=self.opdMap)
#
#   self.NGS_SR_field:         psfLE_NGS Strehl Ratio
#   self.NGS_EE_field:         psfLE_NGS encircled energy
#   self.NGS_FWHM_mas_field:   psfLE_NGS FWHM
#   self.NGS_DL_FWHM_mas:      NGS Diffraction Limited FWHM (not currently used)
#
# [LO: NGS SPECIFIC]
# sim.Ctot = self.mLO.computeTotalResidualMatrix(
#                   self.cartSciencePointingCoords,
#                   self.cartNGSCoords_field, self.NGS_fluxes_field, self.LO_freqs_field,
#                   self.NGS_SR_field, self.NGS_EE_field, self.NGS_FWHM_mas_field,
#                   aNGS_FWHM_DL_mas = self.NGS_DL_FWHM_mas, doAll=True)
#
#   Ctot (residual correlation matrix?):
#       + Turbulence (stats computed using psfLE_NGS from self.ngsPSF)
#       + Noise (sensor)
#       + Aliasing (NGS DL FWHM) (not currently used)
#       + Wind Shake (not currently used)
#       + NGS Coords
#       + Science Coords
#
# finalPSF()
#   [HO: COMMON]
#   PSD_HO <-- self.PSD at science points
#   mask = self.fao.ao.tel.pupil
#   self.opdMap = None (not currently used)
#   psfLongExp = psdSetToPsfSet(PSD_HO, mask,
#                               self.wvl, self.N, self.sx, self.grid_diameter,
#                               self.freq_range, self.dk, self.nPixPSF,
#                               self.wvlMax, self.overSamp,
#                               opdMap=self.opdMap, padPSD=self.nWvl>1)
#
#   [LO: NGS SPECIFIC]
#   finalConvolution()
#       ellp = self.mLO.ellipsesFromCovMats(self.Ctot)
#       resSpec = residualToSpectrum(ellp, self.wvlRef, self.nPixPSF, 1/(self.nPixPSF * self.psInMas))
#       sim.results[i] = convolve(psfLongExp, with resSpec)
#
# sim.results[i].sampling is the final PSF at each science point
# ------------------------------------------------------------------------

def _create_config(base_config_filename, config_path, wavelength, zenith_angle, seeing,
                   r=0.0, theta=0.0, optimization_r=None, optimization_theta=None, remove_moao=False, config_name=None):

    if not isinstance(r, (list, np.ndarray)):
        r = [r]

    if not isinstance(theta, (list, np.ndarray)):
        theta = [theta]

    if config_name is None:
        config_name = 'tiptop'

    config = ConfigParser()
    config.optionxform = str # so keys are case-sensitive
    config.read(base_config_filename)

    config['telescope']['ZenithAngle'] = str(zenith_angle.to(u.deg).value)
    config['atmosphere']['Seeing'] = str(seeing.to(u.arcsec).value)
    config['sources_science']['Wavelength'] = f"[{wavelength.to(u.m).value:.3e}]"
    config['sources_science']['Zenith'] = '[' + ','.join(f"{n:.4f}" for n in r) + ']'
    config['sources_science']['Azimuth'] = '[' + ','.join(f"{n:.4f}" for n in theta) + ']'
    if optimization_r is not None and optimization_theta is not None:
        config['DM']['OptimizationZenith'] = '[' + ','.join(f"{n:.4f}" for n in optimization_r) + ']'
        config['DM']['OptimizationAzimuth'] = '[' + ','.join(f"{n:.4f}" for n in optimization_theta) + ']'
        config['DM']['OptimizationWeight'] = '[' + ','.join(f"{n:.1f}" for n in np.ones(len(optimization_r))) + ']'

    if remove_moao:
        for section in list(config.sections()):
            if section.endswith('_MOAO'):
                config.remove_section(section)

    config_filename = f"{config_path}/{config_name}.ini"
    with open(config_filename, 'w', encoding='utf-8') as configfile:
        config.write(configfile)

    return config_filename, config

def run_simulation(name, base_config_filename, wavelength, zenith_angle, seeing, r, theta,
                   optimization_r=None, optimization_theta=None, ee_size=100*u.mas, output_path='../output', do_plot=False, verbose=False):

    do_moao = 'moao' in name.lower()

    config_filename, config = _create_config(base_config_filename, output_path, wavelength, zenith_angle, seeing,
                                             r=r, theta=theta, optimization_r=optimization_r, optimization_theta=optimization_theta,
                                             remove_moao=not do_moao, config_name=name)

    path2param          = os.path.dirname(config_filename)
    parametersFile      = os.path.splitext(os.path.basename(config_filename))[0]
    outputDir           = output_path
    outputFile          = parametersFile
    doConvolve          = True  # if you want to use the natural convolution operation set to True
    getHoErrorBreakDown = False # If you want HO error breakdown set this to True.
    ensquaredEnergy     = True  # If you want ensquared energy instead of encircled energy set this to True.
    eeRadiusInMas       = ee_size.to(u.mas).value/2 # Radius used for the computation of ensquared energy (half the side of the square)

    results = {
        'wavelength': wavelength,
        'seeing': seeing,
        'zenith_angle': zenith_angle,
        'NGS_zd': np.fromstring(config['sources_LO']['Zenith'].strip('[]'), dtype=np.float64, sep=','),
        'NGS_az': np.fromstring(config['sources_LO']['Azimuth'].strip('[]'), dtype=np.float64, sep=','),
        'LGS_zd': np.fromstring(config['sources_HO']['Zenith'].strip('[]'), dtype=np.float64, sep=','),
        'LGS_az': np.fromstring(config['sources_HO']['Azimuth'].strip('[]'), dtype=np.float64, sep=','),
        'ee_size': ee_size,
        'r': r,
        'theta': theta
    }

    simulation = baseSimulation(
        path2param, parametersFile, outputDir, outputFile,
        doConvolve=doConvolve, getHoErrorBreakDown=getHoErrorBreakDown,
        ensquaredEnergy=ensquaredEnergy, eeRadiusInMas=eeRadiusInMas,
        doPlot=do_plot, verbose=verbose
    )

    simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)

    psfs, sr, fwhm, ee = stats.get_stats(simulation)

    results.update({
        'pixel_scale': simulation.psInMas,
        'psfs': psfs,
        'sr': sr,
        'fwhm': fwhm,
        'ee': ee
    })

    return results, simulation

def save_results(output_path, name, results, verbose=False):
    output_file = f"{output_path}/{name}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"Output File    : {name}.pkl")

def load_results(output_path_or_file_path, name=None):
    if os.path.isfile(output_path_or_file_path):
        output_file = output_path_or_file_path
    else:
        if name is None:
            raise SimulateException("Name must be provided if output_path is a directory")
        output_file = f"{output_path_or_file_path}/{name}.pkl"

    if not os.path.isfile(output_file):
        raise SimulateException(f"File {output_file} does not exist")

    with open(output_file, 'rb') as f:
        results = pickle.load(f)

    return results

def rearrange_matlab_psfs(psfs):
    """
    Rearranges PSFs from 2D or 3D format into a 3D array of shape (num_psfs, res, res).
    If input is 3D, combines along axis 0 using sum or mean.
    """
    if psfs.ndim == 3:
        if psfs.shape[2] == 1:
            psfs = psfs[0, :, :]  # squeeze out first axis
        else:
            method = "sum"
            if method == "sum":
                psfs = np.sum(psfs, axis=0)
            else:
                psfs = np.mean(psfs, axis=0)

    psf_resolution = psfs.shape[0]
    num_psfs = psfs.shape[1] // psf_resolution
    new_psfs = np.zeros((num_psfs, psf_resolution, psf_resolution))

    for i in range(num_psfs):
        new_psfs[i, :, :] = psfs[:, i * psf_resolution : (i + 1) * psf_resolution]

    return new_psfs

def load_matlab_results(output_path_or_file_path, name=None, recompute=False):
    if os.path.isfile(output_path_or_file_path):
        output_file = output_path_or_file_path
    else:
        if name is None:
            raise SimulateException("Name must be provided if output_path is a directory")

        output_file = f"{output_path_or_file_path}/{name}.mat"
        if not os.path.isfile(output_file):
            raise SimulateException(f"File not found: {output_file}")

    if not recompute:
        output_file = os.path.join(os.path.dirname(output_file), f"stats_{os.path.basename(output_file)}")

    matlab_results = loadmat(output_file)

    pixel_scale = matlab_results['parm']['pixelScale']*1000
    wavelength = matlab_results['parm']['sci']['wavelength']
    zenith_angle = matlab_results['parm']['atm']['zenithAngle']/np.pi*180
    ee_size = 100.0 * u.mas
    seeing = 0.6
    grid_mode = 'hex'

    ngs_zd = matlab_results['parm']['nGs']['zeTT'].flatten()
    ngs_az = matlab_results['parm']['nGs']['azTT'].flatten()
    ngs_mag = matlab_results['parm']['nGs']['TTmag'].flatten() - 1.26

    lgs_n = int(matlab_results['parm']['lGs']['n'])
    lgs_zd = matlab_results['parm']['lGs']['zenith']/np.pi*180*3600
    lgs_az = matlab_results['parm']['lGs']['azimuth']/np.pi*180
    lgs_mag = matlab_results['parm']['lGs']['magnitude']

    lgs_zd = np.repeat(lgs_zd, lgs_n)
    lgs_az = np.array([lgs_az + i*360/lgs_n for i in range(lgs_n)])
    lgs_mag = np.repeat(lgs_mag, lgs_n)

    r = matlab_results['parm']['sci']['RHO'].flatten()
    theta = np.rad2deg(matlab_results['parm']['sci']['TH'].flatten())

    results = {
        'grid_mode': grid_mode,
        'pixel_scale': pixel_scale,
        'wavelength': wavelength,
        'seeing': seeing,
        'zenith_angle': zenith_angle,
        'NGS_zd': ngs_zd,
        'NGS_az': ngs_az,
        'NGS_mag': ngs_mag,
        'LGS_zd': lgs_zd,
        'LGS_az': lgs_az,
        'LGS_mag': lgs_mag,
        'ee_size': ee_size,
        'r': r,
        'theta': theta
    }

    if recompute:
        psfs = rearrange_matlab_psfs(matlab_results['psfs'])

        tel_diameter = matlab_results['parm']['tel']['Dsupp']
        pupil_file = os.path.join(os.path.dirname(output_file), 'pupil.mat')
        if os.path.isfile(pupil_file):
            tel_pupil = loadmat(pupil_file)['pupil']
        else:
            raise SimulateException(f"File pupil file is missing: {pupil_file}")

        sr, fwhm, ee = stats.get_stats_matlab(psfs, tel_diameter, tel_pupil, wavelength, pixel_scale, ee_size)
    else:
        sr = matlab_results['sr']
        fwhm = matlab_results['fwhm'] * 1000.0
        ee = matlab_results['ee01']

    results.update({
        'sr': sr,
        'fwhm': fwhm,
        'ee': ee
    })

    return results

def format_contour_label(x):
    s = f"{x:.2f}"
    if s.endswith("0"):
        s = f"{x:.1f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

def plot_fov(results, plot_value='SR', fov=60.0*u.arcsec, contours=None, skip_smoothing=False, skip_contours=False, fixed_range=False, plot_points=False):
    N = len(results['r'])
    radius = fov.to(u.arcsec).value/2

    match results['grid_mode']:
        case 'square':
            mask = results['r'] <= radius
        case 'hex':
            mask = np.ones(N, dtype=bool)
        case _:
            raise SimulateException(f"Unsupported grid mode: {results['grid_mode']}")

    match plot_value:
        case 'SR':
            values = results['sr']
        case 'FWHM':
            values = results['fwhm']
        case 'EE':
            values = results['ee']

    values_mean = np.mean(values[mask])
    values_std = np.std(values[mask])
    values_pv = np.max(values[mask]) - np.min(values[mask])

    m = 200
    x = results['r'] * np.cos(np.deg2rad(results['theta']))
    y = results['r'] * np.sin(np.deg2rad(results['theta']))
    xi = np.linspace(-radius, radius, m)
    yi = np.linspace(-radius, radius, m)
    X, Y = np.meshgrid(xi, yi)
    if skip_smoothing:
        VALUES = griddata((x, y), values, (X, Y), method='nearest')
    else:
        VALUES = Rbf(x, y, values, function='cubic')(X, Y)
    MASK = np.ones(VALUES.shape, dtype=np.float64)
    MASK[np.sqrt(X**2 + Y**2) > radius] = np.nan

    vmin = None
    vmax = None

    match plot_value:
        case 'SR':
            title = 'SR'
            bkg_label = "Strehl Ratio"
            if fixed_range:
                vmin = 0.60
                vmax = 0.001
            cmap = 'plasma'
        case 'FWHM':
            title = 'FWHM'
            bkg_label = 'FWHM [mas]'
            if fixed_range:
                vmin = 40.0
                vmax = 140.0
            cmap = 'plasma_r'
        case 'EE':
            title = 'EE'
            bkg_label = f"EE [{results['ee_size'].to(u.mas).value:.0f} mas]"
            if fixed_range:
                vmin = 0.01
                vmax = 0.7
            cmap = 'plasma'

    fig, ax = plt.subplots(figsize=(5,5))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

    # Plot background
    if not skip_contours:
        if contours is not None:
            levels = contours
        else:
            levels = 8
        cn = ax.contour(xi, yi, VALUES, levels=levels, linewidths=0.5, colors='k')
        ax.clabel(cn, cn.levels, inline=True, fmt=format_contour_label, fontsize=8)
    im = ax.imshow(VALUES*MASK, extent=[xi[0], xi[-1], yi[0], yi[-1]], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.linspace(-radius, radius, 5))
    ax.set_yticks(np.linspace(-radius, radius, 5))
    ax.set_xlabel('["/Sky]')
    ax.set_ylabel('["/Sky]')
    ax.set_title(f"{title} Mean={values_mean:.2f}, Std={values_std:.2f}, PV={values_pv:.2f}", fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel(bkg_label)

    # Plot the circle boundary
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=4)
    ax.add_patch(circle)

    # Plot the NGS
    for i, (zd, az) in enumerate(zip(results['NGS_zd'], results['NGS_az'])):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        plt.scatter(x, y, marker=(5, 1), facecolor='red', edgecolors='k', s=100, linewidths=0.5)

    # Plot the LGS
    for i, (zd, az) in enumerate(zip(results['LGS_zd'], results['LGS_az'])):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        plt.scatter(x, y, marker=(5, 1), facecolor='yellow', edgecolor='k', s=200, linewidths=0.5)

    # Plot FWHM at each point
    if plot_points:
        for i, (zd, az) in enumerate(zip(results['r'], results['theta'])):
            x = zd * np.cos(np.deg2rad(az))
            y = zd * np.sin(np.deg2rad(az))
            fwhm = results['fwhm'][i] / 1000.0 # mas -> arcsec
            circle = plt.Circle((x, y), fwhm/2, fill=False, color='k', linewidth=1)
            ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    plt.show()

def plot_psf(results, index=0, zoom=None, skip_peak_norm=False, skip_cbar=False, fixed_range=False):

    # TODO: option to zoom in to 1.5 * FWHM
    # TODO: option to plot EE box

    cmap = 'hot'

    psf = results['psfs'][index]
    if not skip_peak_norm:
        psf = psf / np.max(psf)
    psf = np.log10(np.abs(psf))

    if 'pixel_scale' not in results:
        pixel_scale = 7.0 # HACK!!!
    else:
        pixel_scale = results['pixel_scale']

    Nx   = psf.shape[0]
    Ny   = psf.shape[1]
    xlim = [-Nx//2*pixel_scale, Nx//2*pixel_scale]
    ylim = [-Ny//2*pixel_scale, Ny//2*pixel_scale]
    x    = np.linspace(xlim[0], xlim[1], Nx)
    y    = np.linspace(ylim[0], ylim[1], Ny)

    if zoom is not None:
        if isinstance(zoom, u.Quantity):
            zoom_px = int(zoom.to(u.mas).value/pixel_scale)
        else:
            zoom_px = int(results['fwhm'][index]*zoom/pixel_scale/2)
        xlim = [-zoom_px*pixel_scale, zoom_px*pixel_scale]
        ylim = [-zoom_px*pixel_scale, zoom_px*pixel_scale]
        x    = x[Nx//2-zoom_px:Nx//2+zoom_px]
        y    = y[Ny//2-zoom_px:Ny//2+zoom_px]
        psf  = psf[Nx//2-zoom_px:Nx//2+zoom_px,Ny//2-zoom_px:Ny//2+zoom_px]

    if fixed_range:
        vmin = -4
        vmax = 0
    else:
        vmin = None
        vmax = None

    levels = [-5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1]

    fig, ax = plt.subplots(figsize=[5,5])
    im = ax.imshow(psf, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    cn = ax.contour(x, y, psf, levels=levels, colors='white', linewidths=0.5)
    ax.clabel(cn, cn.levels, inline=True, fmt=format_contour_label, fontsize=8)
    if not skip_cbar:
        cbar = fig.colorbar(im, ax=ax, format='%.1f')
        cbar.set_label(f"Log {'Relative ' if not skip_peak_norm else ''}Intensity")
    ax.set_xlabel('[mas/Sky]')
    ax.set_ylabel('[mas/Sky]')
    plt.title(f"Center: r={results['r'][index]:.1f}\", theta={results['theta'][index]:.1f}°\nSR={results['sr'][index]:.2f}, FWHM={results['fwhm'][index]:.0f} mas, EE{results['ee_size'].value:.0f}={results['ee'][index]:.2f}")
    plt.show()
