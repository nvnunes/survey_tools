#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from configparser import ConfigParser
from datetime import datetime
import os
import pickle
from astropy.io import fits
import astropy.units as u
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, Rbf
from mastsel.mavisLO import MavisLO
from mastsel.mavisPsf import Field, zeroPad
from mastsel.mavisUtilities import congrid
from p3.aoSystem.fourierModel import fourierModel
from tiptop import __version__ as __tiptop_version__
from tiptop.tiptop import cpuArray, baseSimulation
from tiptop.tiptopUtils import arrayP3toMastsel
import stats

rc("text", usetex=False)

class StructType:
    pass

class SimulateException(Exception):
    pass

def _create_config(base_config_filename, config_path, wavelength, zenith_angle, seeing, opt_type, r=0.0, theta=0.0, config_name=None):
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
    config['sources_science']['Zenith'] = '[' + ','.join(f"{n:.5f}" for n in r) + ']'
    config['sources_science']['Azimuth'] = '[' + ','.join(f"{n:.5f}" for n in theta) + ']'
    config['DM']['OptimizationType'] = f"'{opt_type}'"
    config['DM']['OptimizationZenith'] = '[' + ','.join(f"{n:.5f}" for n in r) + ']'
    config['DM']['OptimizationAzimuth'] = '[' + ','.join(f"{n:.5f}" for n in theta) + ']'
    config['DM']['OptimizationWeight'] = '[' + ','.join(f"{n:.1f}" for n in np.ones(len(r))) + ']'

    config_filename = f"{config_path}/{config_name}.ini"
    with open(config_filename, 'w', encoding='utf-8') as configfile:
        config.write(configfile)

    return config_filename, config

def run_simulation(name, base_config_filename, wavelength, zenith_angle, seeing, r, theta, ee_size=100*u.mas, opt_type='normal', output_path='../output', do_plot=False, verbose=False, return_simulation=False):
    config_filename, config = _create_config(base_config_filename, output_path, wavelength, zenith_angle, seeing, opt_type, r=r, theta=theta, config_name=name)

    path2param          = os.path.dirname(config_filename)
    parametersFile      = os.path.splitext(os.path.basename(config_filename))[0]
    outputDir           = output_path
    outputFile          = parametersFile
    doConvolve          = True  # if you want to use the natural convolution operation set to True
    getHoErrorBreakDown = False # If you want HO error breakdown set this to True.
    ensquaredEnergy     = True  # If you want ensquared energy instead of encircled energy set this to True.
    eeRadiusInMas       = ee_size.to(u.mas).value/2 # Radius used for the computation of ensquared energy (half the side of the square)

    results = StructType()
    results.wavelength = wavelength
    results.seeing = seeing
    results.zenith_angle = zenith_angle
    results.NGS_zd = np.fromstring(config['sources_LO']['Zenith' ].strip('[]'), dtype=np.float64, sep=',')
    results.NGS_az = np.fromstring(config['sources_LO']['Azimuth'].strip('[]'), dtype=np.float64, sep=',')
    results.LGS_zd = np.fromstring(config['sources_HO']['Zenith' ].strip('[]'), dtype=np.float64, sep=',')
    results.LGS_az = np.fromstring(config['sources_HO']['Azimuth'].strip('[]'), dtype=np.float64, sep=',')
    results.ee_size = ee_size
    results.r = r
    results.theta = theta

    simulation = baseSimulation(
        path2param, parametersFile, outputDir, outputFile,
        doConvolve=doConvolve, getHoErrorBreakDown=getHoErrorBreakDown,
        ensquaredEnergy=ensquaredEnergy, eeRadiusInMas=eeRadiusInMas,
        doPlot=do_plot, verbose=verbose
    )

    _run_tiptop(simulation)

    results.psfs = np.array([cpuArray(img.sampling) for img in simulation.results])

    results.sr, results.fwhm, results.ee = stats.get_stats(
        results.psfs,
        simulation.wvl[0],
        simulation.tel_radius,
        simulation.psInMas,
        simulation.fao.ao.tel.pupil,
        simulation.eeRadiusInMas
    )

    if return_simulation:
        return results, simulation
    else:
        return results

def _run_tiptop(sim: baseSimulation):

    if sim.LOisOn:
        sim.configLO()

    sim.results = []

    # ------------------------------------------------------------------------
    ## HO Part with P3 PSDs
    # ------------------------------------------------------------------------

    sim.fao = fourierModel( sim.fullPathFilename, calcPSF=False, verbose=sim.verbose
                        , display=False, getPSDatNGSpositions=sim.LOisOn
                        , computeFocalAnisoCov=False, TiltFilter=sim.LOisOn
                        , getErrorBreakDown=sim.getHoErrorBreakDown, doComputations=False
                        , psdExpansion=True)

    if 'sensor_LO' in sim.my_data_map.keys():
        sim.fao.my_data_map['sensor_LO']['NumberPhotons'] = sim.my_data_map['sensor_LO']['NumberPhotons']
        sim.fao.ao.my_data_map['sensor_LO']['NumberPhotons'] = sim.my_data_map['sensor_LO']['NumberPhotons']
    if 'sources_LO' in sim.my_data_map.keys():
        sim.fao.my_data_map['sources_LO'] = sim.my_data_map['sources_LO']
        sim.fao.ao.my_data_map['sources_LO'] = sim.my_data_map['sources_LO']
        sim.fao.ao.configLOsensor()
        sim.fao.ao.configLO()
        sim.fao.ao.configLO_SC()

    sim.fao.initComputations()

    # High-order PSD caculations at the science directions and NGSs directions
    sim.PSD           = sim.fao.PSD # in nm^2
    sim.PSD           = sim.PSD.transpose()
    sim.N             = sim.PSD[0].shape[0]
    sim.nPointings    = sim.pointings.shape[1]
    sim.nPixPSF       = sim.my_data_map['sensor_science']['FieldOfView']
    sim.overSamp      = int(sim.fao.freq.kRef_)
    sim.freq_range    = sim.N*sim.fao.freq.PSDstep
    sim.pitch         = 1/sim.freq_range
    sim.grid_diameter = sim.pitch*sim.N
    sim.sx            = int(2*np.round(sim.tel_radius/sim.pitch))
    # dk is the same as in p3.aoSystem.powerSpectrumDensity except that it is multiplied by 1e9 instead of 2.
    sim.dk            = 1e9*sim.fao.freq.kcMax_/sim.fao.freq.resAO
    # wvlRef from P3 is required to scale correctly the OL PSD from rad to m
    sim.wvlRef        = sim.fao.freq.wvlRef
    # Define the pupil shape
    sim.mask = Field(sim.wvlRef, sim.N, sim.grid_diameter)
    sim.mask.sampling = congrid(arrayP3toMastsel(sim.fao.ao.tel.pupil), [sim.sx, sim.sx])
    sim.mask.sampling = zeroPad(sim.mask.sampling, (sim.N-sim.sx)//2)
    # error messages for wrong pixel size
    if sim.psInMas != cpuArray(sim.fao.freq.psInMas[0]):
        raise ValueError(f"sensor_science.PixelScale, '{sim.psInMas}', is different from self.fao.freq.psInMas,'{cpuArray(sim.fao.freq.psInMas)}'")

    if sim.fao.ao.tel.opdMap_on is not None:
        sim.opdMap = arrayP3toMastsel(sim.fao.ao.tel.opdMap_on)
    else:
        sim.opdMap = None

    # ----------------------------------------------------------------------------
    ## optional LO part
    # ----------------------------------------------------------------------------

    if sim.LOisOn:
        # ------------------------------------------------------------------------
        # --- NGS PSDs, PSFs and merit functions on PSFs
        # ------------------------------------------------------------------------
        sim.ngsPSF()

        # ------------------------------------------------------------------------
        # --- initialize MASTSEL MavisLO object
        # ------------------------------------------------------------------------
        sim.mLO = MavisLO(sim.path, sim.parametersFile, verbose=sim.verbose)

        # ------------------------------------------------------------------------
        ## total covariance matrix Ctot
        # ------------------------------------------------------------------------
        sim.Ctot          = sim.mLO.computeTotalResidualMatrix(np.array(sim.cartSciencePointingCoords),
                                                                    sim.cartNGSCoords_field, sim.NGS_fluxes_field,
                                                                    sim.LO_freqs_field,
                                                                    sim.NGS_SR_field, sim.NGS_EE_field, sim.NGS_FWHM_mas_field,
                                                                    aNGS_FWHM_DL_mas = sim.NGS_DL_FWHM_mas, doAll=True)

        # ------------------------------------------------------------------------
        # --- optional total focus covariance matrix Ctot
        # ------------------------------------------------------------------------
        if sim.addFocusError:
            # compute focus error
            sim.CtotFocus = sim.mLO.computeFocusTotalResidualMatrix(sim.cartNGSCoords_field, sim.Focus_fluxes_field,
                                                                    sim.Focus_freqs_field, sim.Focus_SR_field,
                                                                    sim.Focus_EE_field, sim.Focus_FWHM_mas_field)

            sim.GF_res = np.sqrt(max(sim.CtotFocus[0], 0))
            # add focus error to PSD using P3 FocusFilter
            FocusFilter = sim.fao.FocusFilter()
            FocusFilter *= 1/FocusFilter.sum()
            for PSDho in sim.PSD:
                PSDho += sim.GF_res**2 * FocusFilter
            sim.GFinPSD = True

    # ------------------------------------------------------------------------
    ## computation of the LO error (this changes for each asterism)
    # ------------------------------------------------------------------------
    sim.LO_res = np.sqrt(np.trace(sim.Ctot,axis1=1,axis2=2))

    # ------------------------------------------------------------------------
    # final PSF computation
    # ------------------------------------------------------------------------
    sim.finalPSF()

    # ------------------------------------------------------------------------
    # final results
    # ------------------------------------------------------------------------
    sim.computeOL_PSD()
    sim.computeDL_PSD()
    sim.cubeResults = []
    cubeResultsArray = []
    for i in range(sim.nWvl):
        if sim.nWvl>1:
            results = sim.results[i]
        else:
            results = sim.results
        cubeResults = []
        for img in results:
            cubeResults.append(cpuArray(img.sampling))
        if sim.nWvl>1:
            sim.cubeResults.append(cubeResults)
            cubeResultsArray.append(np.array(cubeResults))
        else:
            sim.cubeResults = cubeResults
            cubeResultsArray = cubeResults
        sim.cubeResultsArray = np.array(cubeResultsArray)
    sim.computePSF1D()

MAX_VALUE_CHARS = 80
APPEND_TOKEN = '&&&'

def add_hdr_keyword(hdr, key_primary, key_secondary, val, iii=None, jjj=None):
    '''
    This functions add an element of the parmaters dictionary into the fits file header
    '''
    val_string = str(val)
    key = 'HIERARCH '+ key_primary +' '+ key_secondary
    if iii != None:
        key += ' '+str(iii)
    if jjj != None:
        key += ' '+str(jjj)
    margin = 4
    key = key
    current_val_string = val_string
    if len(key) + margin > MAX_VALUE_CHARS:
        print("Error, keywork is not acceptable due to string length.")
        return
    while not len(key) + 1 + len(current_val_string) + margin < MAX_VALUE_CHARS:
        max_char_index = MAX_VALUE_CHARS-len(key)-1-len(APPEND_TOKEN)-margin
        hdr[key+'+'] = current_val_string[:max_char_index]+APPEND_TOKEN        
        current_val_string = current_val_string[max_char_index:]        
    hdr[key] = current_val_string

def save_results_fits(output_path, name, results, simulation):
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=cpuArray(simulation.cubeResultsArray)))
    hdul.append(fits.ImageHDU(data=cpuArray(simulation.psfOL.sampling))) # append open-loop PSF
    hdul.append(fits.ImageHDU(data=cpuArray(simulation.psfDL.sampling))) # append diffraction limited PSF
    hdul.append(fits.ImageHDU(data=cpuArray(simulation.PSD))) # append high order PSD
    hdul.append(fits.ImageHDU(data=cpuArray(simulation.psf1d_data))) # append radial profiles forthe final PSFs

    now = datetime.now()

    # header
    hdr0 = hdul[0].header
    hdr0['TIME'] = now.strftime("%Y%m%d_%H%M%S")
    hdr0['TIPTOP_V'] = __tiptop_version__

    # parameters in the header
    for key_primary in simulation.my_data_map:
        for key_secondary in simulation.my_data_map[key_primary]:
            temp = simulation.my_data_map[key_primary][key_secondary]
            if isinstance(temp, list):
                iii = 0
                for elem in temp:
                    if isinstance(elem, list):
                        jjj = 0
                        for elem2 in elem:
                            add_hdr_keyword(hdr0,key_primary,key_secondary,elem2,iii=str(iii),jjj=str(jjj))
                            jjj += 1
                    else:
                        add_hdr_keyword(hdr0,key_primary,key_secondary,elem,iii=str(iii))
                    iii += 1
            else:
                add_hdr_keyword(hdr0, key_primary,key_secondary,temp)

    # header of the PSFs
    hdr1 = hdul[1].header
    hdr1['TIME'] = now.strftime("%Y%m%d_%H%M%S")
    hdr1['CONTENT'] = "PSF CUBE"
    hdr1['SIZE'] = str(simulation.cubeResultsArray.shape)
    if simulation.nWvl>1:
        for i in range(simulation.nWvl):
            hdr1['WL_NM'+str(i).zfill(3)] = str(int(simulation.wvl[i]*1e9))
    else:
        hdr1['WL_NM'] = str(int(simulation.wvl[0]*1e9))
    hdr1['PIX_MAS'] = str(simulation.psInMas)
    hdr1['CC'] = "CARTESIAN COORD. IN ASEC OF THE "+str(simulation.pointings.shape[1])+" SOURCES"
    for i in range(simulation.pointings.shape[1]):
        hdr1['CCX' + str(i).zfill(4)] = np.round(simulation.pointings[0, i], 3).item()
        hdr1['CCY' + str(i).zfill(4)] = np.round(simulation.pointings[1, i], 3).item()
    if hasattr(simulation,'HO_res'):
        hdr1['RESH'] = "High Order residual in nm RMS"
        for i in range(simulation.HO_res.shape[0]):
            hdr1['RESH'+str(i).zfill(4)] =  np.round(cpuArray(simulation.HO_res[i]),3)
    if hasattr(simulation,'LO_res'):
        hdr1['RESL'] = "Low Order residual in nm RMS"
        for i in range(simulation.LO_res.shape[0]):
            hdr1['RESL'+str(i).zfill(4)] = np.round(cpuArray(simulation.LO_res[i]),3)
    if hasattr(simulation,'GF_res'):
        hdr1['RESF'] = "Global Focus residual in nm RMS (included in PSD)"
        hdr1['RESF0000'] = np.round(cpuArray(simulation.GF_res),3)
    for i in range(simulation.nWvl):
        if simulation.nWvl>1:
            cubeResultsArray = simulation.cubeResultsArray[i]
            wTxt = 'W'+str(i).zfill(2)
            fTxt = 'FW'
            eTxt = 'EE'
            Nfill = 2
        else:
            cubeResultsArray = simulation.cubeResultsArray
            wTxt = ''
            fTxt = 'FWHM'
            if simulation.eeRadiusInMas >= 100:
                eTxt = 'EE'+f"{simulation.eeRadiusInMas/1000:.1f}".replace('.','')
            else:
                eTxt = 'EE'+str(int(np.round(simulation.eeRadiusInMas)))
            Nfill = 4
        for j in range(cubeResultsArray.shape[0]):
            hdr1['SR'+str(j).zfill(Nfill)+wTxt] = float(np.round(results.sr[j],5))
        for j in range(cubeResultsArray.shape[0]):
            hdr1[fTxt+str(j).zfill(Nfill)+wTxt] = np.round(results.fwhm[j],3)
        for j in range(cubeResultsArray.shape[0]):
            hdr1[eTxt+str(j).zfill(Nfill)+wTxt] = np.round(results.ee[j],5)

    # header of the OPEN-LOOP PSF
    hdr2 = hdul[2].header
    hdr2['TIME'] = now.strftime("%Y%m%d_%H%M%S")
    hdr2['CONTENT'] = "OPEN-LOOP PSF"
    hdr2['SIZE'] = str(simulation.psfOL.sampling.shape)

    # header of the DIFFRACTION LIMITED PSF
    hdr3 = hdul[3].header
    hdr3['TIME'] = now.strftime("%Y%m%d_%H%M%S")
    hdr3['CONTENT'] = "DIFFRACTION LIMITED PSF"
    hdr3['SIZE'] = str(simulation.psfDL.sampling.shape)

    # header of the PSD
    hdr4 = hdul[4].header
    hdr4['TIME'] = now.strftime("%Y%m%d_%H%M%S")
    hdr4['CONTENT'] = "High Order PSD"
    hdr4['SIZE'] = str(simulation.PSD.shape)

    # header of the Total PSFs profiles
    hdr5 = hdul[5].header
    hdr5['TIME'] = now.strftime("%Y%m%d_%H%M%S")
    hdr5['CONTENT'] = "Final PSFs profiles"
    hdr5['SIZE'] = str(simulation.psf1d_data.shape)

    hdul.writeto(f"{output_path}/{name}.fits", overwrite=True)

def save_results(output_path, name, results):
    output_file = f"{output_path}/{name}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

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

def format_contour_label(x):
    s = f"{x:.2f}"
    if s.endswith("0"):
        s = f"{x:.1f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

def plot_fov(results, plot_values='SR', fov=60.0*u.arcsec, skip_interpolation=False, fixed_range=False, plot_points=False):
    N = len(results.r)
    radius = fov.to(u.arcsec).value/2

    match results.grid_mode:
        case 'square':
            mask = results.r <= radius
        case 'hex':
            mask = np.ones(N, dtype=bool)
        case _:
            raise SimulateException(f"Unsupported grid mode: {results.grid_mode}")

    match plot_values:
        case 'SR':
            values = results.sr
        case 'FWHM':
            values = results.fwhm
        case 'EE':
            values = results.ee

    values_mean = np.mean(values[mask])
    values_std = np.std(values[mask])
    values_range = np.max(values[mask]) - np.min(values[mask])

    m = 200
    x = results.r * np.cos(np.deg2rad(results.theta))
    y = results.r * np.sin(np.deg2rad(results.theta))
    xi = np.linspace(-radius, radius, m)
    yi = np.linspace(-radius, radius, m)
    X, Y = np.meshgrid(xi, yi)
    if skip_interpolation:
        VALUES = griddata((x, y), values, (X, Y), method='nearest')
    else:
        VALUES = Rbf(x, y, values, function='cubic')(X, Y)
    MASK = np.ones(VALUES.shape, dtype=np.float64)
    MASK[np.sqrt(X**2 + Y**2) > radius] = np.nan

    vmin = None
    vmax = None

    match plot_values:
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
                vmin = 60.0
                vmax = 160.0
            cmap = 'plasma_r'
        case 'EE':
            title = 'EE'
            bkg_label = f"EE [{results.ee_size.to(u.mas).value:.0f} mas]"
            if fixed_range:
                vmin = 0.3
                vmax = 0.7
            cmap = 'plasma'

    fig, ax = plt.subplots(figsize=(5,5))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

    # Plot background
    cn = ax.contour(xi, yi, VALUES, levels=8, linewidths=0.5, colors='k')
    im = ax.imshow(VALUES*MASK, extent=[xi[0], xi[-1], yi[0], yi[-1]], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.linspace(-radius, radius, 5))
    ax.set_yticks(np.linspace(-radius, radius, 5))
    ax.set_xlabel('["/Sky]')
    ax.set_ylabel('["/Sky]')
    ax.set_title(f"{title} Mean={values_mean:.2f}, Std={values_std:.2f}, Range={values_range:.2f}", fontweight='bold')
    ax.clabel(cn, cn.levels, inline=True, fmt=format_contour_label, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel(bkg_label)

    # Plot the circle boundary
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=4)
    ax.add_patch(circle)

    # Plot the NGS
    for i, (zd, az) in enumerate(zip(results.NGS_zd, results.NGS_az)):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        plt.scatter(x, y, marker=(5, 1), facecolor='red', edgecolors='k', s=100, linewidths=0.5)

    # Plot the LGS
    for i, (zd, az) in enumerate(zip(results.LGS_zd, results.LGS_az)):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        circle = plt.Circle((x, y), 4, fill=True, color='orange', linewidth=1)
        plt.scatter(x, y, marker=(5, 1), facecolor='yellow', edgecolor='k', s=200, linewidths=0.5)

    # Plot FWHM at each point
    if plot_points:
        for i, (zd, az) in enumerate(zip(results.r, results.theta)):
            x = zd * np.cos(np.deg2rad(az))
            y = zd * np.sin(np.deg2rad(az))
            fwhm = results.fwhm[i] / 1000.0 # mas -> arcsec
            circle = plt.Circle((x, y), fwhm/2, fill=False, color='k', linewidth=1)
            ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    plt.show()
