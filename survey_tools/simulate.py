#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from collections import namedtuple
from configparser import ConfigParser
import contextlib
from copy import deepcopy
import io
from IPython.display import display, HTML
import os
import pickle
from astropy.table import Table
import astropy.units as u
import h5py
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mat73 import loadmat
import numpy as np
from scipy.interpolate import griddata, Rbf
from tiptop import __version__ as __tiptop_version__
from tiptop.tiptop import baseSimulation
from survey_tools import aostats

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
                   zd=0.0, az=0.0, lgs=None, ngs=None, opt_zd=None, opt_az=None, extra_wfe=None, remove_moao=False, config_name=None):

    if not isinstance(zd, (list, np.ndarray)):
        zd = [zd]

    if not isinstance(az, (list, np.ndarray)):
        az = [az]

    if config_name is None:
        config_name = 'tiptop'

    config = ConfigParser()
    config.optionxform = str # so keys are case-sensitive
    config.read(base_config_filename)

    config['telescope']['ZenithAngle'] = str(zenith_angle.to(u.deg).value)
    if extra_wfe is not None and extra_wfe > 0:
        config['telescope']['extraErrorNm'] = str(extra_wfe)
        config['telescope']['extraErrorExp'] = "-2"
    config['atmosphere']['r0_Value'] = str(get_r0_from_seeing(seeing).to(u.m).value)
    # TODO: Set Cn2Weights based on seeing!
    config['sources_science']['Wavelength'] = f"[{wavelength.to(u.m).value:.3e}]"
    config['sources_science']['Zenith'] = '[' + ','.join(f"{n:.4f}" for n in zd) + ']'
    config['sources_science']['Azimuth'] = '[' + ','.join(f"{n:.4f}" for n in az) + ']'
    if ngs is not None:
        config['sources_LO']['Zenith'] = '[' + ','.join(f"{ngs[i]['zd']:.4f}" for i in range(len(ngs))) + ']'
        config['sources_LO']['Azimuth'] = '[' + ','.join(f"{ngs[i]['az']:.4f}" for i in range(len(ngs))) + ']'
        nph = [get_nph_from_magnitude(config, "R", ngs[i]['mag']) for i in range(len(ngs))]
        config['sensor_LO']['NumberPhotons'] = '[' + ','.join(f"{n:.0f}" for n in nph) + ']'
        config['sensor_LO']['NumberLenslets'] = '[' + ','.join(f"{n:.0f}" for n in split_ini_array(config['sensor_LO']['NumberLenslets'])[0]*np.ones(len(nph))) + ']'
    if lgs is not None:
        config['sources_HO']['Zenith'] = '[' + ','.join(f"{lgs[i]['zd']:.4f}" for i in range(len(lgs))) + ']'
        config['sources_HO']['Azimuth'] = '[' + ','.join(f"{lgs[i]['az']:.4f}" for i in range(len(lgs))) + ']'
        nph = [get_nph_from_magnitude(config, "Na", lgs[i]['mag']) for i in range(len(lgs))]
        config['sensor_HO']['NumberPhotons'] = '[' + ','.join(f"{n:.0f}" for n in nph) + ']'
        config['sensor_HO']['NumberLenslets'] = '[' + ','.join(f"{n:.0f}" for n in split_ini_array(config['sensor_HO']['NumberLenslets'])[0]*np.ones(len(nph))) + ']'
    if opt_zd is not None and opt_az is not None:
        config['DM']['OptimizationZenith'] = '[' + ','.join(f"{n:.4f}" for n in opt_zd) + ']'
        config['DM']['OptimizationAzimuth'] = '[' + ','.join(f"{n:.4f}" for n in opt_az) + ']'
        config['DM']['OptimizationWeight'] = '[' + ','.join(f"{n:.1f}" for n in np.ones(len(opt_zd))) + ']'

    if remove_moao:
        for section in list(config.sections()):
            if section.endswith('_MOAO'):
                config.remove_section(section)

    config_filename = f"{config_path}/{config_name}.ini"
    with open(config_filename, 'w', encoding='utf-8') as configfile:
        config.write(configfile)

    return config_filename, config

def run_simulation(name, base_config_filename, wavelength, zenith_angle, seeing, zd, az,
                   lgs=None, ngs=None, opt_zd=None, opt_az=None, ee_size=100*u.mas, extra_wfe=0.0, extra_vib=0.0,
                   output_path='../output', save_ini=False, index=0, do_plot=False, verbose=False):

    do_moao = 'moao' in name.lower()

    if save_ini:
        config_name = name
    else:
        config_name = f"tmp_{index}"

    config_filename, config = _create_config(base_config_filename, output_path, wavelength, zenith_angle, seeing,
                                             zd=zd, az=az, lgs=lgs, ngs=ngs, opt_zd=opt_zd, opt_az=opt_az, extra_wfe=extra_wfe,
                                             remove_moao=not do_moao, config_name=config_name)

    path2param          = os.path.dirname(config_filename)
    parametersFile      = os.path.splitext(os.path.basename(config_filename))[0]
    outputDir           = output_path
    outputFile          = parametersFile
    doConvolve          = True  # if you want to use the natural convolution operation set to True
    getHoErrorBreakDown = True  # If you want HO error breakdown set this to True.
    ensquaredEnergy     = True  # If you want ensquared energy instead of encircled energy set this to True.
    eeRadiusInMas       = ee_size.to(u.mas).value/2 # Radius used for the computation of ensquared energy (half the side of the square)

    results = {
        'wavelength': wavelength,
        'seeing': seeing,
        'zenith_angle': zenith_angle,
        'NGS_zd': split_ini_array(config['sources_LO']['Zenith']),
        'NGS_az': split_ini_array(config['sources_LO']['Azimuth']),
        'LGS_zd': split_ini_array(config['sources_HO']['Zenith']),
        'LGS_az': split_ini_array(config['sources_HO']['Azimuth']),
        'r': zd,
        'theta': az
    }

    try:
        simulation = baseSimulation(
            path2param, parametersFile, outputDir, outputFile,
            doConvolve=doConvolve, getHoErrorBreakDown=getHoErrorBreakDown,
            ensquaredEnergy=ensquaredEnergy, eeRadiusInMas=eeRadiusInMas,
            doPlot=do_plot, verbose=verbose
        )

        simulation.doOverallSimulation(skipMerit=True, skipPSF1D=True)

        results.update({
            'tel_diameter': simulation.tel_radius*2,
            'tel_pupil': simulation.fao.ao.tel.pupil, 
            'wavelength': simulation.wvl[0], 
            'pixel_scale': simulation.psInMas,
            'ee_size': simulation.eeRadiusInMas*2,
        })

        psfs = np.array([img.sampling for img in simulation.results])

        if extra_vib is not None and extra_vib > 0:
            aostats.add_extra_vibrations(psfs, extra_vib, results['pixel_scale'])

        sr, fwhm, ee = aostats.get_psf_stats(psfs, results['tel_diameter'], results['tel_pupil'], results['wavelength'], results['pixel_scale'], results['ee_size'])

        results.update({
            'psfs': aostats.cpuArray(psfs),
            'sr': sr,
            'fwhm': fwhm,
            'ee': ee,
        })

        results.update({
            'errors': SimulationErrors.get_from_simulation(simulation, extra_vib)
        })

    except Exception as e:
        results = None
        simulation = None

    if not save_ini:
        os.remove(config_filename)

    return results, simulation

def split_ini_array(s):
    return np.fromstring(s.strip('[]'), dtype=np.float64, sep=',')    

def get_r0_from_seeing(seeing, wavelength=0.5*u.micron):
    r0 = 0.98 * wavelength.to(u.m) / seeing.to(u.rad).value
    return r0

def get_nph_from_magnitude(config, band, mag):
    D = float(config['telescope']['TelescopeDiameter'])

    match band:
        case 'R':
            zp = 1.1e13/368
            mag += 1.26 # R -> I4b
            throughput = 0.42
            quantum_efficiency = 0.5 # unknown factor in OOMAO
            sampling_rate = float(config['RTC']['SensorFrameRate_LO'])
            n_lenslets = split_ini_array(config['sensor_LO']['NumberLenslets'])[0]
        case 'Na':
            zp = 3.3e12
            throughput = 0.45
            quantum_efficiency = 1.0 # not used
            sampling_rate = float(config['RTC']['SensorFrameRate_HO'])
            n_lenslets = split_ini_array(config['sensor_HO']['NumberLenslets'])[0]
        case _:
            raise SimulateException(f"Unsupported band: {band}")

    nph = zp * 10**(-0.4*mag) * throughput * quantum_efficiency / sampling_rate * (D/n_lenslets)**2
    return nph

class SimulationErrors(namedtuple('SimulationErrors', 
    [
        'wfeTot', 'wfeNCPA', 'wfeFit', 'wfeDiffRef', 'wfeChrom', 'wfeAl',
        'wfeN', 'wfeST', 'wfeWindShake', 'wfeJitter', 'wfeMcaoCone', 'wfeExtra',
        'wfeS', 'wfeR', 'wfeAni', 'wfeTomo', 'resHO', 'resLO', 'addVib'
    ]
    , defaults=[None] * 19
)):
    @staticmethod
    def get_from_simulation(simulation, extra_vib=None):
        N = simulation.fao.ao.src.nSrc-simulation.fao.ao.ngs.nSrc
        return SimulationErrors(
            wfeTot = simulation.fao.wfeTot[0:N],
            wfeNCPA = simulation.fao.wfeNCPA,
            wfeFit = simulation.fao.wfeFit,
            wfeDiffRef = simulation.fao.wfeDiffRef[0:N],
            wfeChrom = simulation.fao.wfeChrom[0:N],
            wfeAl = simulation.fao.wfeAl,
            wfeN = simulation.fao.wfeN[0:N],
            wfeST = simulation.fao.wfeST[0:N],
            wfeWindShake = simulation.fao.wfeWindShake,
            wfeJitter = simulation.fao.wfeJitter,
            wfeExtra = simulation.fao.wfeExtra,
            wfeS = simulation.fao.wfeS,
            wfeR = simulation.fao.wfeR,
            wfeAni = simulation.fao.wfeAni[0:N] if simulation.fao.nGs == 1 else 0.0,
            wfeTomo = simulation.fao.wfeTomo[0:N] if simulation.fao.nGs > 1 else 0.0,
            resHO = simulation.HO_res,
            resLO = simulation.LO_res,
            addVib = np.ones(N) * extra_vib if extra_vib is not None else 0.0,
        )
    
    def print(self, idx=None, label=None):
        text_width = 40

        if label is None:
            label = 'ERROR BREAKDOWN'

        print('-'*text_width)
        print(label.center(40, ' '))
        print('-'*text_width)
        print('Residual wavefront error:\t%4.2fnm'%(self.wfeTot[idx] if np.size(self.wfeTot) > 1 else self.wfeTot))
        print('NCPA residual:\t\t\t%4.2fnm'%(self.wfeNCPA[idx] if np.size(self.wfeNCPA) > 1 else self.wfeNCPA))
        print('Fitting error:\t\t\t%4.2fnm'%(self.wfeFit[idx] if np.size(self.wfeFit) > 1 else self.wfeFit))
        print('Differential refraction:\t%4.2fnm'%(self.wfeDiffRef[idx] if np.size(self.wfeDiffRef) > 1 else self.wfeDiffRef))
        print('Chromatic error:\t\t%4.2fnm'%(self.wfeChrom[idx] if np.size(self.wfeChrom) > 1 else self.wfeChrom))
        print('Aliasing error:\t\t\t%4.2fnm'%(self.wfeAl[idx] if np.size(self.wfeAl) > 1 else self.wfeAl))
        print('Noise error:\t\t\t%4.2fnm'%(self.wfeN[idx] if np.size(self.wfeN) > 1 else self.wfeN))
        print('Spatio-temporal error:\t\t%4.2fnm'%(self.wfeST[idx] if np.size(self.wfeST) > 1 else self.wfeST))
        print('Wind-shake error:\t\t%4.2fnm'%(self.wfeWindShake[idx] if np.size(self.wfeWindShake) > 1 else self.wfeWindShake))
        print('Additionnal jitter:\t\t%4.2fnm'%(self.wfeJitter[idx] if np.size(self.wfeJitter) > 1 else self.wfeJitter))
        print('Extra error:\t\t\t%4.2fnm'%(self.wfeExtra[idx] if np.size(self.wfeExtra) > 1 else self.wfeExtra))
        print('-'*text_width)
        print('Sole servoLag error:\t\t%4.2fnm'%(self.wfeS[idx] if np.size(self.wfeS) > 1 else self.wfeS))
        print('Sole reconstruction error:\t%4.2fnm'%(self.wfeR[idx] if np.size(self.wfeR) > 1 else self.wfeR))
        print('-'*text_width)
        print('Sole tomographic error:\t\t%4.2fnm'%(self.wfeTomo[idx] if np.size(self.wfeTomo) > 1 else self.wfeTomo))
        print('-'*text_width)
        print('HO_res:\t\t\t\t%4.2fnm'%(self.resHO[idx] if np.size(self.resHO) > 1 else self.resHO))
        print('LO_res:\t\t\t\t%4.2fnm'%(self.resLO[idx] if np.size(self.resLO) > 1 else self.resLO))
        print('-'*text_width)
        if self.addVib is not None:
            print('Additional Vibrations (sigma):\t%4.1fmas'%(self.addVib[idx] if np.size(self.addVib) > 1 else self.addVib))
            print('-'*text_width)

    def to_array(self):
        N = len(self.wfeTot)
        values = np.zeros((N, len(SimulationErrors._fields)))

        for i, field in enumerate(SimulationErrors._fields):
            value = getattr(self, field)
            if np.size(value) == 1:
                values[:,i] = np.full(N, value)
            else:
                values[:,i] = value

        return values

    @staticmethod
    def from_array(array, field_map=None):
        errors = SimulationErrors()

        if field_map is None:
            field_map = SimulationErrors._fields

        updates = {}
        for field in SimulationErrors._fields:
            if field not in field_map:
                continue
            idx = field_map.index(field)
            updates[field] = array[:,idx]

        return errors._replace(**updates)

    @staticmethod
    def num_fields():
        return len(SimulationErrors._fields)
    
    @staticmethod
    def get_field_names():
        return SimulationErrors._fields

def get_sim_is_run(output_path, name, num_sims):
    output_file = f"{output_path}/{name}.pkl"
    data_file = f"{output_path}/h5/{name}_data.h5"
    if os.path.isfile(data_file):
        with h5py.File(data_file, "r") as f:
            return f['errors'][:,0,0] > 0
    else:
        return np.full((num_sims), os.path.isfile(output_file))

class ModuleRemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "simulate" or name == "SimulationErrors":
            return SimulationErrors
        return super().find_class(module, name)

def load_single_asterism(output_path_or_file_path, name=None, asterism_id=None):
    if os.path.isfile(output_path_or_file_path):
        output_file = output_path_or_file_path
        data_file = None
        name = os.path.splitext(os.path.basename(output_file))[0]
    else:
        if name is None:
            raise SimulateException("Name must be provided if output_path is a directory")
        output_file = f"{output_path_or_file_path}/{name}.pkl"
        data_file = f"{output_path_or_file_path}/h5/{name}_data.h5"

    if not os.path.isfile(output_file):
        raise SimulateException(f"File {output_file} does not exist")

    with open(output_file, 'rb') as f:
        results = ModuleRemappingUnpickler(f).load()

    results['name'] = name
    results['theta'][np.isclose(results['theta'], 360.0, atol=2e-4)] = 0.0
    results['x'] = results['r'] * np.cos(np.deg2rad(results['theta']))
    results['y'] = results['r'] * np.sin(np.deg2rad(results['theta']))
    results['x'][np.isclose(results['x'], 0, atol=2e-4)] = 0.0
    results['y'][np.isclose(results['y'], 0, atol=2e-4)] = 0.0

    if 'mode' not in results:
        if 'moao' in name.lower():
            results['mode'] = 'MOAO'
        elif 'ltao' in name.lower():
            results['mode'] = 'LTAO'
        else:
            results['mode'] = 'GLAO'

    if 'ngs' in results and len(results['ngs']) > 1:
        if asterism_id is None or asterism_id == 0:
            raise SimulateException("Asterism ID must be provided if there are multiple asterisms in the file")

        results['ngs'] = [results['ngs'][asterism_id-1]]
        if 'asterisms' in results:
            results['asterisms'] = [results['asterisms'][asterism_id-1]]

        if 'sr' in results and 'fwhm' in results and 'ee' in results:
            results['sr'] = results['sr'][asterism_id-1, :]
            results['fwhm'] = results['fwhm'][asterism_id-1, :]
            results['ee'] = results['ee'][asterism_id-1, :]
            if 'errors' in results:
                results['errors'] = SimulationErrors.from_array(results['errors'][asterism_id-1, :, :], results['error_fields'])
        else:
            if not os.path.isfile(data_file):
                raise SimulateException(f"File {data_file} does not exist")

            with h5py.File(data_file, "r") as f:
                if 'psfs' in f:
                    results['psfs'] = f['psfs'][asterism_id-1, :, :, :]
                results['sr'] = f['sr'][asterism_id-1, :]
                results['fwhm'] = f['fwhm'][asterism_id-1, :]
                results['ee'] = f['ee'][asterism_id-1, :]
                if 'errors' in f:
                    results['errors'] = SimulationErrors.from_array(f['errors'][asterism_id-1, :, :], results['error_fields'])

    return results

def load_asterism_stats(output_path, name):
    output_file = f"{output_path}/{name}.pkl"
    data_file = f"{output_path}/h5/{name}_data.h5"

    with open(output_file, 'rb') as f:
        results = ModuleRemappingUnpickler(f).load()

    results['name'] = name

    if 'mode' not in results:
        if 'moao' in name.lower():
            results['mode'] = 'MOAO'
        elif 'ltao' in name.lower():
            results['mode'] = 'LTAO'
        else:
            results['mode'] = 'GLAO'

    if 'sr' not in results or 'fwhm' not in results or 'ee' not in results:
        if not os.path.isfile(data_file):
            raise SimulateException(f"File {data_file} does not exist")

        with h5py.File(data_file, "r") as f:
            results['sr'] = np.array(f['sr'])
            results['fwhm'] = np.array(f['fwhm'])
            results['ee'] = np.array(f['ee'])

    results['asterism_r'] = np.array([asterism['r'] for asterism in results['asterisms']])
    results['asterism_theta'] = np.array([asterism['theta'] for asterism in results['asterisms']])
    results['asterism_area'] = np.array([asterism['area'] for asterism in results['asterisms']])
    results['asterism_scale'] = np.array([asterism['scale'] for asterism in results['asterisms']])
    results['asterism_score'] = np.array([asterism['score'] for asterism in results['asterisms']])

    results['ngs_mags'] = np.array([[star['mag'] for star in ngs] for ngs in results['ngs']])

    results['ngs_num_bright'] = np.sum(np.array([[star['mag'] < 16 for star in ngs] for ngs in results['ngs']]), axis=1)
    results['ngs_num_nominal'] = np.sum(np.array([[star['mag'] >= 16 and star['mag'] < 18 for star in ngs] for ngs in results['ngs']]), axis=1)
    results['ngs_num_dim']    = np.sum(np.array([[star['mag'] >= 18 for star in ngs] for ngs in results['ngs']]), axis=1)

    results['ngs_mean_mag'] = results['ngs_mags'].mean(axis=1)
    results['ngs_max_mag'] = results['ngs_mags'].max(axis=1)
    results['ngs_min_mag'] = results['ngs_mags'].min(axis=1)

    results['mean_sr'] = results['sr'].mean(axis=1)
    results['mean_fwhm'] = results['fwhm'].mean(axis=1)
    results['mean_ee'] = results['ee'].mean(axis=1)

    results['pv_sr'] = np.abs(results['sr'].max(axis=1) - results['sr'].min(axis=1))
    results['pv_fwhm'] = np.abs(results['fwhm'].max(axis=1) - results['fwhm'].min(axis=1))
    results['pv_ee'] = np.abs(results['ee'].max(axis=1) - results['ee'].min(axis=1))

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

def load_matlab_results(output_path_or_file_path, name=None, recompute=False, extra_vib=None, sort_r=None, sort_theta=None, max_phase_screens=None, ee_size=None, return_psfs=False):
    if max_phase_screens is not None and recompute:
        raise SimulateException("max_phase_screens and recompute=True cannot be used together")

    if extra_vib is not None and not recompute:
        raise SimulateException("extra_vib requires recompute=True")

    if os.path.isfile(output_path_or_file_path):
        output_file = output_path_or_file_path
        output_path = os.path.dirname(output_file)
        name = os.path.splitext(os.path.basename(output_file))[0]
    else:
        output_path = output_path_or_file_path

        if name is None:
            raise SimulateException("Name must be provided if output_path is a directory")

        output_file = f"{output_path}/{name}.mat"
        if not os.path.isfile(output_file):
            raise SimulateException(f"File not found: {output_file}")

    if not recompute:
        output_file = os.path.join(os.path.dirname(output_file), f"stats_{os.path.basename(output_file)}")

    matlab_results = loadmat(output_file)

    if 'moao' in name.lower() or name.isdigit():
        mode = 'MOAO'
    elif 'ltao' in name.lower():
        mode = 'LTAO'
    else:
        mode = 'GLAO'

    pixel_scale = matlab_results['parm']['pixelScale']*1000
    wavelength = matlab_results['parm']['sci']['wavelength']
    zenith_angle = matlab_results['parm']['atm']['zenithAngle']/np.pi*180

    if ee_size is None:
        if mode == 'LTAO':
            ee_size = 50.0 * u.mas
        else:
            ee_size = 100.0 * u.mas
    elif not isinstance(ee_size, u.Quantity):
        ee_size = ee_size * u.mas

    seeing = 0.543 * u.arcsec # median

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
    theta[np.isclose(theta, 360.0, atol=2e-4)] = 0.0
    x = r * np.cos(np.deg2rad(theta))
    y = r * np.sin(np.deg2rad(theta))
    x[np.isclose(x, 0, atol=2e-4)] = 0.0
    y[np.isclose(y, 0, atol=2e-4)] = 0.0

    results = {
        'name': name,
        'mode': mode,
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
        'theta': theta,
        'x': x,
        'y': y
    }

    sort_indices = []
    if sort_r is not None or sort_theta is not None:
        if sort_r is None or sort_theta is None:
            raise SimulateException("sort_r and sort_theta must be provided together")
        if sort_r.shape != r.shape or sort_theta.shape != theta.shape:
            raise SimulateException("sort_r and sort_theta must have the same shape as r and theta")
        
        for sr, st in zip(sort_r, sort_theta):
            if np.isclose(st, 360.0, atol=2e-4):
                st = 0.0
            idx = np.where((np.isclose(r, sr, atol=2e-4)) & (np.isclose(theta, st, atol=2e-4)))[0]
            if len(idx) == 0:
                raise SimulateException(f"Could not match (r, theta)=({sr}, {st}) in MATLAB results")
            sort_indices.append(idx[0])
        sort_indices = np.array(sort_indices)
        r = r[sort_indices]
        theta = theta[sort_indices]
        x = x[sort_indices]
        y = y[sort_indices]

    if recompute:
        psfs = rearrange_matlab_psfs(matlab_results['psfs'])
        if len(sort_indices) > 0:
            psfs = psfs[sort_indices]

        if extra_vib is not None:
            aostats.add_extra_vibrations(psfs, extra_vib, pixel_scale)

        tel_diameter = matlab_results['parm']['tel']['Dsupp']
        pupil_file = os.path.join(output_path, 'pupil.mat')
        if os.path.isfile(pupil_file):
            tel_pupil = loadmat(pupil_file)['pupil']
        else:
            raise SimulateException(f"File pupil file is missing: {pupil_file}")

        sr, fwhm, ee = aostats.get_stats_matlab(psfs, tel_diameter, tel_pupil, wavelength, pixel_scale, ee_size)
    else:
        if max_phase_screens is not None:
            sr = matlab_results['cumSR'][:,max_phase_screens]
            fwhm = matlab_results['cumFWHM'][:,max_phase_screens] * 1000.0
            ee = matlab_results['cumEE01'][:,max_phase_screens]
        else:
            sr = matlab_results['sr']
            fwhm = matlab_results['fwhm'] * 1000.0
            ee = matlab_results['ee01']

        if len(sort_indices) > 0:
            sr = sr[sort_indices]
            fwhm = fwhm[sort_indices]
            ee = ee[sort_indices]

    results.update({
        'sr': sr,
        'fwhm': fwhm,
        'ee': ee
    })

    if return_psfs:
        results['psfs'] = psfs

    return results

def find_point(results, zd, az, max_dist=1.0):
    dists = np.sqrt(results['r']**2 + zd**2 - 2 * results['r'] * zd * np.cos(np.deg2rad(results['theta'] - az)))

    closest_idx = np.argmin(dists)
    if dists[closest_idx] > max_dist:
        raise SimulateException(f"No point ound within 1 arcsec: {zd:.2f}, {az:.2f}")

    return closest_idx

def find_asterism(results, mag=None, zd=None, az=None):
    asterism_id = 1 + next((i for i, ngs in enumerate(results['ngs']) if all((mag is None or star['mag'] == mag[j]) and (zd is None or star['zd'] == zd[j]) and (az is None or star['az'] == az[j]) for j, star in enumerate(ngs))), None)    
    if asterism_id is None:
        raise SimulateException(f"Asterism not found")
    return asterism_id

def compute_difference(results1, results2, relative=False, absolute_value=False, skip_sort=False):
    if results1['r'].shape != results2['r'].shape or results1['theta'].shape != results2['theta'].shape:
        raise SimulateException("results must have the same shape as r and theta")

    if not skip_sort:
        results2 = deepcopy(results2)
        sort_indices = []
        for sr, st in zip(results1['r'], results1['theta']):
            idx = np.where((np.isclose(results2['r'], sr, atol=2e-4)) & (np.isclose(results2['theta'], st, atol=2e-4)))[0]
            if len(idx) == 0:
                raise SimulateException(f"Could not match (r, theta)=({sr}, {st}) results2")
            sort_indices.append(idx[0])
        sort_indices = np.array(sort_indices)
        results2['r'] = results2['r'][sort_indices]
        results2['theta'] = results2['theta'][sort_indices]
        results2['sr'] = results2['sr'][sort_indices]
        results2['ee'] = results2['ee'][sort_indices]
        results2['fwhm'] = results2['fwhm'][sort_indices]
    
    results3 = deepcopy(results1)
    results3['sr'] = results2['sr']-results1['sr']
    results3['fwhm'] = results2['fwhm']-results1['fwhm']
    results3['ee'] = results2['ee']-results1['ee']

    if relative:
        results3['sr'] = results3['sr'] / results1['sr']
        results3['fwhm'] = results3['fwhm'] / results1['fwhm']
        results3['ee'] = results3['ee'] / results1['ee']

    if absolute_value:
        results3['sr'] = np.abs(results3['sr'])
        results3['fwhm'] = np.abs(results3['fwhm'])
        results3['ee'] = np.abs(results3['ee'])

    return results3

def format_contour_label(x):
    s = f"{x:.2f}"
    if s.endswith("0"):
        s = f"{x:.1f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

def get_plot_range(name, mode, plot_value, plot_range=None, fixed_range=False, contours=None, compare_contours=None):
    if plot_range is not None:
        fixed_range = True
        compare_range = plot_range
    elif fixed_range:
        match plot_value:
            case 'SR':
                if mode == 'LTAO' or 'tiled' in name.lower():
                    vmin = 0.10
                    vmax = 0.50
                else:
                    vmin = 0.00
                    vmax = 0.40
            case 'FWHM':
                if mode == 'LTAO' or 'tiled' in name.lower():
                    vmin = 60.0
                    vmax = 80.0
                else:
                    vmin = 60.0
                    vmax = 120.0
            case 'EE':
                if mode == 'LTAO' or 'tiled' in name.lower():
                    vmin = 0.20
                    vmax = 0.60
                else:
                    vmin = 0.00
                    vmax = 0.60
            case _:
                vmin = None
                vmax = None

        plot_range = [vmin, vmax]

    compare_vmin = -40.0
    compare_vmax = 40.0
    compare_range = [compare_vmin, compare_vmax]

    if contours is None:
        match plot_value:
            case 'SR':
                if mode == 'LTAO' or 'tiled' in name.lower():
                    contours = np.arange(0.10, 0.50, 0.02)
                else:
                    contours = np.arange(0.05, 0.50, 0.05)
            case 'FWHM':
                if mode == 'LTAO' or 'tiled' in name.lower():
                    contours = np.arange(50, 80, 1)
                else:
                    contours = np.arange(50, 140, 5)
            case 'EE':
                if mode == 'LTAO' or 'tiled' in name.lower():
                    contours = np.arange(0.10, 1.0, 0.05)
                else:
                    contours = np.arange(0.05, 1.0, 0.05)
            case _:
                contours = None

    if compare_contours is None:
        match plot_value:
            case 'SR':
                compare_contours = np.arange(-50, 50, 5)
            case 'FWHM':
                compare_contours = np.arange(-50, 50, 5)
            case 'EE':
                compare_contours = np.arange(-50, 50, 5)
            case _:
                compare_contours = None

    return plot_range, compare_range, fixed_range, contours, compare_contours

def plot_fov(all_results, asterism_id=1, labels=None, plot_value='SR', plot_fov=None, is_percent=False, contours=None, skip_smoothing=False, skip_contours=False, fixed_range=False, plot_range=None, mark_points=None, plot_mags=False, plot_points=False):
    if not isinstance(all_results, list):
        all_results = [all_results]

    plot_range, _, fixed_range, contours, _ = get_plot_range(all_results[0]['name'], all_results[0]['mode'], plot_value, plot_range, fixed_range, contours)

    if len(all_results) > 1:
        width_ratios = [1.0] * len(all_results) + ([0.1] if fixed_range else [])
        fig = plt.figure(figsize=(5*len(all_results)+0.5, 5))
        gs = GridSpec(1, len(all_results)+(1 if fixed_range else 0), width_ratios=width_ratios, wspace=0.3)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

    for i, results in enumerate(all_results):
        if len(all_results) > 1:
            ax = fig.add_subplot(gs[i])
        
        im = plot_results(ax, results, asterism_id=asterism_id, label=labels[i] if labels is not None else None, plot_value=plot_value, plot_fov=plot_fov, is_percent=is_percent, contours=contours, skip_smoothing=skip_smoothing, skip_contours=skip_contours, plot_range=plot_range, mark_points=mark_points, plot_mags=plot_mags, plot_points=plot_points)

        if len(all_results) == 1 or not fixed_range or i+1 == len(all_results):
            if fixed_range and len(all_results) > 1:
                plot_cbar(fig, im, plot_value, results, is_percent=is_percent, gs=gs[i+1])
            else:
                plot_cbar(fig, im, plot_value, results, is_percent=is_percent)

    plt.show()

def plot_compare_fov(results1, results2, asterism_id=1, labels=None, plot_value='SR', plot_fov=None, compare_absolute=False, requirement=None, plot_range=None, contours=None, compare_contours=None, skip_smoothing=False, skip_contours=False, mark_points=None, plot_mags=False, plot_points=False):
    plot_range, compare_range, _, contours, compare_contours = get_plot_range(results1['name'], results1['mode'], plot_value, plot_range, True, contours, compare_contours)

    results3 = compute_difference(results1, results2, relative=not compare_absolute, absolute_value=requirement is not None)
    if requirement is not None:
        compare_range[0] = 0.0

        if plot_value == 'FWHM':
            compare_contours = [requirement, 1000.0 if compare_absolute else 100.0]
        else:
            compare_contours = [requirement, 1.0 if compare_absolute else 100.0]

    width_ratios = [1, 1, 0.1, 1, 0.1]
    fig = plt.figure(figsize=(15+1.0, 5))
    gs = GridSpec(1, 5, width_ratios=width_ratios, wspace=0.3)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

    ax = fig.add_subplot(gs[0])
    im = plot_results(ax, results1, plot_range, asterism_id=asterism_id, label=labels[0] if labels is not None else None, plot_value=plot_value, plot_fov=plot_fov, contours=contours, skip_smoothing=skip_smoothing, skip_contours=skip_contours, mark_points=mark_points, plot_mags=plot_mags, plot_points=plot_points)
    ax = fig.add_subplot(gs[1])
    im = plot_results(ax, results2, plot_range, asterism_id=asterism_id, label=labels[1] if labels is not None else None, plot_value=plot_value, plot_fov=plot_fov, contours=contours, skip_smoothing=skip_smoothing, skip_contours=skip_contours, mark_points=mark_points, plot_mags=plot_mags, plot_points=plot_points)
    plot_cbar(fig, im, plot_value, results1, gs=gs[2])
    ax = fig.add_subplot(gs[3])
    im, values = plot_results(ax, results3, compare_range, asterism_id=asterism_id, label='Diff' if labels is not None else None, plot_value=plot_value, plot_fov=plot_fov, is_percent=not compare_absolute, contours=compare_contours, skip_smoothing=skip_smoothing, skip_contours=skip_contours, mark_points=mark_points, plot_mags=plot_mags, plot_points=plot_points, return_values=True)
    if requirement is not None:
        if not compare_absolute or plot_value == 'FWHM':
            requirement_fraction = np.sum(values <= requirement) / np.size(values)
        else:
            requirement_fraction = np.sum(values >= requirement) / np.size(values)

        ax.text(0.5, 0.5, f"{requirement_fraction:.1%}", transform=ax.transAxes, fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plot_cbar(fig, im, plot_value, results3, is_percent=not compare_absolute, gs=gs[4])
    plt.show()

def plot_requirement_fov(all_results, plot_value, requirement, asterism_id=1, labels=None, plot_fov=None, plot_range=None, mark_points=None, plot_mags=False, plot_points=False):
    if not isinstance(all_results, list):
        all_results = [all_results]

    plot_range, _, _, _, _ = get_plot_range(all_results[0]['name'], all_results[0]['mode'], plot_value, plot_range, True)
    if plot_value == 'FWHM':
        contours = [requirement, 1000.0]
    else:
        contours = [requirement, 1.0]

    if len(all_results) > 1:
        width_ratios = [1.0] * len(all_results) + [0.1]
        fig = plt.figure(figsize=(5*len(all_results)+0.5, 5))
        gs = GridSpec(1, len(all_results)+1, width_ratios=width_ratios, wspace=0.3)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

    for i, results in enumerate(all_results):
        if len(all_results) > 1:
            ax = fig.add_subplot(gs[i])
        
        im, values = plot_results(ax, results, asterism_id=asterism_id, label=labels[i] if labels is not None else None, plot_value=plot_value, plot_fov=plot_fov, contours=contours, plot_range=plot_range, mark_points=mark_points, plot_mags=plot_mags, plot_points=plot_points, return_values=True)

        if plot_value == 'FWHM':
            requirement_fraction = np.sum(values <= requirement) / np.size(values)
        else:
            requirement_fraction = np.sum(values >= requirement) / np.size(values)

        ax.text(0.5, 0.5, f"{requirement_fraction:.1%}", transform=ax.transAxes, fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        if len(all_results) == 1 or i+1 == len(all_results):
            if len(all_results) > 1:
                plot_cbar(fig, im, plot_value, results, gs=gs[i+1])
            else:
                plot_cbar(fig, im, plot_value, results)

    plt.show()

def get_star_size(mag, default_size=100):
    if mag is None:
        return default_size
    else:
        return 300 - max(0,50*(mag-15))

def plot_results(ax, results, plot_range, asterism_id=1, label=None, plot_value='SR', plot_fov=None, is_percent=False, contours=None, skip_smoothing=False, skip_contours=False, mark_points=None, plot_mags=False, plot_points=False, return_values=False):
    asterism_idx = asterism_id - 1
    N = len(results['r'])

    x = np.round(results['r'] * np.cos(np.deg2rad(results['theta'])), 4)
    y = np.round(results['r'] * np.sin(np.deg2rad(results['theta'])), 4)

    if plot_fov is not None:
        side = plot_fov
    elif results['mode'] == 'LTAO':
        side = 20.0
    else:
        side = 120.0

    if results['mode'] == 'LTAO' or 'tiled' in results['name'].lower():
        Nside = int(np.sqrt(N))
        if np.mod(Nside,2) == 0:
            x -= side/Nside/2
            y -= side/Nside/2

    radius= side/2

    if results['mode'] == 'LTAO' or 'tiled' in results['name'].lower():
        if np.mod(Nside,2) == 0:
            limits = [-8, 8]
        else:
            limits = [-radius, radius]

        if skip_smoothing:
            limits += np.sign(limits)*side/Nside/2
    else:
        limits = [-radius, radius]

    if results['mode'] == 'LTAO' or 'tiled' in results['name'].lower():
        mask = (np.abs(x) <= radius) & (np.abs(y) <= radius)
    else:
        mask = results['r'] <= radius

    match plot_value:
        case 'SR':
            values = results['sr']
        case 'FWHM':
            values = results['fwhm']
        case 'EE':
            values = results['ee']

    if is_percent:
        values *= 100.0

    values_mean = np.mean(values[mask])
    values_std = np.std(values[mask])
    values_pv = np.max(values[mask]) - np.min(values[mask])

    m = 201
    xi = np.linspace(limits[0], limits[1], m)
    yi = np.linspace(limits[0], limits[1], m)
    X, Y = np.meshgrid(xi, yi)
    if skip_smoothing:
        VALUES = griddata((x, y), values, (X, Y), method='nearest')
    else:
        if results['mode'] == 'LTAO' or 'tiled' in results['name'].lower():
            VALUES = griddata((x, y), values, (X, Y), method='cubic')
        else:
            VALUES = Rbf(x, y, values, function='cubic')(X, Y)
    MASK = np.ones(VALUES.shape, dtype=np.float64)
    if results['mode'] == 'LTAO' or 'tiled' in results['name'].lower():
        MASK[(np.abs(X) > limits[1]) | (np.abs(Y) > limits[1])] = np.nan
    else:
        MASK[np.sqrt(X**2 + Y**2) > limits[1]] = np.nan

    vmin = plot_range[0] if plot_range is not None else None
    vmax = plot_range[1] if plot_range is not None else None

    match plot_value:
        case 'SR':
            title = 'SR'
            cmap = 'plasma'
        case 'FWHM':
            title = 'FWHM'
            cmap = 'plasma_r'
        case 'EE':
            title = 'EE'
            cmap = 'plasma'

    if label is not None:
        title = f"{label}: {title}"

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
    if is_percent:
        ax.set_title(f"{title} Mean={values_mean:.1f}%, Std={values_std:.1f}%, PV={values_pv:.1f}%", fontweight='bold')
    else:
        ax.set_title(f"{title} Mean={values_mean:.2f}, Std={values_std:.2f}, PV={values_pv:.2f}", fontweight='bold')

    # Plot the circle boundary
    if results['mode'] != 'LTAO' and 'tiled' not in results['name'].lower():
        circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=4)
        ax.add_patch(circle)

    # Plot the LGS
    if 'LGS_zd' in results:
        lgs_zd = results['LGS_zd']
        lgs_az = results['LGS_az']
        if 'LGS_mag' in results:
            lgs_mag = results['LGS_mag']
        else:
            lgs_mag = np.full(len(lgs_zd), None)
    elif 'lgs' in results:
        lgs_zd = [star['zd'] for star in results['lgs']]
        lgs_az = [star['az'] for star in results['lgs']]
        lgs_mag = [star['mag'] for star in results['lgs']]
    else:
        lgs_zd = []
        lgs_az = []
        lgs_mag = []

    for k, (zd, az) in enumerate(zip(lgs_zd, lgs_az)):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        ax.scatter(x, y, marker=(5, 1), facecolor='yellow', edgecolor='k', s=get_star_size(lgs_mag[k], 200), linewidths=0.5)

    # Plot the NGS
    if 'NGS_zd' in results:
        ngs_zd = results['NGS_zd']
        ngs_az = results['NGS_az']
        if 'NGS_mag' in results:
            ngs_mag = results['NGS_mag']
        else:
            ngs_mag = np.full(len(ngs_zd), None)
    elif 'ngs' in results:
        ngs_zd = [star['zd'] for star in results['ngs'][asterism_idx]]
        ngs_az = [star['az'] for star in results['ngs'][asterism_idx]]
        ngs_mag = [star['mag'] for star in results['ngs'][asterism_idx]]
    else:
        ngs_zd = []
        ngs_az = []
        ngs_mag = []

    for k, (zd, az) in enumerate(zip(ngs_zd, ngs_az)):
        x = zd * np.cos(np.deg2rad(az))
        y = zd * np.sin(np.deg2rad(az))
        ax.scatter(x, y, marker=(5, 1), facecolor='red', edgecolors='k', s=get_star_size(ngs_mag[k], 100), linewidths=0.5)
        if plot_mags and ngs_mag[k] is not None:
            ax.text(x+6, y+5, f"{ngs_mag[k]:.1f}", fontsize=8, ha='center', va='center', color='black')

    # Plot other points
    if mark_points is not None:
        for idx in mark_points:
            x = results['r'][idx] * np.cos(np.deg2rad(results['theta'][idx]))
            y = results['r'][idx] * np.sin(np.deg2rad(results['theta'][idx]))
            ax.scatter(x, y, marker='o', facecolor='blue', edgecolor='k', s=100, linewidths=0.5)

    # Plot FWHM at each point
    if plot_points:
        for k, (zd, az) in enumerate(zip(results['r'], results['theta'])):
            x = zd * np.cos(np.deg2rad(az))
            y = zd * np.sin(np.deg2rad(az))
            fwhm = results['fwhm'][k] / 1000.0 # mas -> arcsec
            circle = plt.Circle((x, y), fwhm/2, fill=False, color='k', linewidth=1)
            ax.add_patch(circle)
            #ax.text(x, y, f"({zd:.0f}, {az:.0f})", fontsize=6, ha='center', va='center', color='blue')

    ax.set_aspect('equal')
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    if return_values:
        return im, values[mask]
    else:
        return im

def plot_cbar(fig, im, plot_value, results, is_percent=False, gs=None):
    match plot_value:
        case 'SR':
            bkg_label = "Strehl Ratio"
        case 'FWHM':
            bkg_label = 'FWHM [mas]'
        case 'EE':
            bkg_label = f"EE [{results['ee_size']:.0f} mas]"

    if gs is not None:
        cax = fig.add_subplot(gs)
        cbar = fig.colorbar(im, cax=cax, format='%.1f%%' if is_percent else '%.2f')
        pos = cax.get_position()
        center_offset = pos.height * 0.2 / 2
        cax.set_position([
            pos.x0 - 0.035,
            pos.y0 + center_offset,
            pos.width,
            pos.height * 0.8
        ])
    else:
        cbar = fig.colorbar(im, ax=fig.gca(), shrink=0.7)

    cbar.ax.set_ylabel(bkg_label)

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

def plot_asterism_stats(all_results, labels, plot_value='SR', plot_x='scale', point_filter=None, plot_range=None, plot_asterism_codes=False, hide_title=False):
    # TODO: accept a variable number of results

    results1 = all_results[0]
    results2 = all_results[1]
    results3 = all_results[2]

    if plot_x == 'mag':
        xValues1 = results1['ngs_mean_mag']
        xValues2 = results2['ngs_mean_mag']
        xValues3 = results3['ngs_mean_mag']
        xLabel = 'Mean NGS Mag'

        cValues1 = results1['asterism_scale']
        cValues2 = results2['asterism_scale']
        cValues3 = results3['asterism_scale']
        cLabel = 'Scale [arcsec]'

        cmap_diverge = True
        cmap_reverse = False
        vmin = 0
        vmax = 120

        xOffset = 0.05
    else:
        xValues1 = results1['asterism_scale']
        xValues2 = results2['asterism_scale']
        xValues3 = results3['asterism_scale']
        xLabel = 'Scale [arcsec]'

        cValues1 = results1['ngs_mean_mag']
        cValues2 = results2['ngs_mean_mag']
        cValues3 = results3['ngs_mean_mag']
        cLabel = 'Mean NGS Mag'

        cmap_diverge = False
        cmap_reverse = True
        vmin = 15
        vmax = 19

        xOffset = 1.5

    if point_filter is None:
        if 'SR' in plot_value:
            mean1 = results1['mean_sr']
            mean2 = results2['mean_sr']
            mean3 = results3['mean_sr']
            pv1 = results1['pv_sr']
            pv2 = results2['pv_sr']
            pv3 = results3['pv_sr']
        elif 'FWHM' in plot_value:
            mean1 = results1['mean_fwhm']
            mean2 = results2['mean_fwhm']
            mean3 = results3['mean_fwhm']
            pv1 = results1['pv_fwhm']
            pv2 = results2['pv_fwhm']
            pv3 = results3['pv_fwhm']
        elif 'EE' in plot_value:
            mean1 = results1['mean_ee']
            mean2 = results2['mean_ee']
            mean3 = results3['mean_ee']
            pv1 = results1['pv_ee']
            pv2 = results2['pv_ee']
            pv3 = results3['pv_ee']
        else:
            raise SimulateException(f"Unknown plot value: {plot_value}")
    else:
        if 'SR' in plot_value:
            mean1 = np.mean(results1['sr'][:,point_filter], axis=1)
            mean2 = np.mean(results2['sr'][:,point_filter], axis=1)
            mean3 = np.mean(results3['sr'][:,point_filter], axis=1)
            pv1 = np.abs(np.max(results1['sr'][:,point_filter], axis=1) - np.min(results1['sr'][:,point_filter], axis=1))
            pv2 = np.abs(np.max(results2['sr'][:,point_filter], axis=1) - np.min(results2['sr'][:,point_filter], axis=1))
            pv3 = np.abs(np.max(results3['sr'][:,point_filter], axis=1) - np.min(results3['sr'][:,point_filter], axis=1))
        elif 'FWHM' in plot_value:
            mean1 = np.mean(results1['fwhm'][:,point_filter], axis=1)
            mean2 = np.mean(results2['fwhm'][:,point_filter], axis=1)
            mean3 = np.mean(results3['fwhm'][:,point_filter], axis=1)
            pv1 = np.abs(np.max(results1['fwhm'][:,point_filter], axis=1) - np.min(results1['fwhm'][:,point_filter], axis=1))
            pv2 = np.abs(np.max(results2['fwhm'][:,point_filter], axis=1) - np.min(results2['fwhm'][:,point_filter], axis=1))
            pv3 = np.abs(np.max(results3['fwhm'][:,point_filter], axis=1) - np.min(results3['fwhm'][:,point_filter], axis=1))
        elif 'EE' in plot_value:
            mean1 = np.mean(results1['ee'][:,point_filter], axis=1)
            mean2 = np.mean(results2['ee'][:,point_filter], axis=1)
            mean3 = np.mean(results3['ee'][:,point_filter], axis=1)
            pv1 = np.abs(np.max(results1['ee'][:,point_filter], axis=1) - np.min(results1['ee'][:,point_filter], axis=1))
            pv2 = np.abs(np.max(results2['ee'][:,point_filter], axis=1) - np.min(results2['ee'][:,point_filter], axis=1))
            pv3 = np.abs(np.max(results3['ee'][:,point_filter], axis=1) - np.min(results3['ee'][:,point_filter], axis=1))
        else:
            raise SimulateException(f"Unknown plot value: {plot_value}")

    if "_REL" in plot_value:
        yValues1 = mean1 / mean1
        yValues2 = mean2 / mean1
        yValues3 = mean3 / mean1
        yLabel = f"Mean {plot_value.replace("_REL","")} / LGS42"
    elif "_PV" in plot_value:
        yValues1 = pv1 / mean1
        yValues2 = pv2 / mean2
        yValues3 = pv3 / mean3
        yLabel = f"PV / Mean {plot_value.replace('_PV','')}"
    else:
        yValues1 = mean1
        yValues2 = mean2
        yValues3 = mean3
        yLabel = f"Mean {plot_value}"

    is_percent = '_PV' in plot_value or '_REL' in plot_value
    if is_percent:
        yValues1 *= 100.0
        yValues2 *= 100.0
        yValues3 *= 100.0

    if 'FWHM' in plot_value:
        yLabel = yLabel.replace('FWHM', f'FWHM [mas]')
    if 'EE' in plot_value:
        yLabel = yLabel.replace('EE', f'EE [{results1["ee_size"]:.0f} mas]')

    if point_filter is not None:
        xValues1 = xValues1[point_filter]
        xValues2 = xValues2[point_filter]
        xValues3 = xValues3[point_filter]
        yValues1 = yValues1[point_filter]
        yValues2 = yValues2[point_filter]
        yValues3 = yValues3[point_filter]

    if cmap_diverge:
        cmap_range = np.linspace(0.0, 1.0, 256)
    else:
        cmap_range = np.linspace(0.0, 0.7, 256)

    fig = plt.figure(figsize=(15, 8))
    plt.rcParams.update({'xtick.labelsize': 12, 'ytick.labelsize': 12})
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap('Reds' + ('_r' if cmap_reverse else ''))
    cmap_segment = mcolors.LinearSegmentedColormap.from_list('cropped', cmap(cmap_range))
    if cmap_diverge:
        colors = cmap_segment(np.linspace(0, 1, 256))
        mid_peak = np.concatenate([colors[:128],colors[:128][::-1]])  # Mirror around midpoint
        cmap_segment = mcolors.LinearSegmentedColormap.from_list("peaked", mid_peak)
    s1 = ax.scatter(xValues1+xOffset, yValues1, s=10, c=cValues1, cmap=cmap_segment, vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap('Greens' + ('_r' if cmap_reverse else ''))
    cmap_segment = mcolors.LinearSegmentedColormap.from_list('cropped', cmap(cmap_range))
    if cmap_diverge:
        colors = cmap_segment(np.linspace(0, 1, 256))
        mid_peak = np.concatenate([colors[:128],colors[:128][::-1]])  # Mirror around midpoint
        cmap_segment = mcolors.LinearSegmentedColormap.from_list("peaked", mid_peak)
    s2 = ax.scatter(xValues2       , yValues2, s=10, c=cValues2, cmap=cmap_segment, vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap('Blues' + ('_r' if cmap_reverse else ''))
    cmap_segment = mcolors.LinearSegmentedColormap.from_list('cropped', cmap(cmap_range))
    if cmap_diverge:
        colors = cmap_segment(np.linspace(0, 1, 256))
        mid_peak = np.concatenate([colors[:128],colors[:128][::-1]])  # Mirror around midpoint
        cmap_segment = mcolors.LinearSegmentedColormap.from_list("peaked", mid_peak)
    s3 = ax.scatter(xValues3-xOffset, yValues3, s=10, c=cValues3, cmap=cmap_segment, vmin=vmin, vmax=vmax)
    if plot_asterism_codes and 'REL' not in plot_value:
        for i in range(len(results1['ngs_num_bright'])):
            if xValues1[i] > 20:
                continue
            asterism_code = f"{results1['ngs_num_bright'][i]}{results1['ngs_num_nominal'][i]}{results1['ngs_num_dim'][i]}"
            ax.annotate(asterism_code, (xValues1[i]+0.5, yValues1[i]), c='black', fontsize=8, ha='left')

    cbar1 = fig.colorbar(s1, ax=ax, pad=-0.085)
    cbar1.set_label(cLabel, fontsize=12)
    cbar1.ax.xaxis.set_label_position('bottom')
    cbar1.ax.set_xlabel(labels[0], fontsize=12)
    cbar1.ax.xaxis.set_label_coords(0.5, -0.02)
    cbar1.ax.tick_params(labelsize=12)

    cbar2 = fig.colorbar(s2, ax=ax, pad=-0.08)
    cbar2.set_ticks([])
    cbar2.ax.xaxis.set_label_position('bottom')
    cbar2.ax.xaxis.set_label_coords(0.5, -0.02)
    cbar2.ax.set_xlabel(labels[1], fontsize=12)

    cbar3 = fig.colorbar(s3, ax=ax, pad=0.03)
    cbar3.set_ticks([])
    cbar3.ax.xaxis.set_label_position('bottom')
    cbar3.ax.xaxis.set_label_coords(0.5, -0.02)
    cbar3.ax.set_xlabel(labels[2], fontsize=12)

    ax.set_xlabel(xLabel, fontsize=12)
    ax.set_ylabel(yLabel, fontsize=12)
    if plot_range is not None:
        ax.set_ylim(np.array(plot_range)*(100.0 if is_percent else 1.0))
    ax.grid()

    if is_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    if not hide_title:
        ax.set_title(f"Compare {yLabel}", fontsize=14, fontweight='bold')

    plt.show()

def _capture_output(func, *args, **kwargs):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func(*args, **kwargs)
    return f.getvalue().replace('\n', '<br>')  # Convert newlines to HTML

def print_error_breakdown(all_results, labels=None, idx=None):
    if not isinstance(all_results, list):
        all_results = [all_results]

    if idx is None:
        if all_results[0]['r'] is not None:
            idx = all_results[0]['r'].argmin()
        else:
            idx = 0

    html = "<table><tr>"
    for i, result in enumerate(all_results):
        html += f"""
        <td style="padding: 20px; text-align: left; vertical-align: top;">
            <pre>{_capture_output(result['errors'].print, idx=idx, label=labels[i] if labels is not None else None)}</pre>
        </td>
        """
    html += "</tr></table>"
    display(HTML(html))

def export_compare_fov(results1, results2, labels=None, filename=None):
    if labels is None:
        labels = ['Result1', 'Result2']
    else:
        labels = [label.replace(' ', '_') for label in labels]

    if filename is None and labels is not None:
        filename = f"compare_{labels[0]}_{labels[1]}.csv"

    table = Table({
        'r': results1['r'],
        'theta': results1['theta'],
        'x': results1['x'],
        'y': results1['y'],
        f'sr_{labels[0]}': results1['sr'],
        f'sr_{labels[1]}': results2['sr'],
        f'ee_{labels[0]}': results1['ee'],
        f'ee_{labels[1]}': results2['ee'],
        f'fwhm_{labels[0]}': results1['fwhm'],
        f'fwhm_{labels[1]}': results2['fwhm'],
    })

    table.write(filename, format='csv', overwrite=True)
