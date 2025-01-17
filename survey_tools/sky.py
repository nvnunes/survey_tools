#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

# pylint: disable=redefined-builtin
from os import path
import pathlib
import numpy as np
from astropy.constants import si as constants
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import ascii
from astropy.table import Table
from scipy.signal import find_peaks, peak_widths # pylint: disable=no-name-in-module

class StructType:
    pass

class AtmosphereException(Exception):
    pass

def get_default_data_path():
    data_path = f"{pathlib.Path(__file__).parent.parent.resolve()}/data/sky"

    if not path.exists(data_path):
        raise AtmosphereException('Data files missing. See "data/README.txt" for more details.')

    return data_path

def _get_airmass_for_filename(location, airmass):
    match location:
        case 'MaunaKea':
            if airmass < 1.25:
                return 1.0
            elif airmass < 1.75:
                return 1.5
            else:
                return 2.0
        case 'Paranal':
            if airmass < 1.05:
                return 1.0
            elif airmass < 1.25:
                return 1.15
            elif airmass < 1.75:
                return 1.5
            else:
                return 2.0

    return airmass

def load_transmission_data(location, airmass, data_path = None):
    if data_path is None:
        data_path = get_default_data_path()

    match location:
        case 'MaunaKea':
            # From: https://www.gemini.edu/observing/telescopes-and-sites/sites#Transmission
            #
            # The infrared spectra of the atmospheric transmission above Mauna Kea that are used
            # in the Integration Time Calculators have been generated using the ATRAN modelling
            # software (Lord, S.D. 1992, NASA Technical Memor. 103957) and are presented separately
            # for the near-IR and mid-IR. Ascii data files of these spectra are available below.
            #
            # Column 1: wavelength [micron]
            # Column 2: transmission [%]
            transmission_data = ascii.read(f"{data_path}/mk_trans_zm_10_{_get_airmass_for_filename(location, airmass)*10:.0f}.dat", names=['wavelength', 'transmission'])
            transmission_data['wavelength'] *= 1e4 # micron -> angstrom
        case 'Paranal':
            # From: https://www.eso.org/sci/facilities/eelt/science/drm/tech_data/background/
            #
            # Column 1: wavelength [um]
            # Column 2: transmission [%]
            transmission_data = ascii.read(f"{data_path}/paranal_trans_airm{_get_airmass_for_filename(location, airmass):.2f}_wav00.4-03.0.dat", names=['wavelength', 'transmission'])
            transmission_data['wavelength'] *= 1e4 # um -> angstrom
        case _:
            raise AtmosphereException('Unknown location')

    return transmission_data

def load_background_data(location, airmass, data_path = None):
    if data_path is None:
        data_path = get_default_data_path()

    match location:
        case 'MaunaKea' | 'Paranal':
            # From: https://www.gemini.edu/observing/telescopes-and-sites/sites#IRSky
            #
            # The files were manufactured starting from the sky transmission files generated
            # by ATRAN (Lord, S. D., 1992, NASA Technical Memorandum 103957). These files were
            # subtracted from unity to give an emissivity and then multiplied by a blackbody
            # function of temperature 273 for Mauna Kea and 280 for Cerro Pachon. To these were
            # added the OH emission spectrum (available from the European Southern Observatory's
            # ISAAC web pages) a set of O2 lines near 1.3 microns with estimated strengths based
            # on observations at Mauna Kea, and the dark sky continuum (in part zodiacal light),
            # approximated as a 5800K gray body times the atmospheric transmission and scaled to
            # produce 18.2 mag/arcsec^2 in the H band, as observed on Mauna Kea by
            # Maihara et al. (1993 PASP, 105, 940).
            #
            # Any use of the data in these tables should reference Lord (1992) and acknowledge Gemini Observatory.
            #
            # Column 1: wavelength [nm]
            # Column 2: emission [ph/sec/arcsec^2/nm/m^2]
            background_data = ascii.read(f"{data_path}/mk_skybg_zm_10_{_get_airmass_for_filename(location, airmass)*10:.0f}_ph.dat", names=['wavelength', 'emission'])
            background_data['wavelength'] *= 10 # nanometer -> angstrom
        # case 'Paranal':
        #     # From: https://www.eso.org/sci/facilities/eelt/science/drm/tech_data/background/

        #     # Column 1: wavelength [um]
        #     # Column 2: emission [photons/m2/s/arcsec2]
        #     sky_background = ascii.read(f"{data_path}/paranal_optical_ir_sky_lines.dat", names=['wavelength', 'emission'])
        #     sky_background['wavelength'] *= 1e4  # um -> angstrom
        #     sky_background['emission']   /= (sky_background['wavelength']/10/spectral_resolving_power) # [photons/m2/s/arcsec2] -> [photons/s/nm/m^2/arcsec^2]
        case _:
            raise AtmosphereException('Unknown location')

    return background_data

def get_vacuum_to_air_wavelength(wavelength):
    # See: https://classic.sdss.org/dr7/products/spectra/vacwavelength.php
    return wavelength / (1 + 2.735182e-4 + 1.314182e2 * np.power(wavelength,-2) + 2.76249e8 * np.power(wavelength,-4))

def get_emission_line_rest_wavelengths(skip_close_doublets=False):
    lines = { # Angstrom
        'OIIa' : 3727.092,
        'OIIb' : 3729.875,
        'Hb'   : 4862.680,
        'OIIIa': 4960.295,
        'OIIIb': 5008.240,
        'NIIa' : 6549.86,
        'Ha'   : 6564.610,
        'NIIb' : 6585.27,
        'SIIa' : 6718.29,
        'SIIb' : 6732.67,
    }

    if skip_close_doublets:
        wavelength_OIIa = lines.pop('OIIa')
        wavelength_OIIb = lines.pop('OIIb')
        lines['OII'] = np.mean([wavelength_OIIa, wavelength_OIIb])

    return lines

# pylint: disable=unused-argument
def get_mean_transmission(transmission_data, wavelengths, fwhm, spectral_resolving_power, trans_dLambda_multiple = 0.5):
    if hasattr(wavelengths, 'shape'):
        N = wavelengths.size
        transmission = np.zeros(wavelengths.shape)
    else:
        N = len(wavelengths)
        transmission = np.zeros((N))

    for i in np.arange(N):
        if N == 1:
            wavelength = wavelengths
        else:
            wavelength = wavelengths[i]

        # Option 1: FWHM of line given dispersion
        # dwavelength = np.sqrt(np.power(current_wavelength/spectral_resolving_power, 2) + np.power(fwhm, 2))
        # Option 2: FWHM of peak given spectral resolving power (probably better as this will correspond to the transmission around the peak)
        dwavelength = wavelength/spectral_resolving_power
        # Option 3: Weighted by gaussian profile (probably the best option)

        if wavelength - dwavelength * trans_dLambda_multiple < transmission_data['wavelength'][0] or wavelength + dwavelength * trans_dLambda_multiple > transmission_data['wavelength'][-1]:
            mean_transmission = 0.0
        else:
            data_filter = (transmission_data['wavelength'] >= wavelength - dwavelength * trans_dLambda_multiple) & (transmission_data['wavelength'] <= wavelength + dwavelength * trans_dLambda_multiple)
            mean_transmission = np.mean(transmission_data['transmission'][data_filter])

        if N == 1:
            transmission = mean_transmission
        else:
            transmission[i] = mean_transmission

    return transmission

def get_low_res_background(background_data, wavelength_range, spectral_resolving_power):
    bin_size = round(background_data['wavelength'][1] - background_data['wavelength'][0], 3)

    mean_wavelength = np.mean(wavelength_range)
    mean_dwavelength = mean_wavelength / spectral_resolving_power

    wavelength_filter = (background_data['wavelength'] >= wavelength_range[0] - mean_dwavelength*10) & (background_data['wavelength'] <= wavelength_range[1] + mean_dwavelength*10)

    if np.sum(wavelength_filter) == 0:
        return None

    wavelengths = background_data['wavelength'][wavelength_filter]

    sigma = mean_dwavelength / 2.35482 # FWHM = 2.35482 * sigma
    kernel = Gaussian1DKernel(stddev = sigma / bin_size)
    emission_low_res = convolve(background_data['emission'][wavelength_filter], kernel)

    final_wavelength_filter = (wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1])

    return Table([
            wavelengths[final_wavelength_filter],
            emission_low_res[final_wavelength_filter]
        ], names=[
            'wavelength',
            'emission'
        ], dtype=[
            np.float64,
            np.float64
        ]
    )

def get_background(background_data, wavelengths, wavelength_range, spectral_resolving_power):
    if np.size(wavelengths) == 1:
        background_low_res = get_low_res_background(background_data, wavelength_range, spectral_resolving_power)
        if wavelengths <= wavelength_range[0] or wavelengths >= wavelength_range[1]:
            return 0.0
        return np.interp(wavelengths, background_low_res['wavelength'], background_low_res['emission'])
    else:
        backgrounds = np.full(wavelengths.shape, np.nan)
        for i in np.arange(len(wavelengths)):
            backgrounds[i] = get_background(background_data, wavelengths[i], [wavelength_range[0][i], wavelength_range[1][i]], spectral_resolving_power)
        return backgrounds

def find_sky_lines(background_data, min_photo_rate = 10.0):
    peaks, _ = find_peaks(background_data['emission'], height=min_photo_rate)
    widths, width_heights, left_ips, right_ips = peak_widths(background_data['emission'], peaks, rel_height=0.5) # FWHM

    lambda0 = background_data['wavelength'][0]
    bin_size = round(background_data['wavelength'][1] - background_data['wavelength'][0], 3)

    sky_lines = Table([
            background_data['wavelength'][peaks],
            background_data['emission'][peaks],
            widths * bin_size,
            width_heights,
            lambda0 + left_ips * bin_size,
            lambda0 + right_ips * bin_size
        ], names=[
            'wavelength',
            'emission',
            'width',
            'width_height',
            'wavelength_low',
            'wavelength_high'
        ], dtype=[
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64
        ]
    )

    return sky_lines

def reject_emission_line(background_data, transmission_data, wavelength, fwhm, spectral_resolving_power, allowed_wavelength_range = None, trans_minimum = 1.0, trans_dLambda_multiple = 0.5, avoid_dLambda_multiple = 1.0, min_photon_rate = 10.0):
    if hasattr(wavelength, 'shape'):
        N = wavelength.size
        rejects = np.ones(wavelength.shape, dtype=np.bool)
    else:
        N = len(wavelength)
        rejects = np.ones((N), dtype=np.bool)

    for i in np.arange(N):
        if N == 1:
            current_wavelength = wavelength
        else:
            current_wavelength = wavelength[i]

        if np.size(fwhm) == 1:
            current_fwhm = fwhm
        else:
            current_fwhm = fwhm[i]

        reject = False

        if current_wavelength == 0:
            reject = True
        else:
            dwavelength = np.sqrt(np.power(current_wavelength/spectral_resolving_power,2) + np.power(current_fwhm,2))
            wavelength_range = current_wavelength + 10*dwavelength*np.array([-0.5, 0.5])
            trans = get_mean_transmission(transmission_data, current_wavelength, current_fwhm, spectral_resolving_power, trans_dLambda_multiple)

            if trans < trans_minimum:
                reject = True

            if not reject and allowed_wavelength_range is not None:
                reject = True
                if np.ndim(allowed_wavelength_range) == 1:
                    if ((current_wavelength - dwavelength) >= allowed_wavelength_range[0] and (current_wavelength + dwavelength) <= allowed_wavelength_range[1]):
                        reject = False
                else:
                    num_ranges = np.shape(allowed_wavelength_range)[0]
                    for j in np.arange(num_ranges):
                        if ((current_wavelength - dwavelength) >= allowed_wavelength_range[j,0] and (current_wavelength + dwavelength) <= allowed_wavelength_range[j,1]):
                            reject = False
                            break

            if not reject and (wavelength_range[0] < background_data['wavelength'][0] or wavelength_range[1] > background_data['wavelength'][-1]):
                reject = True

            if not reject:
                background_low_res = get_low_res_background(background_data, wavelength_range, spectral_resolving_power)
                sky_lines = find_sky_lines(background_low_res, min_photon_rate)

                line_wavelength_low  = current_wavelength - dwavelength * avoid_dLambda_multiple
                line_wavelength_high = current_wavelength + dwavelength * avoid_dLambda_multiple
                reject = np.any((sky_lines['wavelength_low' ] <= line_wavelength_high) & (sky_lines['wavelength_high'] >= line_wavelength_low))

        if N == 1:
            rejects = reject
        else:
            rejects[i] = reject

    return rejects
