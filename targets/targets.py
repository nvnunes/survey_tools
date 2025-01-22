#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from copy import deepcopy
import json
import os
import pickle
import urllib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import numpy as np
from astropy import units as u
from astropy.constants import si as constants
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from photutils.centroids import centroid_com
from photutils.segmentation import detect_sources, make_2dgaussian_kernel
from regions import PixCoord, RectanglePixelRegion
from scipy.ndimage import rotate, zoom
from scipy.stats import norm
from survey_tools import catalog, gaia, healpix, sky
from survey_tools.utility import fit, plot, table

class StructType:
    pass

class TargetsException(Exception):
    pass

#region Instruments

def _get_instrument_details(instrument):
    instrument_details = dict()
    match instrument:
        case _ if 'GNIRS' in instrument:
            instrument_details['location'] = 'MaunaKea'
            instrument_details['wavelength_ranges'] = np.array([[11700, 13700],[14900, 18000],[19100,24900]])   # Angstrom
            instrument_details['field_diameter'] = 60 # arcsec
        case _ if 'ERIS' in instrument:
            instrument_details['location'] = 'Paranal'
            instrument_details['wavelength_ranges'] = np.array([[10900, 14200],[14500, 18700],[19300,24800]])   # Angstrom
            instrument_details['field_diameter'] = 60 # arcsec
        case _:
            raise TargetsException(f"Unknown instrument: {instrument}")

    match instrument:
        case _ if 'Altair' in instrument:
            instrument_details['wfs_band'] = 'R'
            if 'NGS' in instrument:
                instrument_details['min_ngs_mag'] = 8.5
                instrument_details['max_ngs_mag'] = 15.0
                instrument_details['min_ngs_sep'] = 0.0
                instrument_details['max_ngs_sep'] = 25.0
            elif 'LGS' in instrument:
                instrument_details['min_ngs_mag'] = 8.5
                instrument_details['max_ngs_mag'] = 18.5
                instrument_details['min_ngs_sep'] = 0.0
                instrument_details['max_ngs_sep'] = 25.0
            else: # LGS-PI (super-seeing)
                instrument_details['min_ngs_mag'] = 8.5
                instrument_details['max_ngs_mag'] = 14.0
                instrument_details['min_ngs_sep'] = 3.5*60
                instrument_details['max_ngs_sep'] = 7.0*60
        case _:
            instrument_details['wfs_band'] = None

    if 'GNIRS' in instrument:
        if 'LR_IFU' in instrument:
            instrument_details['ifu_width'] = 3.15 # arcsec
            instrument_details['ifu_height'] = 4.8 # arcsec
            instrument_details['ifu_width_spx'] = 21
            instrument_details['ifu_height_spx'] = 32
        elif 'HR_IFU' in instrument:
            instrument_details['ifu_width'] = 1.25 # arcsec
            instrument_details['ifu_height'] = 1.8 # arcsec
            instrument_details['ifu_width_spx'] = 25
            instrument_details['ifu_height_spx'] = 36
    elif 'ERIS' in instrument:
        if '25mas' in instrument:
            instrument_details['ifu_width'] = 0.8 # arcsec
            instrument_details['ifu_height'] = 0.8 # arcsec
            instrument_details['ifu_width_spx'] = 64
            instrument_details['ifu_height_spx'] = 32
        elif '100mas' in instrument:
            instrument_details['ifu_width'] = 3.2 # arcsec
            instrument_details['ifu_height'] = 3.2 # arcsec
            instrument_details['ifu_width_spx'] = 64
            instrument_details['ifu_height_spx'] = 32
        elif '250mas' in instrument:
            instrument_details['ifu_width'] = 8.0 # arcsec
            instrument_details['ifu_height'] = 8.0 # arcsec
            instrument_details['ifu_width_spx'] = 64
            instrument_details['ifu_height_spx'] = 32

    if 'ifu_width' in instrument_details and 'ifu_height' in instrument_details:
        instrument_details['instrument_is_ifu'] = True

        if 'ifu_width' in instrument_details and 'ifu_width_spx' in instrument_details:
            instrument_details['ifu_spaxel_width'] = instrument_details['ifu_width'] / instrument_details['ifu_width_spx']

        if 'ifu_height' in instrument_details and 'ifu_height_spx' in instrument_details:
            instrument_details['ifu_spaxel_height'] = instrument_details['ifu_height'] / instrument_details['ifu_height_spx']
    else:
        instrument_details['instrument_is_ifu'] = False

    return instrument_details

#endregion

#region Details

FLUX_NII_HA_RATIO = 0.1

def append_details(options, galaxies, log_sfr=False, skip_catalog=False):
    if not isinstance(options, dict):
        raise TargetsException('options must be a dict')

    if options.get('instrument', None) is None:
        raise TargetsException('instrument is required but missing in options')

    if options.get('resolving_power', None) is None:
        raise TargetsException('resolving_power is required but missing in options')

    if isinstance(galaxies, Table):
        if 'id' not in galaxies.colnames:
            raise TargetsException('galaxies is missing id column')
    else:
        catalog_names = None
        ids = None

        if isinstance(galaxies, (list, tuple)) and len(galaxies) == 2:
            # List or tuple of catalog name and id
            if isinstance(galaxies[0], str) and isinstance(galaxies[1], int):
                catalog_names = [galaxies[0]]
                ids = [galaxies[1]]
            # List or tuple of catalog name list and id list
            elif isinstance(galaxies[0], (list, tuple)) and isinstance(galaxies[1], (list, tuple)) and (len(galaxies[0]) == len(galaxies[1])) and all(isinstance(catalog_name, str) for catalog_name in galaxies[0]) and all(isinstance(id, int) for id in galaxies[1]):
                catalog_names = galaxies[0]
                ids = galaxies[1]
            # List or tuple of catalog name and list of ids
            elif isinstance(galaxies[0], str) and isinstance(galaxies[1], list) and all(isinstance(id, int) for id in galaxies[1]):
                catalog_names = np.full(len(galaxies[1]), galaxies[0])
                ids = galaxies[1]
        # List or tuple of tuples of catalog name and id
        elif isinstance(galaxies, (list, tuple)) and all(isinstance(g, tuple) for g in galaxies):
            catalog_names, ids = zip(*galaxies)

        if catalog_names is not None and ids is not None:
            galaxies = Table([catalog_names, ids], names=['catalog', 'id'])
        else:
            raise TargetsException('format of galaxies not recognized')

    # Add instrument details
    options.update(_get_instrument_details(options['instrument']))

    # Add default options
    if 'obs_epoch' not in options:
        options['obs_epoch'] = None
    if 'lines' not in options:
        options['lines'] = []
    if 'airmass' not in options:
        options['airmass'] = 1.5
    if 'min_photon_rate' not in options:
        options['min_photon_rate'] = 10 # ph/s/arcsec^2/nm/m^2
    if 'trans_minimum' not in options:
        options['trans_minimum'] = 0.85 # percent
    if 'trans_multiple' not in options:
        options['trans_multiple'] = 0.5 # fraction of FWHM
    if 'avoid_multiple' not in options:
        options['avoid_multiple'] = 0.5 # fraction of FWHM

    # Lookup in Catalog
    if not skip_catalog:
        fill_ra = 'ra' not in galaxies.colnames
        fill_dec = 'dec' not in galaxies.colnames
        fill_z =  'z' not in galaxies.colnames
        fill_flux_radius = 'flux_radius' not in galaxies.colnames

        if fill_ra or fill_dec or fill_z or fill_flux_radius:
            if fill_ra:
                galaxies['ra'] = np.full(len(galaxies), np.nan)
            if fill_dec:
                galaxies['dec'] = np.full(len(galaxies), np.nan)
            if fill_z:
                galaxies['z'] = np.full(len(galaxies), np.nan)
            if fill_flux_radius:
                galaxies['flux_radius'] = np.full(len(galaxies), np.nan)

            catalog_names = np.unique(galaxies['catalog'])
            catalog_datas = {}
            for catalog_name in catalog_names:
                catalog_params = catalog.get_params(catalog_name)
                catalog_data = catalog.CatalogData(catalog_params)
                catalog_datas[catalog_name] = catalog_data

                for i in range(len(galaxies)):
                    if galaxies['catalog'][i] != catalog_name:
                        continue

                    galaxy_id = galaxies['id'][i]
                    idx = catalog_data.get_index(galaxy_id)

                    if np.size(idx) == 0:
                        raise TargetsException(f"Missing id {galaxy_id} in catalog {catalog_name}")
                    if np.size(idx) > 1:
                        raise TargetsException(f"Multiple ids {galaxy_id} in catalog {catalog_name}")

                    if table.has_field(catalog_data, 'best'):
                        if fill_ra:
                            galaxies['ra'][i] = catalog_data.best['ra'][idx] # pylint: disable=no-member
                        if fill_dec:
                            galaxies['dec'][i] = catalog_data.best['dec'][idx] # pylint: disable=no-member
                        if fill_z:
                            galaxies['z'][i] = catalog_data.best['z_best'][idx] # pylint: disable=no-member
                        if fill_flux_radius and catalog.get_has_flux_radius(catalog_data):
                            galaxies['flux_radius'][i] = catalog_data.best['flux_radius'][idx] # pylint: disable=no-member
                    else:
                        if fill_ra:
                            galaxies['ra'][i] = catalog_data.sources[catalog_data.ra_field][idx]
                        if fill_dec:
                            galaxies['dec'][i] = catalog_data.sources[catalog_data.dec_field][idx]
                        if fill_z:
                            galaxies['z'][i] = catalog.get_redshift_any(catalog_data, idx)
                        if fill_flux_radius and catalog.get_has_flux_radius(catalog_data):
                            galaxies['flux_radius'][i] = catalog_data.sources[catalog.get_flux_radius_field(catalog_data)][idx] * catalog.get_flux_radius_factor(catalog_data)

                    if catalog.get_is_flux_radius_kron(catalog_data):
                        galaxies['flux_radius'][i] /= 2.0 # Kron radius is too large

    # Check for required columns
    if 'ra' not in galaxies.colnames:
        raise TargetsException('Missing ra column')
    if 'dec' not in galaxies.colnames:
        raise TargetsException('Missing dec column')
    if 'z' not in galaxies.colnames:
        raise TargetsException('Missing z column')

    # Add default columns
    if 'z_unc' not in galaxies.colnames or np.any(np.isnan(galaxies['z_unc'])):
        if 'default_dz' not in options:
            raise TargetsException('default_dz is required but missing in options')
        if 'z_unc' not in galaxies.colnames:
            galaxies['z_unc'] = options['default_dz'] * (1 + galaxies['z'])
        else:
            missing_filter = np.isnan(galaxies['z_unc'])
            galaxies['z_unc'][missing_filter] = options['default_dz'] * (1 + galaxies['z'][missing_filter])
    if 'flux_radius' not in galaxies.colnames or np.any(np.isnan(galaxies['flux_radius'])):
        if 'default_flux_radius' not in options:
            raise TargetsException('default_flux_radius is required but missing in options')
        if 'flux_radius' not in galaxies.colnames:
            galaxies['flux_radius'] = options['default_flux_radius']
        else:
            missing_filter = np.isnan(galaxies['flux_radius'])
            galaxies['flux_radius'][missing_filter] = options['default_flux_radius']

    # AO Stars
    if options['wfs_band'] is not None:
        galaxies['ngs_count'] = np.zeros(len(galaxies))
        galaxies['ngs_best_sep'] = np.full(len(galaxies), np.nan)
        galaxies['ngs_best_mag'] = np.full(len(galaxies), np.nan)

        for i in range(len(galaxies)):
            stars = find_nearby_stars(galaxies['ra'][i], galaxies['dec'][i], options)
            galaxies['ngs_count'][i] = len(stars)
            if len(stars) > 0:
                galaxies['ngs_best_sep'][i] = np.min(stars['sep'])
                galaxies['ngs_best_mag'][i] = np.min(stars[f"gaia_{options['wfs_band']}"])

    # Emission Lines
    if options['lines'] is None or len(options['lines']) == 0:
        lines = None
    else:
        rest_lambda = sky.get_emission_line_rest_wavelengths()
        if np.any([name not in rest_lambda for name in options['lines']]):
            raise TargetsException(f"Unknown emission line in options: {options['lines']}")

        lines = StructType()
        lines.names = np.array(options['lines'])
        lines.wavelength_rest = np.array([rest_lambda[name] for name in options['lines']])
        lines.wavelength_vac = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.wavelength_atm = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.transmission   = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.fwhm           = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.dwavelength    = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.sigma          = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.flux           = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.flux_type      = np.zeros((len(galaxies), len(lines.names)), dtype=np.int_)
        lines.sb             = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.ph_energy      = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.ph_rate        = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.sky_rate       = np.full((len(galaxies), len(lines.names)), np.nan)
        lines.reject         = np.zeros((len(galaxies), len(lines.names)), dtype=np.bool)

        sky_transmission_data = sky.load_transmission_data(options['location'], options['airmass'])
        sky_background_data = sky.load_background_data(options['location'], options['airmass'])

        # Sort lines so that Ha comes before NIIa, NIIb but after all other lines
        sorted_indices = np.concatenate((
            np.where((lines.names != 'Ha') & (lines.names != 'NIIa') & (lines.names != 'NIIb'))[0],
            [np.where(lines.names == 'Ha')[0][0]],
            np.where((lines.names == 'NIIa') | (lines.names == 'NIIb'))[0]
        ))

        for i in sorted_indices:
            name = lines.names[i]

            lines.wavelength_vac[:,i] = lines.wavelength_rest[i] * (1 + galaxies['z'])
            lines.wavelength_atm[:,i] = sky.get_vacuum_to_air_wavelength(lines.wavelength_vac[:,i])

            if f"fwhm_{name}" in galaxies.colnames:
                lines.fwhm[:,i] = galaxies[f"fwhm_{name}"]
            elif 'dispersion' in galaxies.colnames:
                lines.fwhm[:,i] = lines.wavelength_atm[:,i] * galaxies['dispersion'] / constants.c.to('km/s').value
            elif 'default_dispersion' in options:
                lines.fwhm[:,i] = lines.wavelength_atm[:,i] * options['default_dispersion'] / constants.c.to('km/s').value

            lines.dwavelength[:,i] = np.sqrt((lines.wavelength_atm[:,i]/options['resolving_power'])**2 + lines.fwhm[:,i]**2)
            lines.sigma[:,i] = lines.dwavelength[:,i] / 2.35482 # FWHM -> sigma

            # Flux Measurements (Flux Type 1)
            if f"flux_{name}" in galaxies.colnames:
                lines.flux[:,i] = galaxies[f"flux_{name}"]
                missing_filter = np.isnan(lines.flux[:,i])
                lines.flux[missing_filter, i] = np.nan
                lines.flux_type[~missing_filter,i] = 1
            elif not skip_catalog:
                for catalog_name, catalog_data in catalog_datas.items():
                    if catalog.get_has_lines(catalog_data):
                        galaxy_filter = galaxies['catalog'] == catalog_name
                        catalog_filter = catalog_data.get_index(galaxies['id'][galaxy_filter])
                        lines.flux[galaxy_filter,i] = catalog.get_line_flux(catalog_data, name, catalog_filter, nan_if_empty=True) * 1e-17
                        lines.flux_type[galaxy_filter & ~np.isnan(lines.flux[:,i]),i] = 1

            # Ha Flux from Hb Flux (Flux Type 2)
            if name == 'Ha' and np.any(np.isnan(lines.flux[:,i])):
                missing_filter = np.isnan(lines.flux[:,i])

                if 'flux_Hb' in galaxies.colnames:
                    flux_Hb = galaxies['flux_Hb'][missing_filter]
                    Av = galaxies['Av'][missing_filter] if 'Av' in galaxies.colnames else 0.0
                    lines.flux[missing_filter,i] = catalog.calculate_flux_Ha_from_flux_Hb(flux_Hb, Av, null_if_empty=True)
                    lines.flux_type[missing_filter & ~np.isnan(lines.flux[missing_filter,i]),i] = 2

                elif not skip_catalog:
                    for catalog_name, catalog_data in catalog_datas.items():
                        if catalog.get_has_lines(catalog_data):
                            galaxy_filter = missing_filter & (galaxies['catalog'] == catalog_name)
                            catalog_filter = catalog_data.get_index(galaxies['id'][galaxy_filter])
                            if np.sum(galaxy_filter) > 0:
                                flux_Hb = catalog.get_line_flux(catalog_data, 'Hb', catalog_filter, nan_if_empty=True) * 1e-17
                                if catalog.get_has_spp(catalog_data):
                                    _, _, Av = catalog.get_spp_values(catalog_data, catalog_filter)
                                else:
                                    Av = 0.0
                                lines.flux[galaxy_filter,i] = catalog.calculate_flux_Ha_from_flux_Hb(flux_Hb, Av, null_if_empty=True)
                                lines.flux_type[galaxy_filter & ~np.isnan(lines.flux[:,i]),i] = 2

            # Ha Flux from OIII Flux (Flux Type 3)
            if name == 'Ha' and np.any(np.isnan(lines.flux[:,i])):
                missing_filter = np.isnan(lines.flux[:,i])

                if 'flux_OIII' in galaxies.colnames:
                    flux_OIII = galaxies['flux_OIII'][missing_filter]
                    Av = galaxies['Av'][missing_filter] if 'Av' in galaxies.colnames else 0.0
                    lines.flux[missing_filter,i] = catalog.calculate_flux_Ha_from_flux_OIII(flux_OIII, Av, null_if_empty=True)
                    lines.flux_type[missing_filter & ~np.isnan(lines.flux[missing_filter,i]),i] = 3

                elif not skip_catalog:
                    for catalog_name, catalog_data in catalog_datas.items():
                        if catalog.get_has_lines(catalog_data):
                            galaxy_filter = missing_filter & (galaxies['catalog'] == catalog_name)
                            catalog_filter = catalog_data.get_index(galaxies['id'][galaxy_filter])
                            if np.sum(galaxy_filter) > 0:
                                flux_OIII = catalog.get_line_flux(catalog_data, 'OIII', catalog_filter, nan_if_empty=True) * 1e-17
                                if catalog.get_has_spp(catalog_data):
                                    _, _, Av = catalog.get_spp_values(catalog_data, catalog_filter)
                                else:
                                    Av = 0.0
                                lines.flux[galaxy_filter,i] = catalog.calculate_flux_Ha_from_flux_OIII(flux_OIII, Av, null_if_empty=True)
                                lines.flux_type[galaxy_filter & ~np.isnan(lines.flux[:,i]),i] = 3

            # Ha Flux from OII Flux (Flux Type 4)
            if name == 'Ha' and np.any(np.isnan(lines.flux[:,i])):
                missing_filter = np.isnan(lines.flux[:,i])

                if 'flux_OII' in galaxies.colnames:
                    flux_OII = galaxies['flux_OII'][missing_filter]
                    Av = galaxies['Av'][missing_filter] if 'Av' in galaxies.colnames else 0.0
                    lines.flux[missing_filter,i] = catalog.calculate_flux_Ha_from_flux_OII(flux_OII, Av, null_if_empty=True)
                    lines.flux_type[missing_filter & ~np.isnan(lines.flux[missing_filter,i]),i] = 4

                elif not skip_catalog:
                    for catalog_name, catalog_data in catalog_datas.items():
                        if catalog.get_has_lines(catalog_data):
                            galaxy_filter = missing_filter & (galaxies['catalog'] == catalog_name)
                            catalog_filter = catalog_data.get_index(galaxies['id'][galaxy_filter])
                            if np.sum(galaxy_filter) > 0:
                                flux_OII = catalog.get_line_flux(catalog_data, 'OII', catalog_filter, nan_if_empty=True) * 1e-17
                                if catalog.get_has_spp(catalog_data):
                                    _, _, Av = catalog.get_spp_values(catalog_data, catalog_filter)
                                else:
                                    Av = 0.0
                                lines.flux[galaxy_filter,i] = catalog.calculate_flux_Ha_from_flux_OII(flux_OII, Av, null_if_empty=True)
                                lines.flux_type[galaxy_filter & ~np.isnan(lines.flux[:,i]),i] = 4

            # Ha Flux from SFR (Flux Type 5)
            if name == 'Ha' and np.any(np.isnan(lines.flux[:,i])):
                missing_filter = np.isnan(lines.flux[:,i])

                if 'sfr' in galaxies.colnames or 'lsfr' in galaxies.colnames:
                    if 'lsfr' in galaxies.colnames:
                        sfr = np.power(10, galaxies['lsfr'][missing_filter])
                    elif log_sfr:
                        sfr = np.power(10, galaxies['sfr'][missing_filter])
                    else:
                        sfr = galaxies['sfr'][missing_filter]
                    Av = galaxies['Av'][missing_filter] if 'Av' in galaxies.colnames else 0.0
                    z = galaxies['z'][missing_filter]
                    lines.flux[missing_filter,i] = catalog.compute_flux_Ha_from_SFR(sfr, Av, z, null_if_empty=True)
                    lines.flux_type[missing_filter & ~np.isnan(lines.flux[missing_filter,i]),i] = 5

                elif not skip_catalog:
                    for catalog_name, catalog_data in catalog_datas.items():
                        if catalog.get_has_spp(catalog_data):
                            galaxy_filter = missing_filter & (galaxies['catalog'] == catalog_name)
                            catalog_filter = catalog_data.get_index(galaxies['id'][galaxy_filter])
                            if np.sum(galaxy_filter) > 0:
                                _, lsfr, Av = catalog.get_spp_values(catalog_data, catalog_filter)
                                lines.flux[galaxy_filter,i] = catalog.compute_flux_Ha_from_SFR(np.power(10, lsfr), Av, catalog_filter, null_if_empty=True)
                                lines.flux_type[galaxy_filter,i] = 5

            # NII Flux from Ha
            if 'NII' in name and 'Ha' in options['lines'] and np.any(np.isnan(lines.flux[:,i])):
                Ha_idx = np.where(lines.names == 'Ha')[0][0]
                lines.flux[:,i] = lines.flux[:,Ha_idx] * FLUX_NII_HA_RATIO
                lines.flux_type[:,i] = lines.flux_type[:,Ha_idx]

            # Default Flux (Flux Type 0)
            if np.any(np.isnan(lines.flux[:,i])):
                if f"default_flux_{name}" in options:
                    missing_filter = np.isnan(lines.flux[:,i])
                    lines.flux[missing_filter,i] = options[f"default_flux_{name}"]
                    lines.flux_type[missing_filter,i] = 0
                else:
                    raise TargetsException(f"Missing default_flux_{name} in options")

            lines.sb[:,i] = lines.flux[:,i]/(np.pi*galaxies['flux_radius']**2) # erg/s/cm^2/arcsec^2
            lines.ph_energy[:,i] = 6.626e-27 * 2.998e10 / (lines.wavelength_atm[:,i]/1e8) # erg
            lines.ph_rate[:,i] = lines.sb[:,i] / lines.ph_energy[:,i] / (lines.fwhm[:,i]/10) * 100**2  # ph/s/arcsec^2/nm/m^2

            lines.transmission[:,i] = sky.get_mean_transmission(sky_transmission_data, lines.wavelength_atm[:,i], lines.fwhm[:,i], options['resolving_power'], options['trans_multiple'])
            lines.sky_rate[:,i] = sky.get_background(sky_background_data, lines.wavelength_atm[:,i], [lines.wavelength_atm[:,i] - lines.fwhm[:,i]*10, lines.wavelength_atm[:,i] + lines.fwhm[:,i]*10], options['resolving_power'])
            lines.reject[:,i] = sky.reject_emission_line(sky_background_data, sky_transmission_data, lines.wavelength_atm[:,i], lines.fwhm[:,i], options['resolving_power'], options['wavelength_ranges'], options['trans_minimum'], options['trans_multiple'], options['avoid_multiple'], options['min_photon_rate'])

        for i, name in enumerate(lines.names):
            galaxies[f"trans_{name}"] = lines.transmission[:,i]
            galaxies[f"reject_{name}"] = lines.reject[:,i]

    if not skip_catalog:
        for catalog_data in catalog_datas.values():
            catalog_data.close()

    return galaxies, lines, options

def _get_output_path():
    return f"{os.path.dirname(__file__)}/../output"

def exists_details(experiment):
    return os.path.isfile(f"{_get_output_path()}/{experiment}-galaxies.ecsv") \
       and os.path.isfile(f"{_get_output_path()}/{experiment}-lines.pkl") \
       and os.path.isfile(f"{_get_output_path()}/{experiment}-options.pkl")

def save_details(experiment, galaxies, lines, options):
    output_path = _get_output_path()

    galaxies.write(f"{output_path}/{experiment}-galaxies.ecsv", format='ascii.ecsv', overwrite=True)

    lines_path = f"{output_path}/{experiment}-lines.pkl"
    with open(lines_path, 'wb') as f:
        pickle.dump(lines, f, protocol=pickle.HIGHEST_PROTOCOL)

    options_path = f"{output_path}/{experiment}-options.pkl"
    with open(options_path, 'wb') as f:
        pickle.dump(options, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_details(experiment, id=None, index=None): # pylint: disable=redefined-builtin
    output_path = _get_output_path()

    galaxies = Table.read(f"{output_path}/{experiment}-galaxies.ecsv", format='ascii.ecsv')

    lines_path = f"{output_path}/{experiment}-lines.pkl"
    with open(lines_path, 'rb') as f:
        lines = pickle.load(f)

    options_path = f"{output_path}/{experiment}-options.pkl"
    with open(options_path, 'rb') as f:
        options = pickle.load(f)

    if id is not None or index is not None:
        if index is not None:
            if id is not None and galaxies['id'][index] != id:
                raise TargetsException('galaxy index and id do not match, provide only one')
            galaxy_index = index
        else:
            galaxy_index = np.where(galaxies['id'] == id)[0][0]

        galaxies, lines = _select_target(galaxies, lines, galaxy_index)

    return galaxies, lines, options

def _select_target(galaxies, lines, galaxy_index=0):
    if galaxy_index > np.size(galaxies) - 1:
        raise TargetsException(f"Invalid galaxy index: {galaxy_index}")

    galaxies = galaxies[galaxy_index]

    for attr in lines.__dict__.keys():
        if isinstance(lines.__dict__[attr], np.ndarray) and np.ndim(lines.__dict__[attr]) == 2:
            lines.__dict__[attr] = lines.__dict__[attr][galaxy_index,:]

    return galaxies, lines

def print_summary(galaxies, lines, options, galaxy_index=None, line_name=None):
    if galaxy_index is not None:
        galaxy, lines = _select_target(galaxies, lines, galaxy_index)
    elif np.size(galaxies) > 1:
        galaxy, lines = _select_target(galaxies, lines, 0)
    else:
        galaxy = galaxies

    print(f"Target ID: {galaxy['id']} ({galaxy['catalog']})")
    print(f"       RA: {galaxy['ra']:.6f} deg")
    print(f"      Dec: {galaxy['dec']:.6f} deg")
    print(f"        z: {galaxy['z']:.4f} ± {galaxy['z_unc']:.4f}")

    if line_name is not None:
        line_index = np.where(lines.names == line_name)[0][0]
        print(f"        R: {options['resolving_power']:.0f}")
        print(f"  dLambda: {lines.dwavelength[line_index]/10:.1f} nm")
        print(f"  {line_name} λvac: {lines.wavelength_vac[line_index]/10:.2f} nm")
        print(f"  {line_name} λatm: {lines.wavelength_atm[line_index]/10:.2f} nm")
        print(f"  {line_name} FWHM: {lines.fwhm[line_index]/10:.2f} nm")
        print(f"  {line_name} Flux: {lines.flux[line_index]:.1e} erg/s/cm^2")
        print(f"     Area: {np.pi*galaxy['flux_radius']**2:.1e} arcsec^2 [radius={galaxy['flux_radius']:.1f} arcsec]")
        print(f"  {line_name} SB  : {lines.sb[line_index]:.1e} erg/s/cm^2/arcsec^2  [{lines.sb[line_index]/1000:.1e} W/m^2/arcsec^2    ]")
        print(f"  {line_name} Rate: {lines.ph_rate[line_index]:.1f}     ph/s/arcsec^2/nm/m^2 [{lines.ph_rate[line_index]*lines.ph_energy[line_index]*1e-7*1e9:.1e} J/s/m^2/m/arcsec^2]")
        print(f" Sky Rate: {lines.sky_rate[line_index]:.1f}     ph/s/arcsec^2/nm/m^2 [{lines.sky_rate[line_index]*lines.ph_energy[line_index]*1e-7*1e9:.1e} J/s/m^2/m/arcsec^2]")

#endregion

#region Stars

SEARCH_DISTANCE_FACTOR = 4.0
GAIA_MAPS_LEVEL = None
FLUX_NII_HA_RATIO = 0.1

def _get_data_path():
    return f"{os.path.dirname(__file__)}/../data"

def _query_stars(ra, dec, min_mag=-2.0, max_mag=25.0, min_sep=0.0, max_sep=60.0, wfs_band='R', epoch=None, skip_cache=False, verbose=False):
    search_distance = max_sep/3600*SEARCH_DISTANCE_FACTOR

    if not skip_cache:
        stars = None

        # Try local copy of Gaia database
        maps_gaia_path = f"{_get_data_path()}/maps/gaia"
        if os.path.isdir(maps_gaia_path):
            global GAIA_MAPS_LEVEL # pylint: disable=global-statement
            if GAIA_MAPS_LEVEL is None:
                for level in range(14, 0, -1):
                    if os.path.isdir(f"{maps_gaia_path}/hpx{level}"):
                        GAIA_MAPS_LEVEL = level
                        break
            if GAIA_MAPS_LEVEL is not None:
                outer_pix = healpix.get_healpix_from_skycoord(GAIA_MAPS_LEVEL, SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree)))
                coord = healpix.get_pixel_skycoord(GAIA_MAPS_LEVEL, outer_pix)
                cache_filename = f"{maps_gaia_path}/hpx{GAIA_MAPS_LEVEL}//{int(coord.ra.degree/15)}h/{'+' if coord.dec.degree >= 0 else '-'}{int(np.abs(coord.dec.degree/10))*10:02}/{outer_pix}/gaia.fits"
                if os.path.isfile(cache_filename):
                    with fits.open(cache_filename) as hdul:
                        stars = Table(hdul[1].data) # pylint: disable=no-member

        # Try cached Gaia data
        if stars is None:
            cache_filename = f"{_get_data_path()}/cache/targets/stars-{ra*1e6:.0f}_{dec*1e6:.0f}_{search_distance*1e6:.0f}.fits"
            if os.path.isfile(cache_filename):
                with fits.open(cache_filename) as hdul:
                    stars = Table(hdul[1].data) # pylint: disable=no-member
            else:
                stars = gaia.get_stars_by_ra_dec_distance(ra, dec, search_distance, verbose=verbose)
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                stars.write(cache_filename, format='fits')
    else:
        stars = gaia.get_stars_by_ra_dec_distance(ra, dec, search_distance, verbose=verbose)

    if len(stars) == 0:
        return stars

    if epoch is not None:
        (stars['obs_ra'], stars['obs_dec']) = gaia.apply_proper_motion(stars, epoch=epoch)
        star_coords = SkyCoord(ra=stars['obs_ra'], dec=stars['obs_dec'], unit=(u.degree, u.degree))
    else:
        star_coords = SkyCoord(ra=stars['gaia_ra'], dec=stars['gaia_dec'], unit=(u.degree, u.degree))

    target_coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))
    stars['sep'] = target_coords.separation(star_coords).arcsec

    mag_field = f"gaia_{wfs_band}"
    mag_filter = (stars[mag_field] >= min_mag) & (stars[mag_field] < max_mag)
    sep_filter = (stars['sep'] >= min_sep) & (stars['sep'] < max_sep)

    return stars[mag_filter & sep_filter].copy()

def count_nearby_stars(ra, dec, options, skip_cache=False, verbose=False):
    stars = _query_stars(ra, dec, wfs_band=options['wfs_band'], min_mag=options['min_ngs_mag'], max_mag=options['max_ngs_mag'], min_sep=options['min_ngs_sep'], max_sep=options['max_ngs_sep'], epoch=options['obs_epoch'], skip_cache=skip_cache, verbose=verbose)
    return len(stars)

def find_nearby_stars(ra, dec, options, skip_cache=False, verbose=False):
    stars = _query_stars(ra, dec, wfs_band=options['wfs_band'], min_mag=options['min_ngs_mag'], max_mag=options['max_ngs_mag'], min_sep=options['min_ngs_sep'], max_sep=options['max_ngs_sep'], epoch=options['obs_epoch'], skip_cache=skip_cache, verbose=verbose)
    if len(stars) > 0:
        stars.sort('sep')
    return stars

def fix_gaia_ids_for_display(stars):
    stars = stars.copy()
    stars['gaia_id'] = stars['gaia_id'].astype(str)
    return stars

#endregion

#region Spectra

def plot_catalog_spectra(field, galaxy, catalog_ids, figsize=None, lines=None, options=None, show_emission_lines=True, show_skylines=None):
    num_plots = 0

    for catalog_name in catalog.get_spectra_sources():
        if catalog_name in catalog_ids['catalog']:
            source_id = catalog_ids['id'][catalog_ids['catalog'] == catalog_name][0]
            spectra = catalog.get_spectra(catalog_name, field, source_id, galaxy['ra'], galaxy['dec'])
            if spectra is not None and len(spectra) > 0:
                for spectrum in spectra:
                    _plot_spectra(galaxy, spectrum, figsize=figsize, options=options, lines=lines, show_emission_lines=show_emission_lines, show_skylines=show_skylines)
                    num_plots += 1
            else:
                print(f"No spectra found for {catalog_name} {source_id:.0f}")

    if num_plots == 0:
        print('No spectra found')

def _plot_spectra(galaxy, spectrum, figsize=None, options=None, lines=None, show_emission_lines=True, show_skylines=None):
    is_space_based = catalog.get_is_space_based(spectrum.catalog_source)

    wavelength = spectrum.wavelength

    if table.has_field(spectrum, 'flux_1d'):
        flux_1d = spectrum.flux_1d
    else:
        flux_1d = spectrum.flux

    has2d = table.has_field(spectrum, 'flux_2d') and spectrum.flux_2d is not None
    if has2d:
        flux_2d = spectrum.flux_2d

    spectral_resolution = np.mean(wavelength)/(wavelength[1]-wavelength[0])
    if spectral_resolution < 2000: # low resolution
        lines = None
    elif lines is not None:
        min_wavelength = np.min(lines.wavelength_atm - 10*lines.fwhm)
        max_wavelength = np.max(lines.wavelength_atm + 10*lines.fwhm)

        if (min_wavelength < wavelength[0] and max_wavelength < wavelength[0]) or (min_wavelength > wavelength[-1] and max_wavelength > wavelength[-1]):
            lines = None
        else:
            wavelength_filter = (wavelength >= min_wavelength) & (wavelength <= max_wavelength)

    if lines is None:
        if table.has_field(spectrum, 'zero_padded') and spectrum.zero_padded:
            non_zero_values = np.where(flux_1d != 0)[0]
            if len(non_zero_values) != len(flux_1d):
                flux_1d[:non_zero_values[0]] = np.nan
                flux_1d[non_zero_values[-1]+1:] = np.nan

        value_filter = ~np.isnan(flux_1d)

        min_wavelength = wavelength[np.where(value_filter)[0][0]]
        max_wavelength = wavelength[np.where(value_filter)[0][-1]]
        wavelength_filter = np.ones((len(wavelength)), dtype=bool)

        outlier_filter = fit.is_outlier(flux_1d, num_sigma=5)
        flux_1d[outlier_filter] = np.nan
        if has2d:
            flux_2d[:,outlier_filter] = np.nan

    wavelength_range = max_wavelength - min_wavelength
    ymax = np.nanmax(flux_1d[wavelength_filter]) # pylint: disable=possibly-used-before-assignment
    ymin = np.nanmin(flux_1d[wavelength_filter])
    xnudge = wavelength[1] - wavelength[0]

    if show_skylines is None: # default
        show_skylines = not is_space_based and wavelength_range < 1000

    if has2d:
        subplots = (2,1)
        height_ratios = (1,4)
    else:
        subplots = None
        height_ratios = None

    if table.has_field(spectrum, 'source_name') and table.has_field(spectrum, 'source_id'):
        if f"{spectrum.source_id:.0f}" in spectrum.source_name:
            title = f"{spectrum.catalog_name} {spectrum.source_name}"
        else:
            title = f"{spectrum.catalog_name} {spectrum.source_name} ({spectrum.source_id:.0f})"
    elif table.has_field(spectrum, 'source_name'):
        title = f"{spectrum.catalog_name} {spectrum.source_name}"
    elif table.has_field(spectrum, 'source_id'):
        title = f"{spectrum.catalog_name} {spectrum.source_id:.0f}"
    else:
        title = spectrum.catalog_name

    _, ax = plot.create_plot(subplots=subplots, height_ratios=height_ratios, figsize=figsize, title=title)

    if has2d:
        ax2d = plt.subplot(2, 1, 1)
        ax2d.imshow(
            flux_2d[:,wavelength_filter],
            aspect="auto",
            origin="lower",
            extent=[min_wavelength, max_wavelength, 0, flux_2d.shape[0]],
            vmin=ymax * -0.1,
            vmax=ymax * 0.4,
        )
        ax2d.set_xlim(min_wavelength, max_wavelength)
        ax2d.set_ylim(0, flux_2d.shape[0])
        ax2d.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax2d.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        ax1d = plt.subplot(2, 1, 2)
    else:
        ax1d = ax

    ax1d.plot(wavelength[wavelength_filter], flux_1d[wavelength_filter], zorder=3)
    ax1d.set_xlim(min_wavelength, max_wavelength)
    ax1d.set_xlabel('Wavelength [Angstrom]')

    if show_emission_lines:
        if lines is not None:
            line_names = lines.names
            line_wavelengths = lines.wavelength_atm
        else:
            lines = sky.get_emission_line_rest_wavelengths(skip_close_doublets=True)
            line_names = list(lines.keys())
            line_wavelength_rest = np.array(list(lines.values()))
            line_wavelength_vac = line_wavelength_rest * (1 + galaxy['z'])
            line_wavelengths = sky.get_vacuum_to_air_wavelength(line_wavelength_vac)

        for i, line_name in enumerate(line_names):
            if line_wavelengths[i] >= min_wavelength and line_wavelengths[i] <= max_wavelength:
                ax1d.axvline(line_wavelengths[i], c='r', linestyle='-', zorder=2)
                ax1d.text(line_wavelengths[i] + xnudge, ymin, line_name, rotation=90, color='r')

    if show_skylines:
        if options is None:
            raise TargetsException('options is required when show_skylines is True')

        sky_background_data = sky.load_background_data(options['location'], options['airmass'])
        sky_background_data_low_res = sky.get_low_res_background(sky_background_data, [min_wavelength, max_wavelength], options['resolving_power'])
        if sky_background_data_low_res is not None:
            sky_lines = sky.find_sky_lines(sky_background_data_low_res, options['min_photon_rate'])
            if len(sky_lines) > 0:
                for w in sky_lines['wavelength']:
                    ax1d.axvline(w, color='k', linestyle=':', zorder=1)

    plt.show()

#endregion

#region Cutouts

def get_catalog_filters(field, galaxy):
    source = catalog.get_image_source(galaxy['catalog'])
    filters = catalog.get_image_filters(galaxy['catalog'], field)
    return Table([np.repeat(source, len(filters)), filters], names=['source', 'filter'])

def get_space_filters(field, galaxy,
                      field_radius=6.0 # arcsec
    ):
    if field != 'COSMOS':
        raise TargetsException(f"Unsupported field: {field}")

    mode = 'count' # files
    response = json.loads(urllib.request.urlopen(f"https://grizli-cutout.herokuapp.com/overlap?ra={galaxy['ra']}&dec={galaxy['dec']}&size={field_radius}&mode={mode}").read())
    filters = response['filters']
    if len(filters) == 0:
        return None

    filter_integration = []
    filter_image_count = []
    for filter_name in filters:
        filter_details = response[filter_name]
        filter_integration.append(filter_details[0])
        filter_image_count.append(filter_details[1])

    return Table([filters, filter_integration, filter_image_count], names=['filter', 'total_time', 'image_count'])

def get_space_cutout(field, galaxy, filter_name,
                     field_radius=3, # arcsec
                     skip_cache=False
    ):
    if field != 'COSMOS':
        raise TargetsException('Decon on available for COSMOS')

    position = SkyCoord(ra=galaxy['ra'], dec=galaxy['dec'], unit=(u.degree, u.degree))
    cutout_radius = field_radius * 1.5 # >= sqrt(2) needed to account for all possible rotations

    cache_filename = f"{_get_data_path()}/cache/cutouts/{galaxy['catalog']}-{galaxy['id']}-{filter_name}-{cutout_radius}.fits"
    if os.path.isfile(cache_filename) and not skip_cache:
        with fits.open(cache_filename) as hdul:
            image_data = hdul[0].data # pylint: disable=no-member
            image_wcs = WCS(hdul[0].header) # pylint: disable=no-member
    else:
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        url = f"https://grizli-cutout.herokuapp.com/thumb?ra={position.ra.to_value(u.degree)}&dec={position.dec.to_value(u.degree)}&size={cutout_radius}&filters={filter_name}&output=fits"
        with fits.open(url) as hdul:
            hdul.writeto(cache_filename, overwrite=skip_cache)
            image_data = deepcopy(hdul[0].data) # pylint: disable=no-member
            image_wcs = deepcopy(WCS(hdul[0].header)) # pylint: disable=no-member

    pixel_scale = WCS.proj_plane_pixel_scales(image_wcs)

    space_cutout = StructType()
    space_cutout.name = filter_name
    space_cutout.position = position
    space_cutout.field_radius = field_radius
    space_cutout.cutout_radius = cutout_radius
    space_cutout.data = image_data
    space_cutout.wcs = image_wcs
    space_cutout.pixel_scale = [pixel_scale[0].to_value(u.arcsec), pixel_scale[1].to_value(u.arcsec)]
    return space_cutout

def get_catalog_cutout(field, galaxy, options, filter_name,
                       field_radius=None # arcsec
    ):

    if field_radius is None:
        if 'field_diameter' in options:
            field_radius = options['field_diameter'] / 2
        else:
            raise TargetsException('field_radius is required')

    position = SkyCoord(ra=galaxy['ra'], dec=galaxy['dec'], unit=(u.degree, u.degree))
    cutout_radius = field_radius * 1.25

    catalog_params = catalog.get_params(galaxy['catalog'], field, filter_name)
    with fits.open(catalog_params.catalog_image_path) as image_hdul:
        header = image_hdul[0].header # pylint: disable=no-member
        epoch = 1858.8785 + (header['MJD-END'] + header['MJD-OBS']) / 2 / 365.25
        cutout = Cutout2D(image_hdul[0].data, position, 2*cutout_radius*u.arcsec, wcs=catalog_params.wcs, mode='partial', fill_value=0.0) # pylint: disable=no-member

    pixel_scale = WCS.proj_plane_pixel_scales(cutout.wcs)

    catalog_cutout = StructType()
    catalog_cutout.name = f"{catalog.get_image_source(galaxy['catalog'])} {filter_name}"
    catalog_cutout.position = position
    catalog_cutout.field_radius = field_radius
    catalog_cutout.cutout_radius = cutout_radius
    catalog_cutout.data = cutout.data
    catalog_cutout.wcs = cutout.wcs
    catalog_cutout.pixel_scale = [pixel_scale[0].to_value(u.arcsec), pixel_scale[1].to_value(u.arcsec)]
    catalog_cutout.epoch = catalog_params.catalog_image_epoch if catalog_params.catalog_image_epoch is not None else epoch
    return catalog_cutout

def get_decon_cutout(field, galaxy):
    if field != 'COSMOS':
        raise TargetsException('Decon on available for COSMOS')

    decon_path = f"{_get_data_path()}/decon/{galaxy['id']}.npy"
    if not os.path.isfile(decon_path):
        raise TargetsException(f"Missing decon file: {decon_path}")

    image_data = np.load(decon_path)

    # values should be between 0 and 1 but some values are just outside this range
    image_data[image_data < 0] = 0
    image_data[image_data > 1] = 1

    if 'UVISTA' in galaxy['catalog']:
        position = SkyCoord(ra=galaxy['ra'], dec=galaxy['dec'], unit=(u.degree, u.degree))
    else:
        position = None

    pixel_size = 0.05 # assume pixels are 0.05"

    decon_cutout = StructType()
    decon_cutout.name = 'Decon VzK'
    decon_cutout.position = position
    decon_cutout.field_radius = np.size(image_data, axis=0) * pixel_size / 2
    decon_cutout.cutout_radius = decon_cutout.field_radius
    decon_cutout.data = image_data
    decon_cutout.wcs = None
    decon_cutout.pixel_scale = [pixel_size, pixel_size]
    return decon_cutout

def get_ifu_position(galaxy, options,
                     relative_ifu=None,
                     pa=0.0,          # degrees
                     p=0.0,           # arcsec
                     q=0.0,           # arcsec
                     sp=0.0,          # spaxels
                     sq=0.0,          # spaxels
                     silent=False
    ):

    if not options['instrument_is_ifu']:
        raise TargetsException(f"{options['instrument']} is not a recognized IFU")

    if (p != 0 or q != 0) and (sp != 0 or sq != 0):
        raise TargetsException('Cannot specify both p/q and sp/sq')

    if (sp != 0 or sq != 0):
        p = sp * options['ifu_spaxel_width']
        q = sq * options['ifu_spaxel_height']

    ra = galaxy['ra']
    dec = galaxy['dec']

    if relative_ifu is not None:
        p  += relative_ifu.p
        q  += relative_ifu.q
        pa += relative_ifu.pa

    if (p != 0 or q != 0):
        ifu_ra  = np.round(ra  - p/3600*np.cos(pa/180*np.pi)/np.cos(dec/180*np.pi) + q/3600*np.sin(pa/180*np.pi), 6)
        ifu_dec = np.round(dec - p/3600*np.sin(pa/180*np.pi)/np.cos(dec/180*np.pi) + q/3600*np.cos(pa/180*np.pi), 7)
        if not silent:
            print(f"Source Center: ra={ra:.6f}, dec={dec:.7f}")
            print(f"   IFU Center: ra={ifu_ra:.6f}, dec={ifu_dec:.7f}")
    else:
        ifu_ra  = ra
        ifu_dec = dec

    ifu = StructType()
    ifu.position = SkyCoord(ra=ifu_ra, dec=ifu_dec, unit=(u.degree, u.degree))
    ifu.pa = pa
    ifu.p = p
    ifu.q = q
    return ifu

def plot_galaxy_cutouts(galaxy, options, cutouts,
                        field_radius=3.0, # arcsec
                        resolution=None, # arcsec
                        segmap=False,
                        ifu=None,
                        ifus=None,
                        ifu_grid=False,
                        plot_bkg=False,
                        segmap_threshold=2.0,
                        segmap_pixels=50,
                        asinh_a=0.1,
                        max_cols=4,
                        hide_title=False,
                        center_title=False,
                        fontsize=20,
                        fontweight='bold',
                        fontcolor='yellow',
                        hide_annotations=False,
                        annotation_fontsize=10,
                        annotation_color='yellow',
                        pa_arrow_length=0.025,
                        pa_arrow_width=0.002,
                        ifu_color='yellow',
                        silent=False
    ):

    if ifu is not None and ifus is not None:
        raise TargetsException('Cannot specify both ifu and ifus')

    if ifus is not None and not isinstance(ifus, list):
        raise TargetsException('ifus must be a list')

    if not isinstance(cutouts, list):
        cutouts = [cutouts]

    if ifus is not None and len(ifus) > 1 and len(cutouts) > 1:
        raise TargetsException('Cannot specify multiple ifus with multiple cutouts')

    plot_ifu = ifus is not None or (ifu is not None and (not isinstance(ifu, bool) or ifu))

    if ifus is None:
        if plot_ifu:
            ifus = [ifu]
            del ifu
        else:
            ifus = [False]

    if resolution == 0:
        resolution = None

    if len(ifus) > 1:
        ncols = len(ifus)
    else:
        ncols = len(cutouts)

    if plot_bkg:
        ncols += 1
    if ncols > max_cols:
        nrows = int(np.ceil(ncols / max_cols))
        ncols = max_cols
    else:
        nrows = 1

    _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols,5*nrows), sharex=False, sharey=False, constrained_layout=True)
    if nrows == 1 and ncols == 1:
        ax = axs
        del axs

    position = SkyCoord(ra=galaxy['ra'], dec=galaxy['dec'], unit=(u.degree, u.degree))
    bkg_values = None

    row_num = 0
    col_num = -1
    for cutout in cutouts:
        if field_radius > cutout.field_radius:
            raise TargetsException(f"field_radius ({field_radius:.1f}) must be less than or equal to {cutout.name} field_radius ({cutout.field_radius:.1f})")

        image_data = cutout.data
        image_wcs = cutout.wcs
        pixel_scale = cutout.pixel_scale

        for ifu in ifus:
            # Rotate Image
            if isinstance(ifu, StructType) and ifu.pa != 0.0:
                if image_wcs is None:
                    rotated_data = rotate(image_data, ifu.pa)
                    rotated_wcs = None
                else:
                    rot_point1 = [image_data.shape[0]/2.0-1, image_data.shape[1]/2.0-1]
                    rotated_data = rotate(image_data, ifu.pa)
                    rot_point2 = [rotated_data.shape[0]/2.0-1, rotated_data.shape[1]/2.0-1]

                    rot_matrix = [[np.cos(np.radians(ifu.pa)), -np.sin(np.radians(ifu.pa))], [np.sin(np.radians(ifu.pa)),  np.cos(np.radians(ifu.pa))]]
                    rev_rot_matrix = [[np.cos(np.radians(-ifu.pa)), -np.sin(np.radians(-ifu.pa))], [np.sin(np.radians(-ifu.pa)),  np.cos(np.radians(-ifu.pa))]]

                    rotated_wcs = deepcopy(image_wcs)
                    rotated_wcs.wcs.crpix = np.transpose(np.matmul(rev_rot_matrix, np.transpose(image_wcs.wcs.crpix - rot_point1))) + rot_point2
                    rotated_wcs.wcs.cd = np.dot(image_wcs.wcs.cd, rot_matrix)
            else:
                rotated_data = image_data
                rotated_wcs = image_wcs

            # Crop to Field Size
            if rotated_wcs is None:
                crop_radius = int(field_radius / pixel_scale[0])
                startx = rotated_data.shape[0]//2 - crop_radius
                starty = rotated_data.shape[1]//2 - crop_radius
                plot_data = rotated_data[starty:starty+2*crop_radius, startx:startx+2*crop_radius, :]
                plot_wcs = None
            else:
                plot_cutout = Cutout2D(rotated_data, position, 2*field_radius*u.arcsec, rotated_wcs)
                plot_data = plot_cutout.data
                plot_wcs = plot_cutout.wcs

            # Set Position Angle
            if plot_wcs is not None:
                plot_pa = np.degrees(np.arctan2(plot_wcs.wcs.cd[1,0], plot_wcs.wcs.cd[1,1]))
            elif isinstance(ifu, StructType):
                plot_pa = ifu.pa
            else:
                plot_pa = 0.0

            # Measure Background
            bkg_mean, _, bkg_std = sigma_clipped_stats(plot_data)

            # If not an RGB image
            if np.ndim(plot_data) != 3:
                # Change Resolution
                if resolution is not None:
                    fwhm_pixels = int(np.round(resolution / pixel_scale[1]))

                    method = 'zoom'
                    if method == 'zoom':
                        plot_data_no_bkg = zoom(zoom(plot_data - bkg_mean, 1/fwhm_pixels), fwhm_pixels)
                        plot_data = zoom(zoom(plot_data, 1/fwhm_pixels), fwhm_pixels)
                    else:
                        kernel_size = fwhm_pixels + np.mod(fwhm_pixels+1, 2)
                        kernel = make_2dgaussian_kernel(fwhm_pixels, size=kernel_size)
                        plot_data_no_bkg = convolve(plot_data - bkg_mean, kernel)
                        plot_data = convolve(plot_data, kernel)
                else:
                    plot_data_no_bkg = plot_data - bkg_mean

                # Make Segmentation Map (detect sources)
                if segmap or plot_bkg:
                    segment_map = detect_sources(plot_data_no_bkg, segmap_threshold * bkg_std, npixels=segmap_pixels).data
                    if plot_bkg:
                        bkg_values = plot_data.flatten()[segment_map.flatten() == 0]

                # Normalize Plot Data
                plot_data = ImageNormalize(stretch=AsinhStretch(a=asinh_a))(plot_data)

            # IFU Details
            if plot_ifu and plot_wcs is not None:
                if isinstance(ifu, StructType):
                    [ifu_x, ifu_y] = np.array(plot_wcs.world_to_pixel(ifu.position))
                else:
                    [ifu_x, ifu_y] = np.array(plot_wcs.world_to_pixel(position))
                ifu_position_px = PixCoord(x=ifu_x, y=ifu_y)
                ifu_width_px = options['ifu_width'] / pixel_scale[0]
                ifu_height_px = options['ifu_height'] / pixel_scale[1]

            # Plot Filter Image
            col_num += 1
            if col_num+1 > max_cols:
                col_num = 0
                row_num += 1

            if nrows > 1 or ncols > 1:
                ax = axs[row_num, col_num] if nrows > 1 else axs[col_num]

            ax.set_axis_off() # pylint: disable=possibly-used-before-assignment

            if np.ndim(plot_data) == 3:
                ax.imshow(plot_data, origin='lower')
            else:
                ax.imshow(plot_data, origin='lower', cmap='viridis')

            if not hide_title:
                title_location = 'bottom'
                if plot_ifu and hide_annotations:
                    top_px = np.size(plot_data, axis=1) - (ifu_position_px.y + ifu_height_px/2) # pylint: disable=possibly-used-before-assignment
                    bottom_px = ifu_position_px.y - ifu_height_px/2
                    if top_px > bottom_px:
                        title_location = 'top'

                if center_title:
                    title_x = 0.5
                else:
                    title_x = 0.05
                if title_location == 'top':
                    title_y = 0.92
                else:
                    title_y = 0.05

                if len(ifus) > 1 and table.has_field(ifu, 'name'):
                    image_name = ifu.name
                else:
                    image_name = cutout.name
                ax.text(title_x, title_y, image_name, transform=ax.transAxes, ha='center' if center_title else 'left', fontsize=fontsize, fontweight=fontweight, color=fontcolor)

            # Plot Position Angle Arrow
            if plot_pa != 0 and not hide_annotations:
                arrow_x = 0.90 if plot_pa >= 0 else 0.09
                arrow_y = 0.935
                arrow_head_width =  pa_arrow_length/2
                dx = -pa_arrow_length * np.sin(np.radians(-plot_pa))
                dy =  pa_arrow_length * np.cos(np.radians(-plot_pa))
                ax.add_patch(FancyArrow(arrow_x, arrow_y, dx, dy, width=pa_arrow_width, head_width=arrow_head_width, head_length=arrow_head_width, transform=ax.transAxes, edgecolor=annotation_color, facecolor=annotation_color))
                ax.text(arrow_x, arrow_y-0.025, f"{plot_pa:.1f}°", transform=ax.transAxes, ha='center', va='center', fontsize=annotation_fontsize, color=annotation_color)

                # Check Aligment
                check_alignment = False
                if check_alignment:
                    (x,y) = cutout.wcs.world_to_pixel(SkyCoord(ra=position.ra, dec=position.dec))
                    ax.add_patch(Circle((x,y), 1, facecolor='red', edgecolor='red', lw=1))
                    (x,y) = cutout.wcs.world_to_pixel(SkyCoord(ra=position.ra, dec=position.dec+0.25*u.arcsec))
                    ax.add_patch(Circle((x,y), 1, facecolor='red', edgecolor='red', lw=1))
                    (x,y) = cutout.wcs.world_to_pixel(SkyCoord(ra=position.ra, dec=position.dec+0.50*u.arcsec))
                    ax.add_patch(Circle((x,y), 1, facecolor='red', edgecolor='red', lw=1))
                    (x,y) = cutout.wcs.world_to_pixel(SkyCoord(ra=position.ra, dec=position.dec+0.75*u.arcsec))
                    ax.add_patch(Circle((x,y), 1, facecolor='red', edgecolor='red', lw=1))
                    (x,y) = cutout.wcs.world_to_pixel(SkyCoord(ra=position.ra, dec=position.dec+1.00*u.arcsec))
                    ax.add_patch(Circle((x,y), 1, facecolor='red', edgecolor='red', lw=1))

            if np.ndim(plot_data) == 3:
                continue

            # Plot Resolution Circle
            if resolution is not None and not hide_annotations:
                circle_radius = resolution/2.0/field_radius
                circle_x = 0.90 if plot_pa < 0 else 0.09
                circle_y = 0.935
                ax.add_patch(Circle((circle_x, circle_y+0.015), circle_radius, transform=ax.transAxes, facecolor='none', edgecolor=annotation_color, lw=1))
                ax.text(circle_x, circle_y-0.025, f"{resolution}\"", transform=ax.transAxes, ha='center', va='center', fontsize=annotation_fontsize, color=annotation_color)

            # Plot Segmentation Map
            if segmap:
                ax.contour(segment_map, origin="lower", colors="yellow", linewidths=1)

            if plot_ifu:
                ifu_spaxel_width_px = ifu_width_px / options['ifu_width_spx'] # pylint: disable=possibly-used-before-assignment
                ifu_spaxel_height_px = ifu_height_px / options['ifu_height_spx'] # pylint: disable=possibly-used-before-assignment

                if ifu_grid:
                    for j in np.arange(options['ifu_width_spx']):
                        ifu_grid_center = PixCoord(x = ifu_position_px.x - ifu_width_px/2 + ifu_spaxel_width_px * (0.5 + j), y = ifu_position_px.y) # pylint: disable=possibly-used-before-assignment
                        ifu_grid_region = RectanglePixelRegion(ifu_grid_center, width=ifu_spaxel_width_px, height=ifu_height_px)
                        ifu_grid_region.plot(ax=ax, facecolor='none', edgecolor=ifu_color, linewidth=0.5)

                    for j in np.arange(options['ifu_height_spx']):
                        ifu_grid_center = PixCoord(x = ifu_position_px.x, y = ifu_position_px.y - ifu_height_px/2 + ifu_spaxel_height_px * (0.5 + j))
                        ifu_grid_region = RectanglePixelRegion(ifu_grid_center, width=ifu_width_px, height=ifu_spaxel_height_px)
                        ifu_grid_region.plot(ax=ax, facecolor='none', edgecolor=ifu_color, linewidth=0.5)

                ifu_region = RectanglePixelRegion(ifu_position_px, width=ifu_width_px, height=ifu_height_px)
                ifu_region.plot(ax=ax, facecolor='none', edgecolor=ifu_color, linewidth=4)
                ax.add_patch(Circle((ifu_position_px.x, ifu_position_px.y), 0.5, edgecolor='red', facecolor='red', lw=0.5))

                if segmap:
                    ifu_segment_map = ifu_region.to_mask().get_values(segment_map)
                    ifu_source_pixels = np.sum(ifu_segment_map > 0)
                    ifu_background_pixels = np.sum(ifu_segment_map == 0)
                    ifu_source_fraction = ifu_source_pixels / (ifu_source_pixels + ifu_background_pixels)
                    ifu_background_fraction = ifu_background_pixels / (ifu_source_pixels + ifu_background_pixels)
                    ifu_source_spx = np.round(ifu_source_pixels / (ifu_spaxel_width_px*ifu_spaxel_height_px),1)
                    ifu_background_spx = np.round(ifu_background_pixels / (ifu_spaxel_width_px*ifu_spaxel_height_px),1)

                    if ifu_source_pixels < ifu_background_pixels:
                        ifu_spaxel_ratio_text = f"1 : {ifu_background_pixels / ifu_source_pixels:.1f}"
                    else:
                        ifu_spaxel_ratio_text = f"{ifu_source_pixels / ifu_background_pixels:.1f} : 1"

                    if not silent:
                        print(f"  IFU Spaxels: {ifu_source_spx} ({ifu_source_fraction*100:.0f}%) source")
                        print(f"               {ifu_background_spx} ({ifu_background_fraction*100:.0f}%) sky")
                        print(f"               {ifu_spaxel_ratio_text}")

                # ax.text(ifu_center.x, ifu_center.y+ifu_pixel_height/2-7, f"{ifu_width_sky:.2f}'' x {ifu_height_sky:.2f}''", ha='center', fontsize=fontsize/1.5, fontweight='bold', color='yellow')

    # Plot Background Noise Distribution
    if plot_bkg and bkg_values is not None:
        col_num += 1
        if col_num+1 > max_cols:
            col_num = 0
            row_num += 1

        ax = axs[row_num, col_num] if nrows > 1 else axs[col_num]
        ax.hist(bkg_values) # pylint: disable=used-before-assignment

def plot_dithers(galaxy, options, space_cutout, ifu,
                 p=None,
                 q=None,
                 sp=None,
                 sq=None,
                 hide_annotations=True,
                 center_title=True,
                 **kwargs
    ):

    if (p is not None and q is not None) == (sp is not None and sq is not None):
        raise TargetsException('Either p/q or sp/sq must be specified')

    if p is not None and q is not None:
        p = np.asarray(p)
        q = np.asarray(q)

        if np.size(p) != np.size(q):
            raise TargetsException('p and q must have the same length')
    else:
        sp = np.asarray(sp)
        sq = np.asarray(sq)

        if np.size(sp) != np.size(sq):
            raise TargetsException('sp and sq must have the same length')

        p = sp * options['ifu_spaxel_width']
        q = sq * options['ifu_spaxel_height']

    dither_ifus = []
    for i in range(len(p)): # pylint: disable=consider-using-enumerate
        dither_ifu = get_ifu_position(galaxy, options, relative_ifu=ifu, p=p[i], q=q[i], silent=True)
        dither_ifu.name = f"p={p[i]:.3f}\", q={q[i]:.3f}\""
        dither_ifus.append(dither_ifu)

    plot_galaxy_cutouts(galaxy, options, space_cutout,
                        ifus=dither_ifus,
                        hide_annotations=hide_annotations,
                        center_title=center_title,
                        silent=True,
                        **kwargs)

def plot_finder_chart(field_cutout,
                      options,
                      field_radius=None, #arcsec
                      ifu=None,
                      stars=None,
                      ngs_id=None,
                      star_radius=0.5, # arcsec
                      asinh_a=0.1,
                      vmin=None,
                      vmax=None,
                      annotation_color='red'
    ):

    plot_ifu = ifu is not None and (not isinstance(ifu, bool) or ifu)

    if field_radius is None:
        if table.has_field(field_cutout, 'field_radius'):
            field_radius = field_cutout.field_radius
        elif 'field_diameter' in options:
            field_radius = options['field_diameter'] / 2
        else:
            raise TargetsException('field_radius is required')

    if isinstance(ifu, StructType):
        position_px = field_cutout.wcs.world_to_pixel(ifu.position)
    else:
        position_px = field_cutout.wcs.world_to_pixel(field_cutout.position)

    field_radius_px = field_radius/field_cutout.pixel_scale[1]
    ifu_width_px = options['ifu_width'] / field_cutout.pixel_scale[0]
    ifu_height_px = options['ifu_height'] / field_cutout.pixel_scale[1]

    cutout_norm = ImageNormalize(stretch=AsinhStretch(a=asinh_a))
    if vmin is not None:
        cutout_norm.vmin = vmin
    if vmax is not None:
        cutout_norm.vmax = vmax

    plt.figure(figsize=(8,8))
    ax = plt.subplot(projection=field_cutout.wcs)
    plt.imshow(field_cutout.data, origin='lower', cmap='gray_r', norm=cutout_norm)
    ax.tick_params(direction='in')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Plot Field Circle
    ax.add_patch(plt.Circle(position_px, field_radius_px, edgecolor=annotation_color, facecolor='none', linewidth=1.0))

    # Plot IFU
    if plot_ifu:
        if isinstance(ifu, StructType):
            pa = ifu.pa
        else:
            pa = 0.0

        ax.add_patch(plt.Rectangle(
            (position_px[0]-ifu_width_px/2, position_px[1]-ifu_height_px/2),
            ifu_width_px,
            ifu_height_px,
            angle=pa,
            rotation_point='center',
            edgecolor=annotation_color,
            facecolor='none',
            linewidth=1.0
        ))

    # Plot Stars
    if stars is not None and len(stars) > 0:
        (image_ra, image_dec) = gaia.apply_proper_motion(stars, epoch=field_cutout.epoch)
        for i in range(len(stars)):
            star_pos = SkyCoord(ra=image_ra[i], dec=image_dec[i], unit=(u.degree, u.degree))
            star_px = field_cutout.wcs.world_to_pixel(star_pos)
            star_radius_px = star_radius/field_cutout.pixel_scale[1]
            filled = ngs_id is not None and stars['gaia_id'][i] == ngs_id
            ax.add_patch(plt.Circle(star_px, star_radius_px, edgecolor=annotation_color, facecolor=annotation_color if filled else 'none', linewidth=1.0))

def compare_star_centroid(
        field_cutout,
        stars,
        star_id,
        options,
        ifu=None,
        radius=5.0, # arcsec
        annotation_color='red',
        catalog_color='cyan'
):
    plot_ifu = ifu is not None and (not isinstance(ifu, bool) or ifu)

    star = stars[stars['gaia_id'] == star_id]
    if len(star) == 0:
        raise TargetsException(f"Star {star_id} not found")
    if len(star) > 1:
        raise TargetsException(f"Multiple stars with ID {star_id}")

    (image_ra, image_dec) = gaia.apply_proper_motion(star, epoch=field_cutout.epoch)
    star_pos = SkyCoord(ra=image_ra[0], dec=image_dec[0], unit=(u.degree, u.degree))

    centroid_cutout = Cutout2D(field_cutout.data, star_pos, 2*radius*u.arcsec, wcs=field_cutout.wcs, mode='partial', fill_value=0.0) # pylint: disable=no-member
    centroid_x, centroid_y = centroid_com(centroid_cutout.data)
    centroid_pos = centroid_cutout.wcs.pixel_to_world(centroid_x, centroid_y)
    offset = centroid_pos.separation(star_pos).arcsec

    print(f"     Gaia ID: {star_id}")
    print(f"Gaia Catalog: {star['gaia_ra'][0]:.6f}, {star['gaia_dec'][0]:.7f}")
    print(f" Image Epoch: {star_pos.ra.degree:.6f}, {star_pos.dec.degree:.7f}")
    print(f"    Centroid: {centroid_pos.ra.degree:.6f}, {centroid_pos.dec.degree:.7f}")
    print(f"      Offset: {offset:.2f} arcsec" + f"({offset/options['ifu_spaxel_width']:.1f} spx)" if ifu is not None else "")

    star_cutout = Cutout2D(field_cutout.data, star_pos, 1.25*2*radius*u.arcsec, wcs=field_cutout.wcs, mode='partial', fill_value=0.0) # pylint: disable=no-member
    star_pixel_scale = WCS.proj_plane_pixel_scales(star_cutout.wcs)
    star_px = star_cutout.wcs.world_to_pixel(star_pos)
    centroid_px = star_cutout.wcs.world_to_pixel(centroid_pos)
    distance_px = radius/star_pixel_scale[1].to_value(u.arcsec)

    cutout_norm = ImageNormalize(stretch=AsinhStretch(a=0.1))

    plt.figure(figsize=(6,6))
    ax = plt.subplot(projection=star_cutout.wcs)
    plt.imshow(star_cutout.data, origin='lower', cmap='gray', norm=cutout_norm)
    ax.tick_params(direction='in')
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.scatter(star_px[0], star_px[1], c=catalog_color, s=50, marker='x', label='Catalog')
    ax.scatter(centroid_px[0], centroid_px[1], c=annotation_color, s=50, marker='x', label='Centroid')
    ax.add_patch(plt.Circle(star_px, distance_px, edgecolor=annotation_color, facecolor='none', linewidth=1.0))

    if plot_ifu:
        if isinstance(ifu, StructType):
            pa = ifu.pa
        else:
            pa = 0.0

        ifu_width_px = options['ifu_width'] / star_pixel_scale[0].to_value(u.arcsec)
        ifu_height_px = options['ifu_height'] / star_pixel_scale[1].to_value(u.arcsec)

        ax.add_patch(plt.Rectangle(
            (star_px[0]-ifu_width_px/2, star_px[1]-ifu_height_px/2),
            ifu_width_px,
            ifu_height_px,
            angle=pa,
            rotation_point='center',
            edgecolor=catalog_color,
            facecolor='none',
            linewidth=1.0
        ))

    ax.legend()

#endregion

#region Sky

def plot_sky_transmission(galaxy, lines, options, line_names=None):
    # Legend:
    # - Dotted line is expected wavelength based on redshift
    # - Shadded light gray is uncertainty of wavelength based on redshift uncertainty
    # - Shadded dark gray is region over which transmision is calculated
    # - Horizontal lines are average transmission (green means averge above threshold while red means below)

    if line_names is not None:
        line_indexes = np.where(np.isin(lines.names, line_names))[0]
    else:
        line_indexes = np.arange(len(lines.names))

    mid_wavelength = np.mean(lines.wavelength_atm[line_indexes])
    min_wavelength_range = (np.max(lines.wavelength_atm[line_indexes]) - np.min(lines.wavelength_atm[line_indexes])) * 1.2

    sky_transmission_data = sky.load_transmission_data(options['location'], options['airmass'])

    plot_dwavelength = max(min_wavelength_range, mid_wavelength/options['resolving_power']*8)
    plot_xrange = np.round(np.array([mid_wavelength - plot_dwavelength, mid_wavelength + plot_dwavelength])/10, 0)*10
    plot_yrange = [70, 101]

    wavelength_filter = (sky_transmission_data['wavelength'] >= plot_xrange[0]) & (sky_transmission_data['wavelength'] <= plot_xrange[1])
    sky_wavelengths = sky_transmission_data['wavelength'][wavelength_filter]

    _, ax = plot.create_plot(title=f"{options['location']} Sky Transmission")
    ax.plot(sky_wavelengths, sky_transmission_data['transmission'][wavelength_filter]*100, linestyle='-', color='b', linewidth=1)
    ax.set_xlabel('Wavelength [Angstrom]')
    ax.set_ylabel('Transmission [%]')
    ax.set_ylim(plot_yrange)

    for i in line_indexes:
        wavelength_unc = lines.wavelength_atm[i] * galaxy['z_unc']
        color = 'r' if lines.transmission[i] < options['trans_minimum'] else 'g'
        plt.axvline(x=lines.wavelength_atm[i]-wavelength_unc, linestyle='-', linewidth=0.5, color='k')
        ax.fill_between(sky_wavelengths, plot_yrange[0], plot_yrange[1], where=(abs(sky_wavelengths - lines.wavelength_atm[i]) < wavelength_unc), facecolor=[0.8,0.8,0.8], alpha=0.3)
        plt.axvline(x=lines.wavelength_atm[i]+wavelength_unc, linestyle='-', linewidth=0.5, color='k')
        ax.fill_between(sky_wavelengths, plot_yrange[0], plot_yrange[1], where=(abs(sky_wavelengths - lines.wavelength_atm[i]) < lines.dwavelength[i]*options['trans_multiple']), facecolor=[0.5,0.5,0.5], alpha=0.3)
        plt.axvline(x=lines.wavelength_atm[i], linestyle=':', linewidth=1, color='k')
        plt.hlines(y=lines.transmission[i]*100, xmin=(lines.wavelength_atm[i] - lines.dwavelength[i]*options['trans_multiple']), xmax=(lines.wavelength_atm[i] + lines.dwavelength[i]*options['trans_multiple']), linestyle='-', linewidth=2, color=color)
        plt.text(0.05, 0.07, f"id={galaxy['id']}\nz={galaxy['z']:.4f}\ndz={galaxy['z_unc']:.4f}", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

        print(f"Sky Transmission at {lines.wavelength_atm[i]/10:.2f} nm ({lines.names[i]}): {lines.transmission[i]*100:.1f}%")

    plt.show()

def plot_sky_background(galaxy, lines, options, line_names=None):
    # Legend:
    # - Dotted line is expected wavelength based on redshift
    # - Shadded light gray is uncertainty of wavelength based on redshift uncertainty
    # - Shadded dark gray is sky line avoidance region
    # - Green profile means no line in avoidance region
    # - Red profile means there is a sky line in avoidance region

    if line_names is not None:
        line_indexes = np.where(np.isin(lines.names, line_names))[0]
    else:
        line_indexes = np.arange(len(lines.names))

    mid_wavelength = np.mean(lines.wavelength_atm[line_indexes])
    min_wavelength_range = (np.max(lines.wavelength_atm[line_indexes]) - np.min(lines.wavelength_atm[line_indexes])) * 1.2

    sky_background_data = sky.load_background_data(options['location'], options['airmass'])

    plot_dwavelength = max(min_wavelength_range, mid_wavelength/options['resolving_power']*8)
    plot_xrange = np.round(np.array([mid_wavelength - plot_dwavelength, mid_wavelength + plot_dwavelength])/10, 0)*10
    plot_yrange = [0.1, 1e4]

    sky_wavelengths = np.linspace(plot_xrange[0], plot_xrange[1], 1000)
    sky_background_data_low_res = sky.get_low_res_background(sky_background_data, plot_xrange, options['resolving_power'])
    sky_lines = sky.find_sky_lines(sky_background_data_low_res, options['min_photon_rate'])

    _, ax = plot.create_plot(title=f"{options['location']} Sky Background")
    ax.plot(sky_background_data_low_res['wavelength'], sky_background_data_low_res['emission'], linestyle='-', color='b', linewidth=1)
    ax.scatter(sky_lines['wavelength'], sky_lines['emission'], marker='x', color='b')
    plt.hlines(y=sky_lines["width_height"], xmin=sky_lines["wavelength_low"], xmax=sky_lines["wavelength_high"], linestyle='-', color='b')
    ax.axhline(options['min_photon_rate'], linestyle=':', linewidth=1, color='k')
    ax.set_xlim(plot_xrange)
    ax.set_ylim(plot_yrange)
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength [Angstrom]')
    ax.set_ylabel('Emission [$ph/s/arcsec^2/nm/m^2$]')

    for i in line_indexes:
        wavelength_unc = lines.wavelength_atm[i] * galaxy['z_unc']
        color = 'r' if lines.reject[i] else 'g'
        plt.axvline(x=lines.wavelength_atm[i]-wavelength_unc, linestyle='-', linewidth=0.5, color='k')
        ax.fill_between(sky_wavelengths, plot_yrange[0], plot_yrange[1], where=(abs(sky_wavelengths - lines.wavelength_atm[i]) < wavelength_unc), facecolor=[0.8,0.8,0.8], alpha=0.3)
        plt.axvline(x=lines.wavelength_atm[i]+wavelength_unc, linestyle='-', linewidth=0.5, color='k')
        ax.fill_between(sky_background_data_low_res['wavelength'], plot_yrange[0], plot_yrange[1], where=(abs(sky_background_data_low_res['wavelength'] - lines.wavelength_atm[i]) < lines.dwavelength[i]*options['avoid_multiple']), facecolor=[0.5,0.5,0.5], alpha=0.3)
        ax.plot(sky_wavelengths, lines.ph_rate[i] * np.sqrt(2*np.pi) * lines.sigma[i] * norm.pdf(sky_wavelengths, lines.wavelength_atm[i], lines.sigma[i]), linestyle='-', linewidth=2, color=color)
        plt.axvline(x=lines.wavelength_atm[i], linestyle=':', linewidth=1, color='k')
        plt.text(0.05, 0.82, f"id={galaxy['id']}\nz={galaxy['z']:.4f}\ndz={galaxy['z_unc']:.4f}", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

        index = np.argmin(np.abs(sky_background_data_low_res['wavelength'] - lines.wavelength_atm[i]))
        print(f"Sky Background at {sky_background_data_low_res['wavelength'][index]/10:.2f} nm ({lines.names[i]}) = {sky_background_data_low_res['emission'][index]:.1f} ph/s/arcsec^2/nm/m^2")

    plt.show()

#endregion
