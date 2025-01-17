#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from astropy.constants import si as constants
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from survey_tools import catalog, gaia, healpix, sky
from survey_tools.utility import fit, plot, table

class StructType:
    pass

class TargetsException(Exception):
    pass

SEARCH_DISTANCE_FACTOR = 4.0
GAIA_MAPS_LEVEL = None
FLUX_NII_HA_RATIO = 0.1

def _query_stars(ra, dec, min_mag=-2.0, max_mag=25.0, min_sep=0.0, max_sep=60.0, wfs_band='R', epoch=None, skip_cache=False, verbose=False):
    search_distance = max_sep/3600*SEARCH_DISTANCE_FACTOR

    if not skip_cache:
        stars = None
        data_path = f"{os.path.dirname(__file__)}/../data"

        # Try local copy of Gaia database
        maps_gaia_path = f"{data_path}/maps/gaia"
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
            cache_filename = f"{data_path}/cache/targets/stars-{ra*1e6:.0f}_{dec*1e6:.0f}_{search_distance*1e6:.0f}.fits"
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

def _get_instrument_details(instrument):
    instrument_details = dict()
    match instrument:
        case _ if 'GNIRS' in instrument:
            instrument_details['location'] = 'MaunaKea'
            instrument_details['wavelength_ranges'] = np.array([[11700, 13700],[14900, 18000],[19100,24900]])   # Angstrom
        case _ if 'ERIS' in instrument:
            instrument_details['location'] = 'Paranal'
            instrument_details['wavelength_ranges'] = np.array([[10900, 14200],[14500, 18700],[19300,24800]])   # Angstrom
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

    return instrument_details

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

def plot_catalog_spectra(galaxy, catalog_ids, field, figsize=None, lines=None, options=None, show_emission_lines=True, show_skylines=None):
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
