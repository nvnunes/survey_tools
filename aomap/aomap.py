#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from contextlib import contextmanager
import os
import signal
import time
import traceback
import yaml
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from dustmaps.config import config as dustmaps_config
import dustmaps.gaia_tge as gaia_tge
from joblib import Parallel, delayed
from survey_tools import gaia, healpix

#region Globals

class StructType:
    pass

class AOMapException(Exception):
    pass

@contextmanager
def prevent_interruption():
    was_interrupted = False

    def handler(signum, frame): # pylint: disable=unused-argument
        nonlocal was_interrupted
        was_interrupted = True

    previous_handler = signal.signal(signal.SIGINT, handler)

    yield

    signal.signal(signal.SIGINT, previous_handler)

    if was_interrupted:
        raise KeyboardInterrupt

FITS_COLUMN_DONE = 'DONE'
FITS_COLUMN_EXCLUDED = 'EXCLUDED'
FITS_COLUMN_PIX = 'PIX'
FITS_COLUMN_MODEL_DENSITY = 'MODEL_DENSITY'
FITS_COLUMN_DUST_EXTINCTION = 'DUST_EXTINCTION'
FITS_COLUMN_STAR_COUNT = 'STAR_COUNT'
FITS_COLUMN_NGS_COUNT_PREFIX = 'NGS_COUNT'
FITS_COLUMN_NGS_PIX_PREFIX = 'NGS_PIX'

#endregion

#region Config

def read_config(config_or_filename):
    if not isinstance(config_or_filename, str):
        return config_or_filename
    else:
        filename = config_or_filename

    if not os.path.isfile(filename):
        raise AOMapException(f"Config file not found: {filename}")

    with open(filename, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)

    # convert dict to struct (personal preference)
    config = StructType()
    for key, value in config_data.items():
        setattr(config, key, value)

    if not hasattr(config, 'cores') or config.cores < -1 or config.cores == 0:
        config.cores = 1

    if not hasattr(config, 'chunk_multiple'):
        config.chunk_multiple = 0

    if not hasattr(config, 'outer_level'):
        raise AOMapException("outer_level not defined in config")

    if not hasattr(config, 'inner_level'):
        raise AOMapException("inner_level not defined in config")

    if not hasattr(config, 'max_data_level'):
        config.max_data_level = config.outer_level
        if config.max_data_level < config.outer_level:
            raise AOMapException('max_data_level must be greater than or equal to outer_level')
        if config.max_data_level > config.inner_level:
            raise AOMapException('max_data_level must be less than or equal to inner_level')

    if not hasattr(config, 'build_level'):
        config.build_level = None

    if not hasattr(config, 'build_pixs'):
        config.build_pixs = None

    if not isinstance(config.build_pixs, list):
        config.build_pixs = [config.build_pixs]

    if not hasattr(config, 'ao_systems'):
        config.ao_systems = {}

    if not hasattr(config, 'exclude_min_galactic_latitude'):
        config.exclude_min_galactic_latitude = 0.0

    return config

#endregion

#region Paths

def _is_good_FITS(filename):
    try:
        with fits.open(filename) as hdul:
            hdul.verify('exception')
    except fits.VerifyError:
        return False

    return True

def _get_outer_filename(config):
    return f"{config.folder}/outer.fits"

def _get_data_filename(config, level):
    return f"{config.folder}/data-hpx{level}.fits"

def _get_map_filename(config, key, level):
    return f"{config.folder}/{key}-hpx{level}.fits"

def _get_outer_pixel_path(config, outer_pix):
    coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix)
    return f"{config.folder}/inner/hpx{config.outer_level}-{config.inner_level}/{int(coord.ra.degree/15)}h/{'+' if coord.dec.degree >= 0 else '-'}{int(np.abs(coord.dec.degree/10))*10:02}/{outer_pix}"

def _get_gaia_path(config, outer_pix):
    coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix)
    return f"{config.folder}/gaia/hpx{config.outer_level}/{int(coord.ra.degree/15)}h/{'+' if coord.dec.degree >= 0 else '-'}{int(np.abs(coord.dec.degree/10))*10:02}/{outer_pix}"

def _get_inner_pixel_data_filename(config, outer_pix):
    return f"{_get_outer_pixel_path(config, outer_pix)}/inner.fits"

#endregion

#region Build

def build_inner(config_or_filename, mode='recalc', pixs=None, force_reload_gaia=False, max_pixels=None, verbose=False):
    config = read_config(config_or_filename)

    if config.build_level is not None and config.build_level >= config.outer_level:
        raise AOMapException('build_level must be less than outer_level')

    if config.build_level is not None and config.build_pixs is None:
        raise AOMapException('build_pixs required if build_level is provided')

    if config.build_level is None and config.build_pixs is not None:
        raise AOMapException('build_level required if build_pixs is provided')

    if config.build_pixs is None:
        build_pixs = None
    else:
        if config.build_level == config.outer_level:
            build_pixs = config.build_pixs
        else:
            build_pixs = healpix.get_subpixels(config.build_level, config.build_pixs, config.outer_level)

    if build_pixs is not None and pixs is not None and not np.all(np.isin(pixs, build_pixs)):
        raise AOMapException('pixs must be included in build_pixs')

    force_create = mode.startswith('re') and pixs is None
    _create_outer(config, force_create=force_create)
    outer = _load_outer(config)
    done = _get_outer_done(outer)
    excluded = _get_outer_excluded(outer)

    npix = healpix.get_npix(config.outer_level)

    if config.chunk_multiple == 0:
        chunk_size = npix
    else:
        chunk_size = (config.cores if config.cores >= 1 else os.cpu_count()) * config.chunk_multiple

    todo = np.zeros((npix), dtype=bool)
    if pixs is not None:
        todo[pixs] = True
    elif build_pixs is not None:
        todo[build_pixs] = ~done[build_pixs] & ~excluded[build_pixs]
    else:
        todo = ~done & ~excluded

    if np.all(todo):
        todo_pix = np.arange(npix)
    else:
        todo_pix = np.where(todo)[0]

    if len(todo_pix) == 0:
        if verbose:
            print(f"Building inner pixels at level {config.inner_level} already done")
        return

    if build_pixs is not None and verbose:
        print(f"Building total of {len(todo_pix)}/{len(build_pixs)} outer pixs")

    if max_pixels is not None and max_pixels < len(todo_pix):
        todo_pix = todo_pix[:max_pixels]

    num_todo = len(todo_pix)
    chunk_size = min(chunk_size, num_todo)
    num_chunks = int(np.ceil(num_todo / chunk_size))

    if verbose:
        print(f"Building inner pixels at level {config.inner_level}:")
        if todo_pix[0] > 0:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"\r  {current_time}: {todo_pix[0]+1}/{npix} (start)", end='', flush=True)

        start_time = time.time()
        last_time = start_time

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i+1)*chunk_size, num_todo)

        if config.cores == 1:
            num_excluded = 0
            for outer_pix in todo_pix[start_idx:end_idx]:
                results = _build_inner_data(config, mode, outer_pix, force_reload_gaia)
                done[outer_pix] = results[0]
                excluded[outer_pix] = results[1]
                num_excluded += results[1]
        else:
            results = np.array(Parallel(n_jobs=config.cores)(delayed(_build_inner_data)(config, mode, outer_pix, force_reload_gaia) for outer_pix in todo_pix[start_idx:end_idx]))
            done[todo_pix[start_idx:end_idx]]= results[:,0]
            excluded[todo_pix[start_idx:end_idx]] = results[:,1]
            num_excluded = np.sum(results[:,1])

        outer.flush()

        if verbose:
            elapsed_time = time.time() - last_time
            last_time = time.time()
            current_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"\r  {current_time}: {todo_pix[end_idx-1]+1}/{npix} ({end_idx-start_idx}px in {elapsed_time:.2f}s, {num_excluded} excluded)           ", end='', flush=True)

    if verbose:
        total_time = time.time() - start_time
        print(f"\n  done: {num_todo}px in {total_time:.1f}s")

    outer.close()

def build_data(config_or_filename, mode='build', verbose = False):
    config = read_config(config_or_filename)

    dustmaps_config.reset()
    dustmaps_config['data_dir'] = '../data/dust'
    os.makedirs(dustmaps_config['data_dir'], exist_ok=True)
    gaia_tge.fetch()

    allow_missing_inner_data = config.build_level is not None and config.build_pixs is not None

    force_create = mode.startswith('re')

    levels = []
    level_data = []
    for level in range(config.outer_level, config.max_data_level+1):
        data_filename = _get_data_filename(config, level)
        if force_create or not os.path.isfile(data_filename) or not _is_good_FITS(data_filename):
            _create_data(config, level)
            levels.append(level)
            level_data.append(_load_data(config, level, update=True))

    if len(levels) == 0:
        if verbose:
            print(f"Building data for levels {config.outer_level}-{config.max_data_level} already done")
        return

    npix = healpix.get_npix(config.outer_level)

    if config.chunk_multiple == 0:
        chunk_size = npix
    else:
        chunk_size = config.chunk_multiple

    num_chunks = int(np.ceil(npix / chunk_size))

    if verbose:
        if len(levels) == 1:
            print(f"Building data for level {levels}:")
        else:
            print(f"Building data for levels {levels[0]}-{levels[-1]}:")

    dust = gaia_tge.GaiaTGEQuery(healpix_level= 'optimum')

    start_time = time.time()
    last_time = start_time

    for i in range(num_chunks):
        outer_start_pix = i * chunk_size
        outer_end_pix = min((i+1)*chunk_size, npix)

        # NOTE: parallelizing the following isn't effective as it is disk IO limited
        for outer_pix in range(outer_start_pix, outer_end_pix):
            _set_data_pixel_values(config, levels, level_data, outer_pix, dust, allow_missing_inner_data=allow_missing_inner_data)

        for data in level_data:
            data.flush()

        if verbose:
            elapsed_time = time.time() - last_time
            last_time = time.time()
            current_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"\r  {current_time}: {outer_end_pix}/{npix} ({outer_end_pix-outer_start_pix}px in {elapsed_time:.2f}s)          ", end='', flush=True)

    if verbose:
        total_time = time.time() - start_time
        print(f"\n  done: {npix}px in {total_time:.1f}s")

    for data in level_data:
        data.close()

#endregion

#region Data

def _create_data(config, level):
    npix = healpix.get_npix(level)

    cols = []
    cols.append(fits.Column(name=FITS_COLUMN_PIX, format='K', array=np.arange(npix)))
    cols.append(fits.Column(name=FITS_COLUMN_MODEL_DENSITY, format='D', array=np.zeros((npix)), unit='density'))
    cols.append(fits.Column(name=FITS_COLUMN_DUST_EXTINCTION, format='D', array=np.zeros((npix)), unit='mag'))
    cols.append(fits.Column(name=FITS_COLUMN_STAR_COUNT, format='K', array=np.zeros((npix), dtype=np.int_), unit='stars'))
    for ao_system in config.ao_systems:
        cols.append(fits.Column(name=_get_ngs_count_field(ao_system), format='K', array=np.zeros((npix), dtype=np.int_), unit='NGS'))
        cols.append(fits.Column(name=_get_ngs_pix_field(ao_system), format='K', array=np.zeros((npix), dtype=np.int_), unit='NGS Pix'))

    hdu = fits.BinTableHDU.from_columns(cols)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])

    data_filename = _get_data_filename(config, level)
    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
    with prevent_interruption():
        hdul.writeto(data_filename, overwrite=True)

    return hdul

def _load_data(config, level, update=False):
    filename = _get_data_filename(config, level)
    if not os.path.isfile(filename):
        raise AOMapException(f"Data file not found: {filename}")
    return fits.open(filename, mode='update' if update else 'readonly')

def _set_data_pixel_values(config, levels, level_data, outer_pix, dust, allow_missing_inner_data=False):
    inner_filename = _get_inner_pixel_data_filename(config, outer_pix)
    if not os.path.isfile(inner_filename) or not _is_good_FITS(inner_filename):
        if allow_missing_inner_data:
            return
        else:
            raise AOMapException(f"Inner data not found for outer pixel {outer_pix}")

    dust_level = config.max_data_level
    _ , coords = healpix.get_subpixels_skycoord(config.outer_level, outer_pix, dust_level)
    dust_extinction = dust.query(coords)
    if config.inner_level > dust_level:
        dust_extinction = np.repeat(dust_extinction, 4**(config.inner_level - dust_level))

    inner_data = _load_inner(config, outer_pix)
    aggregate_data = _get_initial_aggregate_pixel_values(config, inner_data, dust_extinction)
    inner_data.close()

    level_done = np.zeros((len(levels)), dtype=bool)
    for level in range(config.inner_level, 0, -1):
        if level in levels:
            level_index = levels.index(level)

            level_data[level_index][1].data[FITS_COLUMN_MODEL_DENSITY][aggregate_data.pix] = aggregate_data.mean_model_density
            level_data[level_index][1].data[FITS_COLUMN_DUST_EXTINCTION][aggregate_data.pix] = aggregate_data.mean_dust_extinction
            level_data[level_index][1].data[FITS_COLUMN_STAR_COUNT][aggregate_data.pix] = aggregate_data.sum_star_count
            for i, ao_system in enumerate(config.ao_systems):
                level_data[level_index][1].data[_get_ngs_count_field(ao_system)][aggregate_data.pix] = aggregate_data.sum_ngs_count[:,i]
                level_data[level_index][1].data[_get_ngs_pix_field(ao_system)][aggregate_data.pix] = aggregate_data.sum_ngs_pix[:,i]

            level_done[level_index] = True

            if np.all(level_done):
                break

        _aggregate_pixel_values(config, aggregate_data)

def _get_initial_aggregate_pixel_values(config, inner_data, dust_extinction):
    aggregate_data = StructType()
    aggregate_data.pix = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_PIX)
    aggregate_data.mean_model_density = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_MODEL_DENSITY)
    aggregate_data.mean_dust_extinction = dust_extinction
    aggregate_data.sum_star_count = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_STAR_COUNT)
    if len(config.ao_systems) > 0:
        aggregate_data.sum_ngs_count = np.zeros((len(aggregate_data.pix), len(config.ao_systems)), dtype=np.int_)
        for i, ao_system in enumerate(config.ao_systems):
            aggregate_data.sum_ngs_count[:,i] = _get_inner_pixel_data_column(inner_data, _get_ngs_count_field(ao_system))
        aggregate_data.sum_ngs_pix = np.minimum(aggregate_data.sum_ngs_count, 1)
    return aggregate_data

def _aggregate_pixel_values(config, aggregate_data):
    aggregate_data.pix = _decrease_pix_level(aggregate_data.pix)
    aggregate_data.mean_model_density = _decrease_values_level(aggregate_data.mean_model_density, 'mean')
    aggregate_data.mean_dust_extinction = _decrease_values_level(aggregate_data.mean_dust_extinction, 'mean')
    aggregate_data.sum_star_count = _decrease_values_level(aggregate_data.sum_star_count, 'sum')
    if len(config.ao_systems) > 0:
        aggregate_data.sum_ngs_count = _decrease_values_level(aggregate_data.sum_ngs_count, 'sum')
        aggregate_data.sum_ngs_pix = _decrease_values_level(aggregate_data.sum_ngs_pix, 'sum')

def _decrease_pix_level(pixs, num_levels=1):
    for _ in range(num_levels):
        pixs = pixs[::4] // 4
    return pixs

def _decrease_values_level(values, method, num_levels=1):
    for _ in range(num_levels):
        if np.ndim(values) == 2:
            reshaped_values = values.reshape(-1, 4, values.shape[1])
        elif np.ndim(values) == 1:
            reshaped_values = values.reshape(-1, 4)
        else:
            raise AOMapException('Unsupported array dimensions')

        match method:
            case 'mean':
                values = reshaped_values.mean(axis=1)
            case 'sum':
                values = reshaped_values.sum(axis=1)
            case _:
                raise AOMapException(f"Unknown method: {method}")

    return values

#endregion

#region Outer

def _create_outer(config, force_create=False):
    npix = healpix.get_npix(config.outer_level)
    filename = _get_outer_filename(config)

    use_existing = not force_create and os.path.isfile(filename) and _is_good_FITS(filename)
    if use_existing:
        return fits.open(filename, mode='update')

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    cols = []
    cols.append(fits.Column(name=FITS_COLUMN_DONE, format='L', array=np.zeros((npix), dtype=np.bool), unit=None))
    cols.append(fits.Column(name=FITS_COLUMN_EXCLUDED, format='L', array=np.zeros((npix), dtype=np.bool), unit=None))
    pixel_data_hdu = fits.BinTableHDU.from_columns(cols)

    hdul = fits.HDUList([fits.PrimaryHDU(), pixel_data_hdu])
    hdul.writeto(filename, overwrite=True)

    return hdul

def _load_outer(config):
    filename = _get_outer_filename(config)
    if not os.path.isfile(filename):
        raise AOMapException('Outer data not found')
    return fits.open(filename, mode='update')

def _get_outer_done(outer):
    return outer[1].data[FITS_COLUMN_DONE]

def _get_outer_excluded(outer):
    return outer[1].data[FITS_COLUMN_EXCLUDED]

#endregion

#region Inner

def _build_inner_data(config, mode, outer_pix, force_reload_gaia):
    galactic_coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix).galactic
    excluded = False

    # Exclusions that DO NOT require inner_data go here:
    if config.exclude_min_galactic_latitude > 0:
        excluded = np.abs(galactic_coord.b.degree) < config.exclude_min_galactic_latitude

    if not excluded:
        skip = False

        match mode:
            case 'build':
                inner_filename = _get_inner_pixel_data_filename(config, outer_pix)
                use_existing = os.path.isfile(inner_filename) and _is_good_FITS(inner_filename)
            case 'rebuild':
                use_existing = False
            case 'recalc':
                use_existing = True

        if use_existing:
            inner_data = _load_inner(config, outer_pix)
        else:
            try:
                inner_data = _create_inner(config, outer_pix, num_retries=3, force_reload_gaia=force_reload_gaia)
            except (ConnectionResetError, FileNotFoundError) as e:
                skip = True
                print(f"Error building inner data for {outer_pix}:\n{e}")
                traceback.print_exc()

        if skip:
            # Leave this outer pixel to do in the future
            return np.array([False, False])

        if inner_data is not None:
            # Exclusions that DO require inner_data go here: excluded = ...
            inner_data.close()

    return np.array([True, excluded])

def _get_inner_pixel_data_column(inner_data, column_name):
    if column_name not in inner_data[1].columns.names:
        column_name = _get_field_from_key(column_name)
        if column_name not in inner_data[1].columns.names:
            raise AOMapException(f"Column {column_name} not found in inner data")
    return inner_data[1].data[column_name]

def _get_inner_pixel_data_column_format(inner_data, column_name):
    if column_name not in inner_data[1].columns.names:
        column_name = _get_field_from_key(column_name)
        if column_name not in inner_data[1].columns.names:
            raise AOMapException(f"Column {column_name} not found in inner data")
    return inner_data[1].columns[column_name].format

def _get_inner_pixel_data_column_unit(inner_data, column_name):
    if column_name not in inner_data[1].columns.names:
        column_name = _get_field_from_key(column_name)
        if column_name not in inner_data[1].columns.names:
            raise AOMapException(f"Column {column_name} not found in inner data")
    return inner_data[1].columns[column_name].unit

def _get_ngs_count_field(ao_system):
    return f"{FITS_COLUMN_NGS_COUNT_PREFIX}_{_get_field_from_key(ao_system['name'])}"

def _get_ngs_pix_field(ao_system):
    return f"{FITS_COLUMN_NGS_PIX_PREFIX}_{_get_field_from_key(ao_system['name'])}"

def _create_inner(config, outer_pix, num_retries=3, force_reload_gaia=False):
    # Compute Galaxy Density Model
    pixs, coords = healpix.get_subpixels_skycoord(config.outer_level, outer_pix, config.inner_level)
    galaxy_model = _compute_galaxy_model(coords.galactic)

    # Get Stars from Gaia
    gaia_stars = _get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=num_retries, force_reload=force_reload_gaia)

    # Determine Inner Pixel of Gaia Stars
    gaia_pixs = healpix.get_healpix_from_skycoord(config.inner_level, SkyCoord(ra=gaia_stars['gaia_ra'], dec=gaia_stars['gaia_dec'], unit=(u.degree, u.degree)))

    # Count Gaia Stars per Inner Pixel
    unique, counts = np.unique(gaia_pixs, return_counts=True)
    count_map = dict(zip(unique, counts))
    star_count = np.array([count_map.get(p, 0) for p in pixs])

    # Build FITS table
    cols = []
    cols.append(fits.Column(name=FITS_COLUMN_PIX, format='K', array=pixs))
    cols.append(fits.Column(name=FITS_COLUMN_MODEL_DENSITY, format='D', array=galaxy_model, unit='density'))
    cols.append(fits.Column(name=FITS_COLUMN_STAR_COUNT, format='K', array=star_count, unit='stars'))
    for ao_system in config.ao_systems:
        ngs_filter = (gaia_stars[f"gaia_{ao_system['band']}"] >= ao_system['mag_min']) & (gaia_stars[f"gaia_{ao_system['band']}"] < ao_system['mag_max'])
        unique, counts = np.unique(gaia_pixs[ngs_filter], return_counts=True)
        count_map = dict(zip(unique, counts))
        ngs_count = np.array([count_map.get(p, 0) for p in pixs])
        cols.append(fits.Column(name=_get_ngs_count_field(ao_system), format='K', array=ngs_count, unit='NGS'))

    hdu = fits.BinTableHDU.from_columns(cols)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])

    inner_filename = _get_inner_pixel_data_filename(config, outer_pix)
    os.makedirs(os.path.dirname(inner_filename), exist_ok=True)
    with prevent_interruption():
        hdul.writeto(inner_filename, overwrite=True)

    return hdul

def _load_inner(config, outer_pix):
    filename = _get_inner_pixel_data_filename(config, outer_pix)
    if not os.path.isfile(filename):
        raise AOMapException(f"Inner data not found for outer pixel {outer_pix}")
    return fits.open(filename, mode='readonly')

def _compute_galaxy_model(coords):
    rho0 = 1/25.5 # so range is from 0 to 1
    R0 = 8.2 # kpc
    hz = 0.3 # kpc
    hR = 2.5 # kpc

    l = coords.l.radian
    b = coords.b.radian
    d = R0
    z = d * np.sin(b)
    R = np.sqrt(R0**2 + (d*np.cos(b))**2 - 2*R0*d*np.cos(b)*np.cos(l))

    values = rho0 * np.exp(-np.abs(z)/hz) * np.exp(-(R-R0)/hR) + 0.0001 * np.random.normal(0, 1)
    values = np.abs(values) # don't allow negatives

    return values

def _get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=1, force_reload=False):
    filename = f"{_get_gaia_path(config, outer_pix)}/gaia.fits"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not force_reload and os.path.isfile(filename):
        with fits.open(filename, mode='readonly') as hdul:
            gaia_stars = Table(hdul[1].data) # pylint: disable=no-member
    else:
        for attempt in range(num_retries):
            try:
                gaia_stars = gaia.get_stars_by_healpix(config.outer_level, outer_pix)
                break
            except ConnectionResetError as e:
                if attempt < num_retries - 1:
                    time.sleep(1)
                else:
                    raise e

        hdu = fits.BinTableHDU(gaia_stars)
        with prevent_interruption():
            hdu.writeto(filename, overwrite=True)

    return gaia_stars

#endregion

#region Map

def _get_field_from_key(key):
    return key.replace('-','_').upper()

def _get_field_aggregate_method(key):
    if 'count' in key.lower():
        return 'sum'
    else:
        return 'mean'

def _get_map_title(key, ao_system=None):
    match key:
        case 'model-density':
            title = 'Galaxy Model'
        case 'dust-extinction':
            title = 'Dust Extinction'
        case 'star-count':
            title = 'Star Count'
        case 'star-density':
            title = 'Stellar Density'
        case _ if key.startswith('ngs-count'):
            title = 'NGS Count'
        case _ if key.startswith('ngs-density'):
            title = 'NGS Density'
        case _ if key.startswith('ngs-pix'):
            title = 'NGS Pix'
        case _ if key.startswith('ngs-pix-density'):
            title = 'NGS Pix Density'
        case _ if key.startswith('ao-friendly'):
            title = 'AO-Friendly Areas'
        case _:
            title = f"{key} Map"

    if ao_system is not None:
        title = f"{title} ({ao_system['name']})"

    return title

def _get_map_norm(key): # pylint: disable=unused-argument
    return 'log'

def _read_FITS_single_column_values(config, hdu, level, key, ao_system):
    data = hdu.data
    cols = hdu.columns

    level_area = healpix.get_area(level).to(u.arcmin**2).value
    inner_area = healpix.get_area(config.inner_level).to(u.arcmin**2).value

    match key:
        case _ if key.startswith('ao-friendly'):
            if ao_system is not None:
                fov = ao_system['fov']
                fov_area = np.pi * (fov/2)**2
                wfs = ao_system['wfs']
                max_field = 'STAR_COUNT'
                min_field = _get_ngs_pix_field(ao_system)
                field = _get_ngs_pix_field(ao_system)
            else:
                fov_area = np.pi
                wfs = 3
                max_field = 'STAR_COUNT'
                min_field = 'STAR_COUNT'
                field = 'STAR_COUNT'

            max_count = 1/inner_area * level_area
            min_count = wfs/fov_area * level_area

            has_values = (data[max_field] < max_count) & (data[min_field] > min_count)
            values = np.full((len(data)), np.nan)
            values[has_values] = data[field][has_values] / level_area
            is_density = True
        case _:
            field = _get_field_from_key(key)
            if 'DENSITY' in field and key != 'model-density':
                field = field.replace('DENSITY', 'COUNT')
                factor = 1/level_area
                is_density = True
            else:
                if cols[field].format == 'K':
                    factor = 1
                else:
                    factor = 1.0
                is_density = False

            values = factor * data[field]

    FITS_format = cols[field].format # pylint: disable=no-member
    unit = cols[field].unit # pylint: disable=no-member
    if is_density:
        FITS_format = 'D'
        unit = f"{unit}/arcmin^2"

    return (values, FITS_format, unit)

def _read_FITS_column_values(config, hdu, level, keys, ao_system, return_details=False):
    if not isinstance(keys, list):
        keys = [keys]

    values = []
    FITS_formats = []
    units = []

    for key in keys:
        (tmp_values, tmp_FITS_format, tmp_unit) = _read_FITS_single_column_values(config, hdu, level, key, ao_system)
        values.append(tmp_values)
        FITS_formats.append(tmp_FITS_format)
        units.append(tmp_unit)

    if len(values)==1:
        if return_details:
            return (values[0], FITS_formats[0], units[0])
        return values[0]
    else:
        if return_details:
            return (values, FITS_formats, units)
        return values

def _get_inner_values(config, outer_pix, keys, ao_system, return_details=False):
    inner_data = _load_inner(config, outer_pix)
    retval = _read_FITS_column_values(config, inner_data[1], config.inner_level, keys, ao_system, return_details=return_details)
    inner_data.close()
    return retval

def _get_data_values(config, level, keys, ao_system, return_details=False):
    hdul = _load_data(config, level)
    retval = _read_FITS_column_values(config, hdul[1], level, keys, ao_system, return_details=return_details)
    hdul.close()
    return retval

def get_map_data(config, map_level, key, level=None, pixs=None, coords=(), allow_slow=False):
    if pixs is not None and level is None:
        raise AOMapException('level required if pixs is provided')

    if level is not None and pixs is None:
        raise AOMapException('pixs required if level is provided')

    if level is not None and level >= map_level:
        raise AOMapException('level must be less than map_level')

    if pixs is not None and not (isinstance(pixs, list) or isinstance(pixs, np.ndarray)):
        pixs = [pixs]

    ao_key = next((prefix for prefix in ['ngs-count', 'ngs-pix', 'ngs-density', 'ao-friendly'] if f"{prefix}-" in key), None)
    if ao_key is not None:
        ao_system_name = key.replace(f"{ao_key}-",'')
        ao_system = next((system for system in config.ao_systems if system['name'] == ao_system_name), None)
    else:
        ao_system = None

    if map_level > config.max_data_level:
        if pixs is None:
            raise AOMapException('pixs required as full sky maps at this level are not supported')

        if map_level > config.inner_level:
            raise AOMapException(f"map_level must be less than oe equal to {config.inner_level}")

        if level >= config.inner_level:
            raise AOMapException(f"level must be less than {config.inner_level}")

        if level < config.outer_level:
            outer_pixs = healpix.get_subpixels(level, pixs, config.outer_level)
            level = config.outer_level
        elif level > config.outer_level:
            outer_pixs = []
            for pix in pixs:
                tmp_outer_pixs = healpix.get_parent_pixel(level, pix, config.outer_level)
                if tmp_outer_pixs not in outer_pixs:
                    outer_pixs.append(tmp_outer_pixs)
        else:
            outer_pixs = pixs

        max_outer_pix = 100
        if len(outer_pixs) > max_outer_pix and not allow_slow:
            raise AOMapException('map will take a long time to build, set allow_slow=True to continue or use lower map level')

        ([map_pixs, map_values], [_, FITS_format], [_, unit]) = _get_inner_values(config, outer_pixs[0], ['pix', key], ao_system, return_details=True)

        if len(pixs) > 1:
            for outer_pix in outer_pixs[1:]:
                [tmp_map_pixs, tmp_map_values] = _get_inner_values(config, outer_pix, ['pix', key], ao_system)
                map_pixs = np.concatenate((map_pixs, tmp_map_pixs))
                map_values = np.concatenate((map_values, tmp_map_values))

        if level > config.outer_level:
            inner_filter = healpix.get_subpixel_indexes(level, pixs, config.inner_level, config.outer_level)
            map_pixs = map_pixs[inner_filter]
            map_values = map_values[inner_filter]

        if map_level < config.inner_level:
            map_pixs = _decrease_pix_level(map_pixs, config.inner_level - map_level)
            map_values = _decrease_values_level(map_values, _get_field_aggregate_method(key), config.inner_level - map_level)
    else:
        ([map_pixs, map_values], [_, FITS_format], [_, unit]) = _get_data_values(config, max(map_level, config.outer_level), ['pix', key], ao_system, return_details=True)

        if map_level < config.outer_level:
            map_pixs = _decrease_pix_level(map_pixs, config.outer_level - map_level)
            map_values = _decrease_values_level(map_values, _get_field_aggregate_method(key), config.outer_level - map_level)

        if pixs is not None:
            map_pixs = healpix.get_subpixels(level, pixs, map_level)
            map_values = map_values[map_pixs]

    map_coords = healpix.get_pixel_skycoord(map_level, map_pixs)

    if len(coords) > 0:
        row_filter = healpix.filter_skycoords(map_coords, coords)
        map_pixs = map_pixs[row_filter]
        map_values = map_values[row_filter]
        map_coords = map_coords[row_filter]

    if FITS_format == 'K':
        num_format = '%.0f'
    else:
        num_format = '%.1f'

    map_data = StructType()
    map_data.key = key
    map_data.level = map_level
    map_data.pixs = map_pixs
    map_data.values = map_values
    map_data.coords = map_coords
    map_data.FITS_format = FITS_format
    map_data.unit = unit
    map_data.num_format = num_format
    map_data.norm = _get_map_norm(key)
    map_data.title = _get_map_title(key)
    return map_data

def get_map_table(config, map_level, key, level=None, pixs=None, coords=()):
    map_data = get_map_data(config, map_level, key, level=level, pixs=pixs, coords=coords)

    return Table([
        map_data.pixs,
        map_data.values,
        map_data.coords.ra.degree,
        map_data.coords.dec.degree
    ], names=[
        'Pix',
        'Value',
        'RA',
        'Dec'
    ])

def plot_map(map_data=None,
            galactic=False,                # rotate map from celestial to galactic coordinates
            projection='astro',            # astro | cart [hours | degrees | longitude]
            zoom=None,                     # zoom in on the map
            rotation=None,                 # longitudinal rotation angle in degrees
            width=None,                    # figure width
            height=None,                   # figure height
            dpi=None,                      # figure dpi
            xsize=None,                    # x-axis pixel dimensions
            grid=True,                     # display grid lines
            cmap=None,                     # specify the colormap
            norm=None,                     # color normalization
            vmin=None,                     # minimum value for color normalization
            vmax=None,                     # maximum value for color normalization
            cbar=True,                     # display the colorbar
            cbar_ticks=None,               # colorbar ticks
            cbar_format=None,              # colorbar number format
            cbar_unit=None,                # colorbar unit
            boundaries_level=None,         # HEALpix level for boundaries
            boundaries_pixs=None,          # HEALpix pixels for boundaries
            title=None,                    # plot title
            tissot=False,                  # display Tissot's indicatrices
            milkyway=False,                # display outline of Milky Way
            milkyway_width=None            # number of degrees north/south to draw dotted line
    ):

    if map_data is None:
        values = None
        level = None
        pixs = None
        skycoords = None
        zoom = False
    else:
        level = map_data.level
        values = map_data.values
        pixs = map_data.pixs
        skycoords = map_data.coords

        if norm is None:
            norm = map_data.norm
        if cbar_unit is None:
            cbar_unit = map_data.unit
        if cbar_format is None:
            cbar_format = map_data.num_format
        if title is None:
            title = map_data.title

    if galactic:
        xlabel = 'GLON'
        ylabel = 'GLAT'
    else:
        xlabel = 'RA'
        ylabel = 'DEC'

    projection_words = projection.lower().split()

    if zoom is None:
        if pixs is not None and len(pixs) < healpix.get_npix(level):
            zoom = (max(map_data.coords.ra.degree) - min(map_data.coords.ra.degree)) < 180
            if zoom:
                rotation = (rotation if rotation is not None else 0) + np.mean(map_data.coords.ra.degree)
        else:
            zoom = False

    if 'astro' in projection_words:
        if zoom:
            projection = 'gnomonic'
        else:
            projection = 'mollweide'
    elif 'cart' in projection_words or 'cartesian' in projection_words:
        projection = 'cartesian'
    else:
        projection = projection_words[0]

    if 'hours' in projection_words:
        grid_longitude = 'hours'
    elif 'degrees' in projection_words:
        grid_longitude = 'degrees'
    elif 'longitude' in projection_words:
        grid_longitude = 'longitude'
    elif galactic:
        grid_longitude = 'degrees'
    else:
        grid_longitude = 'hours'

    if norm is not None and vmin is not None:
        values[values < vmin] = vmin

    if norm is not None and vmax is not None:
        values[values > vmax] = vmax

    if norm is not None and 'log' in norm and values is not None and np.any(values <= 0):
        values = values.astype(float, copy=True)
        values[values <= 0.0] = np.nan

    return healpix.plot(values, level=level, pixs=pixs, skycoords=skycoords, plot_properties={
        'galactic': galactic,
        'projection': projection,
        'zoom': zoom,
        'rotation': rotation,
        'xsize': xsize,
        'width': width,
        'height': height,
        'dpi': dpi,
        'cmap': cmap,
        'norm': norm,
        'vmin': vmin,
        'vmax': vmax,
        'grid': grid,
        'grid_longitude': grid_longitude,
        'cbar': cbar,
        'cbar_ticks': cbar_ticks,
        'cbar_format': cbar_format,
        'cbar_unit': cbar_unit,
        'boundaries_level': boundaries_level,
        'boundaries_pixs': boundaries_pixs,
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'tissot': tissot,
        'milkyway': milkyway,
        'milkyway_width': milkyway_width
    })

def save_map(config, map_data, filename=None, overwrite=False):
    if filename is None:
        filename = _get_map_filename(config, map_data.key, map_data.level)
    else:
        if not bool(os.path.dirname(filename)):
            filename = f"{config.folder}/{filename}"

    col = fits.Column(name='VALUE', format=map_data.FITS_format, array=map_data.values, unit=map_data.unit)

    hdu = fits.BinTableHDU.from_columns([col])
    header = hdu.header
    header['PIXTYPE'] = 'HEALPIX'
    header['ORDERING'] = 'NESTED'
    header['COORDSYS'] = 'C'
    header['EXTNAME'] = f"{map_data.key}"
    header['NSIDE'] = 2**map_data.level
    header['FIRSTPIX'] = map_data.pixs[0]
    header['LASTPIX'] = map_data.pixs[-1]
    header['INDXSCHM'] = 'IMPLICIT'
    header['OBJECT'] = 'FULLSKY'
    hdu.writeto(filename, overwrite=overwrite)

#endregion
