#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from contextlib import contextmanager
import multiprocessing as mp
import os
import shutil
import signal
import time
import traceback
from mocpy import MOC
import yaml
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u
import matplotlib.pyplot as plt
from dustmaps.config import config as dustmaps_config
import dustmaps.gaia_tge as gaia_tge
from joblib import Parallel, delayed
from survey_tools import asterism, gaia, healpix
from survey_tools._optional_ao_tools import get_training_module

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
FITS_COLUMN_ASTERISM_COUNT_PREFIX = 'ASTERISM_COUNT'
FITS_COLUMN_ASTERISM_COVERAGE_PREFIX = 'ASTERISM_COVERAGE'
FITS_COLUMN_ASTERISM_SR_MAX_PREFIX = 'ASTERISM_SR_MAX'
FITS_COLUMN_ASTERISM_EE_MAX_PREFIX = 'ASTERISM_EE_MAX'
FITS_COLUMN_ASTERISM_FWHM_MIN_PREFIX = 'ASTERISM_FWHM_MIN'
FITS_COLUMN_ID_SUFFIX = 'ID'
FITS_COLUMN_DISTANCE_SUFFIX = 'DISTANCE'
INFERENCE_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS'
)

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

    if not hasattr(config, 'asterism_epoch'):
        config.asterism_epoch = None

    if not hasattr(config, 'prediction_wavelength'):
        config.prediction_wavelength = 1.654
    config.prediction_wavelength *= u.micron

    if not hasattr(config, 'seeing_reference_wavelength'):
        config.seeing_reference_wavelength = 0.5
    config.seeing_reference_wavelength *= u.micron

    if not hasattr(config, 'seeing_reference_sr'):
        config.seeing_reference_sr = 0.0
    if not hasattr(config, 'seeing_reference_ee'):
        config.seeing_reference_ee = 0.02
    if not hasattr(config, 'seeing_reference_fwhm'):
        config.seeing_reference_fwhm = 650.0

    if not hasattr(config, 'build_level'):
        config.build_level = None

    if not hasattr(config, 'build_pixs'):
        config.build_pixs = None
    elif not isinstance(config.build_pixs, list):
        config.build_pixs = [config.build_pixs]

    if not hasattr(config, 'ao_systems'):
        config.ao_systems = {}
    else:
        for ao_system in config.ao_systems:
            if 'rank' not in ao_system:
                ao_system['rank'] = 1
            if 'name' not in ao_system:
                raise AOMapException('ao_system name required')
            if 'band' not in ao_system:
                raise AOMapException('ao_system band required')
            if 'fov' not in ao_system:
                raise AOMapException('ao_system fov required')
            if 'fov_1ngs' not in ao_system:
                ao_system['fov_1ngs'] = ao_system['fov']
            if 'lgs' not in ao_system:
                ao_system['lgs'] = [
                    {"zd": 30.0, "az": 45    },
                    {"zd": 30.0, "az": 45+90 },
                    {"zd": 30.0, "az": 45+180},
                    {"zd": 30.0, "az": 45+270},
                ]
            if 'min_wfs' not in ao_system:
                raise AOMapException('ao_system min_wfs required')
            if 'max_wfs' not in ao_system:
                raise AOMapException('ao_system max_wfs required')
            if 'min_mag' not in ao_system:
                raise AOMapException('ao_system min_mag required')
            if 'max_mag' not in ao_system:
                raise AOMapException('ao_system max_mag required')
            if 'min_sep' not in ao_system:
                raise AOMapException('ao_system min_sep required')
            if 'max_sep' not in ao_system:
                raise AOMapException('ao_system max_sep required')
            if 'min_rel_sep' not in ao_system:
                ao_system['min_rel_sep'] = 0.0
            if 'max_rel_sep' not in ao_system:
                ao_system['max_rel_sep'] = 0.0
            if 'min_rel_area' not in ao_system:
                ao_system['min_rel_area'] = 0.0
            if 'max_rel_area' not in ao_system:
                ao_system['max_rel_area'] = 0.0
            if 'models' not in ao_system:
                ao_system['models'] = {}
            if 'rot_range' not in ao_system:
                ao_system['rot_range'] = None
            if 'rot_step' not in ao_system:
                ao_system['rot_step'] = None

            ao_system['fov']      *= u.arcsec
            ao_system['fov_1ngs'] *= u.arcsec
            ao_system['min_sep']  *= u.arcsec
            ao_system['max_sep']  *= u.arcsec

    if not hasattr(config, 'max_dust_extinction'):
        config.max_dust_extinction = None

    if not hasattr(config, 'min_ecliptic_latitude'):
        config.min_ecliptic_latitude = None

    if not hasattr(config, 'asterisms_min_galactic_latitude'):
        config.asterisms_min_galactic_latitude = 20.0

    if not hasattr(config, 'asterisms_galactic_latitude_bypass_pixs'):
        config.asterisms_galactic_latitude_bypass_pixs = []
    elif not isinstance(config.asterisms_galactic_latitude_bypass_pixs, list):
        config.asterisms_galactic_latitude_bypass_pixs = [config.asterisms_galactic_latitude_bypass_pixs]
    config.asterisms_galactic_latitude_bypass_pixs = [int(pix) for pix in config.asterisms_galactic_latitude_bypass_pixs]

    if not hasattr(config, 'asterisms_max_star_density'):
        config.asterisms_max_star_density = 2.0

    if not hasattr(config, 'asterisms_max_dust_extinction'):
        config.asterisms_max_dust_extinction = None

    if not hasattr(config, 'asterisms_max_bright_star_mag'):
        config.asterisms_max_bright_star_mag = None

    if not hasattr(config, 'asterisms_max_overlap'):
        config.asterisms_max_overlap = None

    if not hasattr(config, 'coverage_ee_threshold_resolved'):
        config.coverage_ee_threshold_resolved = 0.25

    if not hasattr(config, 'coverage_ee_threshold_mean'):
        config.coverage_ee_threshold_mean = 0.25

    return config

def _get_effective_cores(config):
    if config.cores == -1:
        return os.cpu_count() or 1
    return config.cores

def _configure_inference_threads(num_threads, verbose=False, context=None):
    num_threads = max(1, int(num_threads))

    for env_var in INFERENCE_THREAD_ENV_VARS:
        os.environ[env_var] = str(num_threads)

    try:
        import torch
        torch.set_num_threads(num_threads)
        # Keep interop at 1 so process-parallel build_inner workers do not oversubscribe.
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass

    if verbose and context is not None:
        print(f"  {context}: inference threads = {num_threads}")

def get_ao_system(config, ao_system_name):
    return next((system for system in config.ao_systems if system['name'] == ao_system_name), None)

def get_prediction_wavelength(config):
    return config.prediction_wavelength

def scale_seeing_performance(prediction_wavelength, sr, ee, fwhm, reference_wavelength):
    scaled = StructType()
    wavelength_ratio = (prediction_wavelength / reference_wavelength).to(u.dimensionless_unscaled).value

    if sr is None:
        scaled.sr = None
    elif sr <= 0:
        scaled.sr = sr
    else:
        scaled.sr = np.power(sr, np.power(wavelength_ratio, -2.0))

    scaled.ee = None if ee is None else ee * np.power(wavelength_ratio, 2.0/5.0)
    scaled.fwhm = None if fwhm is None else fwhm * np.power(wavelength_ratio, -1.0/5.0)

    return scaled

def get_seeing_baseline_performance(config):
    return scale_seeing_performance(
        config.prediction_wavelength,
        config.seeing_reference_sr,
        config.seeing_reference_ee,
        config.seeing_reference_fwhm,
        config.seeing_reference_wavelength
    )

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

def _get_hour_deg_for_path(outer_pix, coord):
    hour = int(np.floor(coord.ra.degree/15))
    deg = int(np.floor(np.abs(coord.dec.degree/10))*10)
    # Hack: not clear why, but the following pixels end up in a different hour folder 
    # during the build versus in export asterisms
    if outer_pix in [8960, 8972, 9023, 9152, 9200, 9203, 9215, 11264, 11312, 11327, 11468]:
        hour = 14
    return hour, deg

def _get_outer_pixel_path(config, outer_pix):
    coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix)
    hour, deg = _get_hour_deg_for_path(outer_pix, coord)
    return f"{config.folder}/inner/hpx{config.outer_level}-{config.inner_level}/{hour}h/{'+' if coord.dec.degree >= 0 else '-'}{deg:02}/{outer_pix}"

def _get_gaia_path(config, outer_pix):
    coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix)
    hour, deg = _get_hour_deg_for_path(outer_pix, coord)
    return f"{config.folder}/gaia/hpx{config.outer_level}/{hour}h/{'+' if coord.dec.degree >= 0 else '-'}{deg:02}/{outer_pix}"

def _get_inner_pixel_data_filename(config, outer_pix):
    return f"{_get_outer_pixel_path(config, outer_pix)}/inner.fits"

def _get_inner_pixel_data_cache_filename(config, outer_pix):
    return f"{config.folder}/cache/inner/{outer_pix}.fits"

def _get_gaia_stars_cache_filename(config, outer_pix):
    return f"{config.folder}/cache/gaia/{outer_pix}.fits"

def _get_asterisms_data_filename(config, outer_pix, ao_system_name):
    return f"{_get_outer_pixel_path(config, outer_pix)}/asterisms-{ao_system_name}.fits"

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
    elif mode == 'build':
        if build_pixs is not None:
            todo[build_pixs] = ~done[build_pixs] & ~excluded[build_pixs]
        else:
            todo = ~done & ~excluded
    elif build_pixs is not None:
        todo[build_pixs] = True
    else:
        todo[:] = True

    if np.all(todo):
        todo_pix = np.arange(npix)
    else:
        todo_pix = np.where(todo)[0]

    if len(todo_pix) == 0:
        print(f"Building inner pixels at level {config.inner_level} already done")
        outer.close()
        return False

    if build_pixs is not None:
        print(f"Building total of {len(todo_pix)}/{len(build_pixs)} outer pixs")

    if max_pixels is not None and max_pixels < len(todo_pix):
        todo_pix = todo_pix[:max_pixels]

    num_todo = len(todo_pix)
    chunk_size = min(chunk_size, num_todo)
    num_chunks = int(np.ceil(num_todo / chunk_size))

    print(f"Building inner pixels at level {config.inner_level}:")
    if todo_pix[0] > 0:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"\r  {current_time}: {todo_pix[0]+1}/{npix} (start)", end='', flush=True)

    start_time = time.time()
    last_time = start_time
    did_work = False

    def warmup():
        # build_inner parallelizes over outer pixels, so keep each worker's inference
        # single-threaded to avoid multiplying backend threads by process count.
        _configure_inference_threads(1, verbose=verbose, context='build_inner worker')
        _load_models(config.ao_systems)

    if config.cores > 1:
        mp.set_start_method('spawn', force=True)
        parallel_pool = Parallel(n_jobs=config.cores, backend="loky", max_nbytes=None, timeout=None, idle_worker_timeout=None, initializer=warmup)
    else:
        warmup()

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i+1)*chunk_size, num_todo)

        if config.cores == 1:
            num_excluded = 0
            num_asterisms = 0
            for outer_pix in todo_pix[start_idx:end_idx]:
                results = _build_outer_pix(config, mode, outer_pix, force_reload_gaia, verbose)
                done[outer_pix] = results[0]
                excluded[outer_pix] = results[1]
                num_excluded += results[1]
                num_asterisms += results[2]
                did_work = did_work or results[0]
        else:
            results = np.array(parallel_pool(delayed(_build_outer_pix)(config, mode, outer_pix, force_reload_gaia, verbose) for outer_pix in todo_pix[start_idx:end_idx]))
            done[todo_pix[start_idx:end_idx]] = results[:,0]
            excluded[todo_pix[start_idx:end_idx]] = results[:,1]
            num_excluded = np.sum(results[:,1])
            num_asterisms = np.sum(results[:,2])
            did_work = did_work or np.any(results[:,0])

        outer.flush()

        elapsed_time = time.time() - last_time
        last_time = time.time()
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"\r  {current_time}: {todo_pix[end_idx-1]+1}/{npix} ({end_idx-start_idx}px in {elapsed_time:.2f}s, {num_excluded} excluded, {num_asterisms} asterisms)           ", end='', flush=True)

    total_time = time.time() - start_time
    print(f"\n  done: {num_todo}px in {total_time:.1f}s")

    outer.close()
    return did_work

def append_asterism_dust(config_or_filename, mode='build', pixs=None, verbose = False): # pylint: disable=unused-argument
    config = read_config(config_or_filename)

    dust = _get_dust()

    if pixs is not None:
        npix = len(pixs)
        chunk_size = npix
        num_chunks = 1
    else:
        npix = healpix.get_npix(config.outer_level)

        if config.chunk_multiple == 0:
            chunk_size = npix
        else:
            chunk_size = config.chunk_multiple

        num_chunks = int(np.ceil(npix / chunk_size))

    print('Appending asterism dust extinction:')

    start_time = time.time()
    last_time = start_time
    count = 0

    for i in range(num_chunks):
        if pixs is not None:
            chunk_pixs = pixs
        else:
            outer_start_pix = i * chunk_size
            outer_end_pix = min((i+1)*chunk_size, npix)
            chunk_pixs = range(outer_start_pix, outer_end_pix)

        # NOTE: parallelizing the following isn't effective as it is disk IO limited
        for outer_pix in chunk_pixs:
            count += 1
            for ao_system in config.ao_systems:
                _append_asterism_dust_extinction(config, outer_pix, ao_system, dust)

        elapsed_time = time.time() - last_time
        last_time = time.time()
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"\r  {current_time}: {count}/{npix} ({len(chunk_pixs)}px in {elapsed_time:.2f}s)          ", end='', flush=True)

    total_time = time.time() - start_time
    print(f"\n  done: {npix}px in {total_time:.1f}s")

def append_asterism_stats(config_or_filename, mode='build', pixs=None, verbose = False): # pylint: disable=unused-argument
    config = read_config(config_or_filename)
    # append_asterism_stats is single-process, so let backend inference use the
    # configured core count rather than pinning each model call to one thread.
    _configure_inference_threads(_get_effective_cores(config), verbose=verbose, context='append_asterism_stats')

    if pixs is not None:
        npix = len(pixs)
        chunk_size = npix
        num_chunks = 1
    else:
        npix = healpix.get_npix(config.outer_level)

        if config.chunk_multiple == 0:
            chunk_size = npix
        else:
            chunk_size = config.chunk_multiple

        num_chunks = int(np.ceil(npix / chunk_size))

    print('Appending asterism counts, coverage, and performance:')

    start_time = time.time()
    last_time = start_time
    count = 0
    num_skipped = 0
    did_work = False

    for i in range(num_chunks):
        if pixs is not None:
            chunk_pixs = pixs
        else:
            outer_start_pix = i * chunk_size
            outer_end_pix = min((i+1)*chunk_size, npix)
            chunk_pixs = range(outer_start_pix, outer_end_pix)

        # NOTE: parallelizing the following isn't effective as it is disk IO limited
        for outer_pix in chunk_pixs:
            count += 1
            inner_data = _load_inner(config, outer_pix, update=True)
            if mode == 'build' and _has_complete_asterism_stats(inner_data, config.ao_systems):
                num_skipped += 1
                inner_data.close()
                continue

            for ao_system in config.ao_systems:
                asterism_count = _get_inner_pixel_asterism_count(config, outer_pix, ao_system)
                count_column = _get_asterism_count_field(ao_system)
                if count_column not in inner_data[1].columns.names: # pylint: disable=no-member
                    col = fits.Column(name=count_column, format='K', array=asterism_count, unit='asterisms')
                    inner_data[1].columns.add_col(col) # pylint: disable=no-member
                else:
                    inner_data[1].data[count_column] = asterism_count # pylint: disable=no-member

                asterism_stats = _get_inner_pixel_asterism_coverage_and_performance(config, outer_pix, ao_system)
                asterism_coverage = asterism_stats.coverage
                coverage_column = _get_asterism_coverage_field(ao_system)
                if coverage_column not in inner_data[1].columns.names: # pylint: disable=no-member
                    col = fits.Column(name=coverage_column, format='D', array=asterism_coverage, unit='percent')
                    inner_data[1].columns.add_col(col) # pylint: disable=no-member
                else:
                    inner_data[1].data[coverage_column] = asterism_coverage # pylint: disable=no-member

                resolved_coverage_column = _get_asterism_coverage_resolved_field(ao_system)
                if resolved_coverage_column not in inner_data[1].columns.names: # pylint: disable=no-member
                    col = fits.Column(name=resolved_coverage_column, format='D', array=asterism_stats.resolved_coverage, unit='percent')
                    inner_data[1].columns.add_col(col) # pylint: disable=no-member
                else:
                    inner_data[1].data[resolved_coverage_column] = asterism_stats.resolved_coverage # pylint: disable=no-member

                mean_coverage_column = _get_asterism_coverage_mean_field(ao_system)
                if mean_coverage_column not in inner_data[1].columns.names: # pylint: disable=no-member
                    col = fits.Column(name=mean_coverage_column, format='D', array=asterism_stats.mean_coverage, unit='percent')
                    inner_data[1].columns.add_col(col) # pylint: disable=no-member
                else:
                    inner_data[1].data[mean_coverage_column] = asterism_stats.mean_coverage # pylint: disable=no-member

                performance_fields = [
                    (_get_asterism_sr_max_field(ao_system), 'D', asterism_stats.sr_max, None),
                    (_get_asterism_ee_max_field(ao_system), 'D', asterism_stats.ee_max, None),
                    (_get_asterism_ee_max_field_mean_field(ao_system), 'D', asterism_stats.ee_max_field_mean, None),
                    (_get_asterism_ee_max_id_field(ao_system), 'D', asterism_stats.ee_max_id, None),
                    (_get_asterism_ee_max_distance_field(ao_system), 'D', asterism_stats.ee_max_distance, 'arcsec'),
                    (_get_asterism_fwhm_min_field(ao_system), 'D', asterism_stats.fwhm_min, None)
                ]
                for column_name, fmt, values, unit in performance_fields:
                    if column_name not in inner_data[1].columns.names: # pylint: disable=no-member
                        col = fits.Column(name=column_name, format=fmt, array=values, unit=unit)
                        inner_data[1].columns.add_col(col) # pylint: disable=no-member
                    else:
                        inner_data[1].data[column_name] = values # pylint: disable=no-member

            with prevent_interruption():
                inner_data.flush()

            inner_data.close()
            did_work = True

        elapsed_time = time.time() - last_time
        last_time = time.time()
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"\r  {current_time}: {count}/{npix} ({len(chunk_pixs)}px in {elapsed_time:.2f}s, {num_skipped} skipped)          ", end='', flush=True)

    total_time = time.time() - start_time
    print(f"\n  done: {npix}px in {total_time:.1f}s")
    return did_work

def build_data(config_or_filename, mode='build', verbose = False): # pylint: disable=unused-argument
    config = read_config(config_or_filename)

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
        print(f"Building data for levels {config.outer_level}-{config.max_data_level} already done")
        return False

    dust = _get_dust()

    npix = healpix.get_npix(config.outer_level)

    if config.chunk_multiple == 0:
        chunk_size = npix
    else:
        chunk_size = config.chunk_multiple

    num_chunks = int(np.ceil(npix / chunk_size))

    if len(levels) == 1:
        print(f"Building data for level {levels}:")
    else:
        print(f"Building data for levels {levels[0]}-{levels[-1]}:")

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

        elapsed_time = time.time() - last_time
        last_time = time.time()
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"\r  {current_time}: {outer_end_pix}/{npix} ({outer_end_pix-outer_start_pix}px in {elapsed_time:.2f}s)          ", end='', flush=True)

    total_time = time.time() - start_time
    print(f"\n  done: {npix}px in {total_time:.1f}s")

    for data in level_data:
        data.close()
    return True

def _get_survey_extent_specs():
    return [
        ['ews', [
            "../data//euclid/rsd2024a-footprint-equ-13-year1-MOC.fits",
            "../data//euclid/rsd2024a-footprint-equ-13-year2-MOC.fits",
            "../data//euclid/rsd2024a-footprint-equ-13-year3-MOC.fits",
            "../data//euclid/rsd2024a-footprint-equ-13-year4-MOC.fits",
            "../data//euclid/rsd2024a-footprint-equ-13-year5-MOC.fits",
            "../data//euclid/rsd2024a-footprint-equ-13-year6-MOC.fits"
               ]
        ],
        ['ews-yr1', "../data//euclid/rsd2024a-footprint-equ-13-year1-MOC.fits"],
        ['ews-yr2', "../data//euclid/rsd2024a-footprint-equ-13-year2-MOC.fits"],
        ['ews-yr3', "../data//euclid/rsd2024a-footprint-equ-13-year3-MOC.fits"],
        ['ews-yr4', "../data//euclid/rsd2024a-footprint-equ-13-year4-MOC.fits"],
        ['ews-yr5', "../data//euclid/rsd2024a-footprint-equ-13-year5-MOC.fits"],
        ['ews-yr6', "../data//euclid/rsd2024a-footprint-equ-13-year6-MOC.fits"],
        ['edf-north', "../data//euclid/EuclidMOC_EDFN_rsd2024c_depth13_atLeast2visitsPlanned.fits"],
        ['edf-south', "../data//euclid/EuclidMOC_EDFS_rsd2024c_depth13_atLeast2visitsPlanned.fits"],
        ['edf-fornax', "../data//euclid/EuclidMOC_EDFF_rsd2024c_depth13_atLeast2visitsPlanned.fits"]
    ]

def build_survey_extent(config_or_filename, mode='build', verbose = False): # pylint: disable=unused-argument
    config = read_config(config_or_filename)

    surveys = _get_survey_extent_specs()
    levels = []
    level_data = []
    for level in range(config.outer_level, config.max_data_level+1):
        levels.append(level)
        level_data.append(_load_data(config, level, update=True))

    if mode == 'build':
        expected_columns = {_get_survey_column(survey[0]) for survey in surveys}
        complete = all(expected_columns.issubset(set(data[1].columns.names)) for data in level_data) # pylint: disable=no-member
        if complete:
            if len(level_data) == 1:
                print(f"Survey extent for level {levels} already done")
            else:
                print(f"Survey extent for levels {levels[0]}-{levels[-1]} already done")
            for data in level_data:
                data.close()
            return False

    if len(levels) == 1:
        print(f"Appending survey extent for level {levels}:")
    else:
        print(f"Appending survey extent for levels {levels[0]}-{levels[-1]}:")

    start_time = time.time()

    for survey in surveys:
        print(f"  Adding Survey: {survey[0]}")

        if isinstance(survey[1], list):
            mocs = []
            for filename in survey[1]:
                mocs.append(MOC.from_fits(filename))
            moc = mocs[0].union(*mocs[1:])
        else:
            moc = MOC.from_fits(survey[1])

        for level in levels:
            data = level_data[levels.index(level)]
            _set_survey_filter_column(level, data, survey[0], moc)
            data.flush()

    total_time = time.time() - start_time
    print(f"\n  done in {total_time:.1f}s")

    for data in level_data:
        data.close()
    return True

def _get_dust():
    dustmaps_config.reset()
    dustmaps_config['data_dir'] = '../data/dust'
    os.makedirs(dustmaps_config['data_dir'], exist_ok=True)
    gaia_tge.fetch()

    return gaia_tge.GaiaTGEQuery(healpix_level= 'optimum')

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
        cols.append(fits.Column(name=_get_asterism_count_field(ao_system), format='K', array=np.zeros((npix), dtype=np.int_), unit='asterisms'))
        cols.append(fits.Column(name=_get_asterism_coverage_field(ao_system), format='D', array=np.zeros((npix)), unit='percent'))
        cols.append(fits.Column(name=_get_asterism_coverage_resolved_field(ao_system), format='D', array=np.zeros((npix)), unit='percent'))
        cols.append(fits.Column(name=_get_asterism_coverage_mean_field(ao_system), format='D', array=np.zeros((npix)), unit='percent'))
        cols.append(fits.Column(name=_get_asterism_sr_max_field(ao_system), format='D', array=np.full((npix), np.nan), unit=None))
        cols.append(fits.Column(name=_get_asterism_ee_max_field(ao_system), format='D', array=np.full((npix), np.nan), unit=None))
        cols.append(fits.Column(name=_get_asterism_fwhm_min_field(ao_system), format='D', array=np.full((npix), np.nan), unit=None))

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
    aggregate_data = _get_initial_aggregate_pixel_values(config, outer_pix, inner_data, dust_extinction)
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
                level_data[level_index][1].data[_get_asterism_count_field(ao_system)][aggregate_data.pix] = aggregate_data.sum_asterism_count[:,i]
                level_data[level_index][1].data[_get_asterism_coverage_field(ao_system)][aggregate_data.pix] = aggregate_data.mean_asterism_coverage[:,i]
                level_data[level_index][1].data[_get_asterism_coverage_resolved_field(ao_system)][aggregate_data.pix] = aggregate_data.mean_asterism_coverage_resolved[:,i]
                level_data[level_index][1].data[_get_asterism_coverage_mean_field(ao_system)][aggregate_data.pix] = aggregate_data.mean_asterism_coverage_mean[:,i]
                level_data[level_index][1].data[_get_asterism_sr_max_field(ao_system)][aggregate_data.pix] = aggregate_data.max_asterism_sr[:,i]
                level_data[level_index][1].data[_get_asterism_ee_max_field(ao_system)][aggregate_data.pix] = aggregate_data.max_asterism_ee[:,i]
                level_data[level_index][1].data[_get_asterism_fwhm_min_field(ao_system)][aggregate_data.pix] = aggregate_data.min_asterism_fwhm[:,i]

            level_done[level_index] = True

            if np.all(level_done):
                break

        _aggregate_pixel_values(config, aggregate_data)

def _get_initial_aggregate_pixel_values(config, outer_pix, inner_data, dust_extinction): # pylint: disable=unused-argument
    aggregate_data = StructType()
    aggregate_data.pix = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_PIX)
    aggregate_data.mean_model_density = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_MODEL_DENSITY)
    aggregate_data.mean_dust_extinction = dust_extinction
    aggregate_data.sum_star_count = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_STAR_COUNT)
    if len(config.ao_systems) > 0:
        aggregate_data.sum_ngs_count = np.zeros((len(aggregate_data.pix), len(config.ao_systems)), dtype=np.int_)
        aggregate_data.sum_asterism_count = np.zeros((len(aggregate_data.pix), len(config.ao_systems)), dtype=np.int_)
        aggregate_data.mean_asterism_coverage = np.zeros((len(aggregate_data.pix), len(config.ao_systems)))
        aggregate_data.mean_asterism_coverage_resolved = np.zeros((len(aggregate_data.pix), len(config.ao_systems)))
        aggregate_data.mean_asterism_coverage_mean = np.zeros((len(aggregate_data.pix), len(config.ao_systems)))
        aggregate_data.max_asterism_sr = np.full((len(aggregate_data.pix), len(config.ao_systems)), np.nan)
        aggregate_data.max_asterism_ee = np.full((len(aggregate_data.pix), len(config.ao_systems)), np.nan)
        aggregate_data.min_asterism_fwhm = np.full((len(aggregate_data.pix), len(config.ao_systems)), np.nan)
        for i, ao_system in enumerate(config.ao_systems):
            aggregate_data.sum_ngs_count[:,i] = _get_inner_pixel_data_column(inner_data, _get_ngs_count_field(ao_system))
            aggregate_data.sum_asterism_count[:,i] = _get_inner_pixel_data_column(inner_data, _get_asterism_count_field(ao_system))
            aggregate_data.mean_asterism_coverage[:,i] = _get_inner_pixel_data_column(inner_data, _get_asterism_coverage_field(ao_system))
            aggregate_data.mean_asterism_coverage_resolved[:,i] = _get_inner_pixel_data_column(inner_data, _get_asterism_coverage_resolved_field(ao_system))
            aggregate_data.mean_asterism_coverage_mean[:,i] = _get_inner_pixel_data_column(inner_data, _get_asterism_coverage_mean_field(ao_system))
            aggregate_data.max_asterism_sr[:,i] = _get_inner_pixel_data_column(inner_data, _get_asterism_sr_max_field(ao_system))
            aggregate_data.max_asterism_ee[:,i] = _get_inner_pixel_data_column(inner_data, _get_asterism_ee_max_field(ao_system))
            aggregate_data.min_asterism_fwhm[:,i] = _get_inner_pixel_data_column(inner_data, _get_asterism_fwhm_min_field(ao_system))
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
        aggregate_data.sum_asterism_count = _decrease_values_level(aggregate_data.sum_asterism_count, 'sum')
        aggregate_data.mean_asterism_coverage = _decrease_values_level(aggregate_data.mean_asterism_coverage, 'mean')
        aggregate_data.mean_asterism_coverage_resolved = _decrease_values_level(aggregate_data.mean_asterism_coverage_resolved, 'mean')
        aggregate_data.mean_asterism_coverage_mean = _decrease_values_level(aggregate_data.mean_asterism_coverage_mean, 'mean')
        aggregate_data.max_asterism_sr = _decrease_values_level(aggregate_data.max_asterism_sr, 'mean')
        aggregate_data.max_asterism_ee = _decrease_values_level(aggregate_data.max_asterism_ee, 'mean')
        aggregate_data.min_asterism_fwhm = _decrease_values_level(aggregate_data.min_asterism_fwhm, 'mean')

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

def _get_survey_column(survey):
    return f"survey_{_get_field_from_key(survey)}".upper()

def _set_survey_filter_column(level, data, survey, moc):
    survey_column = _get_survey_column(survey)

    if survey_column not in data[1].columns.names:
        npix = healpix.get_npix(level)
        col = fits.Column(name=survey_column, format='L', array=np.zeros((npix), dtype=np.bool), unit=None)
        data[1].columns.add_col(col)

    level_pixs = moc.degrade_to_order(level).flatten()

    data[1].data[survey_column][level_pixs] = True

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

def _build_outer_pix(config, mode, outer_pix, force_reload_gaia, verbose=False):
    [success, excluded] = _build_inner_data(config, mode, outer_pix, force_reload_gaia)

    max_num_asterisms = 0
    if success and not excluded:
        for ao_system in config.ao_systems:
            [success, _, num_asterisms] = _build_asterisms(config, mode, outer_pix, ao_system['name'], verbose=verbose)
            max_num_asterisms = max(max_num_asterisms, num_asterisms)
            if not success:
                break

    if not success:
        excluded = False

    return [success, excluded, max_num_asterisms]

#endregion

#region Inner

def _build_inner_data(config, mode, outer_pix, force_reload_gaia):
    inner_data = None

    # Exclusions that DO NOT require inner_data go here: excluded = ...
    excluded = False

    if excluded:
        success = True
    else:
        match mode:
            case 'build' | 'recalc':
                inner_filename = _get_inner_pixel_data_filename(config, outer_pix)
                use_existing = os.path.isfile(inner_filename) and _is_good_FITS(inner_filename)
            case 'rebuild':
                use_existing = False

        if use_existing:
            # Uncomment if inner data is needed for exclusions below
            # inner_data = _load_inner(config, outer_pix)
            success = True
        else:
            try:
                inner_data = _create_inner(config, outer_pix, num_retries=3, force_reload_gaia=force_reload_gaia)
                success = True
            except (ConnectionResetError, FileNotFoundError) as e:
                print(f"Error building inner data for {outer_pix}:\n{e}")
                traceback.print_exc()
                # Leave this outer pixel to do in the future
                success = False
                return np.array([False, False])

        if success:
            # Exclusions that DO require inner_data go here: excluded = ...
            pass

        if inner_data is not None:
            inner_data.close()

    if not success:
        excluded = False

    return np.array([success, excluded])

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

def _get_asterism_count_field(ao_system):
    return f"{FITS_COLUMN_ASTERISM_COUNT_PREFIX}_{_get_field_from_key(ao_system['name'])}"

def _get_asterism_coverage_field(ao_system):
    return f"{FITS_COLUMN_ASTERISM_COVERAGE_PREFIX}_{_get_field_from_key(ao_system['name'])}"

def _get_asterism_coverage_resolved_field(ao_system):
    return f"{FITS_COLUMN_ASTERISM_COVERAGE_PREFIX}_{_get_field_from_key(ao_system['name'])}_RESOLVED"

def _get_asterism_coverage_mean_field(ao_system):
    return f"{FITS_COLUMN_ASTERISM_COVERAGE_PREFIX}_{_get_field_from_key(ao_system['name'])}_MEAN"

def _get_asterism_sr_max_field(ao_system):
    return f"{FITS_COLUMN_ASTERISM_SR_MAX_PREFIX}_{_get_field_from_key(ao_system['name'])}"

def _get_asterism_ee_max_field(ao_system):
    return f"{FITS_COLUMN_ASTERISM_EE_MAX_PREFIX}_{_get_field_from_key(ao_system['name'])}"

def _get_asterism_ee_max_field_mean_field(ao_system):
    return f"{_get_asterism_ee_max_field(ao_system)}_FIELD_MEAN"

def _get_asterism_ee_max_id_field(ao_system):
    return f"{_get_asterism_ee_max_field(ao_system)}_{FITS_COLUMN_ID_SUFFIX}"

def _get_asterism_ee_max_distance_field(ao_system):
    return f"{_get_asterism_ee_max_field(ao_system)}_{FITS_COLUMN_DISTANCE_SUFFIX}"

def _get_asterism_fwhm_min_field(ao_system):
    return f"{FITS_COLUMN_ASTERISM_FWHM_MIN_PREFIX}_{_get_field_from_key(ao_system['name'])}"

def _get_managed_asterism_stats_fields(ao_system):
    return [
        _get_asterism_count_field(ao_system),
        _get_asterism_coverage_field(ao_system),
        _get_asterism_coverage_resolved_field(ao_system),
        _get_asterism_coverage_mean_field(ao_system),
        _get_asterism_sr_max_field(ao_system),
        _get_asterism_ee_max_field(ao_system),
        _get_asterism_ee_max_field_mean_field(ao_system),
        _get_asterism_ee_max_id_field(ao_system),
        _get_asterism_ee_max_distance_field(ao_system),
        _get_asterism_fwhm_min_field(ao_system)
    ]

def _has_complete_asterism_stats(inner_data, ao_systems):
    columns = set(inner_data[1].columns.names) # pylint: disable=no-member
    for ao_system in ao_systems:
        if not set(_get_managed_asterism_stats_fields(ao_system)).issubset(columns):
            return False
    return True

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
        ngs_filter = (gaia_stars[f"gaia_{ao_system['band']}"] >= ao_system['min_mag']) & (gaia_stars[f"gaia_{ao_system['band']}"] < ao_system['max_mag'])
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

def _load_inner(config, outer_pix, update=False, use_cache=False):
    if update and use_cache:
        raise AOMapException("Cannot update and use cache at the same time")

    filename = None

    if use_cache:
        cache_filename = _get_inner_pixel_data_cache_filename(config, outer_pix)
        if os.path.isfile(cache_filename):
            filename = cache_filename

    if filename is None:
        filename = _get_inner_pixel_data_filename(config, outer_pix)

    if not os.path.isfile(filename):
        raise AOMapException(f"Inner data not found for outer pixel {outer_pix}")

    inner_data = fits.open(filename, mode='update' if update else 'readonly')

    if use_cache and cache_filename != filename:
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        shutil.copy(filename, cache_filename)

    return inner_data

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

def _get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=1, force_reload=False, use_cache=False):
    filename = None

    if use_cache and not force_reload:
        cache_filename = _get_gaia_stars_cache_filename(config, outer_pix)
        if os.path.isfile(cache_filename):
            filename = cache_filename

    if filename is None:
        filename = f"{_get_gaia_path(config, outer_pix)}/gaia.fits"

    if not force_reload and os.path.isfile(filename):
        with fits.open(filename, mode='readonly') as hdul:
            gaia_stars = Table(hdul[1].data) # pylint: disable=no-member
    else:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

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

    if use_cache and cache_filename != filename:
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        shutil.copy(filename, cache_filename)

    return gaia_stars

#endregion

#region Asterisms

def _build_asterisms(config, mode, outer_pix, ao_system_name, verbose=False):
    # Exclusions that DO NOT require asterisms go here: excluded = ...
    bypass_latitude_cut = outer_pix in config.asterisms_galactic_latitude_bypass_pixs
    galactic_coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix).galactic
    excluded = (not bypass_latitude_cut) and (
        np.abs(galactic_coord.b.degree) < config.asterisms_min_galactic_latitude
    )

    if excluded:
        success = True
        num_asterisms = 0
    else:
        match mode:
            case 'build':
                asterism_filename = _get_asterisms_data_filename(config, outer_pix, ao_system_name)
                use_existing = os.path.isfile(asterism_filename) and _is_good_FITS(asterism_filename)
            case 'rebuild' | 'recalc':
                use_existing = False

        if use_existing:
            success = True
            num_asterisms = 0
        else:
            success, num_asterisms = _create_asterisms(config, outer_pix, ao_system_name, verbose=verbose)

    if not success:
        excluded = False

    return [success, excluded, num_asterisms]

def _create_asterisms(config, outer_pix, ao_system_name, verbose=False):
    asterisms = find_outer_asterisms(config, outer_pix, ao_system_name, verbose=verbose)
    if asterisms is None:
        return False
    if len(asterisms) > 0:
        _save_asterisms(config, outer_pix, ao_system_name, asterisms)
    return True, len(asterisms)

def get_stars_for_asterisms(config, outer_pix, neighbour_level=None, required_band=None, use_cache=False):
    gaia_data = _get_gaia_stars_in_outer_pixel(config, outer_pix, use_cache=use_cache)
    bypass_latitude_cut = outer_pix in config.asterisms_galactic_latitude_bypass_pixs

    if neighbour_level is not None:
        subpixs = healpix.get_subpixels(config.outer_level, outer_pix, neighbour_level)
        bordering_pixs = np.setdiff1d(healpix.get_neighbours(neighbour_level, subpixs).flatten(), subpixs)

        gaia_data['pix'] = healpix.get_healpix_from_skycoord(neighbour_level, SkyCoord(ra=gaia_data['gaia_ra'], dec=gaia_data['gaia_dec'], unit=(u.degree, u.degree)))
        all_gaia_data = [gaia_data]
        for pix in healpix.get_neighbours(config.outer_level, outer_pix):
            if pix < 0:
                continue
            coord = healpix.get_pixel_skycoord(neighbour_level, pix)
            if (not bypass_latitude_cut) and np.abs(coord.galactic.b.degree) < config.asterisms_min_galactic_latitude:
                continue
            neighbour_gaia_data = _get_gaia_stars_in_outer_pixel(config, pix, use_cache=use_cache)
            neighbour_gaia_data['pix'] = healpix.get_healpix_from_skycoord(neighbour_level, SkyCoord(ra=neighbour_gaia_data['gaia_ra'], dec=neighbour_gaia_data['gaia_dec'], unit=(u.degree, u.degree)))
            all_gaia_data.append(neighbour_gaia_data[np.isin(neighbour_gaia_data['pix'], bordering_pixs)])

        gaia_data = vstack(all_gaia_data)

    if config.asterism_epoch is not None:
        (gaia_data['gaia_ra'], gaia_data['gaia_dec']) = gaia.apply_proper_motion(gaia_data, epoch=config.asterism_epoch)
        gaia_data['gaia_ref_epoch'] = config.asterism_epoch

    for colname in gaia_data.colnames:
        if colname.startswith('gaia_'):
            gaia_data.rename_column(colname, colname[5:])

    star_filter = ~np.isnan(gaia_data['ra']) & ~np.isnan(gaia_data['dec'])
    gaia_data = gaia_data[star_filter]

    if required_band is not None:
        star_filter = ~np.isnan(gaia_data[required_band])
        gaia_data = gaia_data[star_filter]

    if isinstance(gaia_data, np.ma.MaskedArray):
        gaia_data = gaia_data.filled()

    return gaia_data

def find_outer_asterisms(config, outer_pix, ao_system_name, skip_overlaps=False, return_detail = False, use_cache=False, verbose=False):
    ao_system = get_ao_system(config, ao_system_name)
    band = ao_system['band']
    fov = ao_system['fov']
    fov_1ngs = ao_system['fov_1ngs']
    fov_rad = fov.to(u.rad).value
    fov_1ngs_rad = fov_1ngs.to(u.rad).value
    fov_level = healpix.get_level_with_resolution(fov)
    fov_level_area = healpix.get_area(fov_level).to(u.arcmin**2).value

    stars = get_stars_for_asterisms(config, outer_pix, neighbour_level=fov_level, required_band=band, use_cache=use_cache)

    # Filter NGS on Magnitude
    mag_filter = (stars[band] >= ao_system['min_mag']) & (stars[band] < ao_system['max_mag'])
    ngs = stars[mag_filter]

    # Filter NGS on Stellar Density
    unique_pixs, star_counts = np.unique(stars['pix'], return_counts=True)
    remove_pixs = unique_pixs[star_counts/fov_level_area > config.asterisms_max_star_density]
    if len(remove_pixs) > 0:
        ngs = ngs[~np.isin(ngs['pix'], remove_pixs)]

    # Find Asterisms
    asterisms = asterism.find_asterisms(
        ngs,
        mag_field = band,
        min_stars = ao_system['min_wfs'],
        max_stars = ao_system['max_wfs'],
        min_separation = ao_system['min_sep'].to(u.arcsec).value,
        max_separation = ao_system['max_sep'].to(u.arcsec).value,
        max_1ngs_distance = fov_1ngs.to(u.arcsec).value/2,
        verbose = verbose
    )

    asterisms.remove_column('star1_idx')
    asterisms.remove_column('star2_idx')
    asterisms.remove_column('star3_idx')

    asterism_centres = SkyCoord(ra=asterisms['ra'], dec=asterisms['dec'], unit=(u.degree, u.degree))

    if config.asterisms_max_bright_star_mag is not None:
        bright_star_filter = stars[band] < config.asterisms_max_bright_star_mag
        if np.sum(bright_star_filter) > 0:
            bright_stars = stars[bright_star_filter]
            bright_star_coords = SkyCoord(ra=bright_stars['ra'], dec=bright_stars['dec'], unit=(u.degree, u.degree))
            bright_star_sep = [np.min(centre.separation(bright_star_coords)) for centre in asterism_centres]
            asterism_filter = [sep > 2*fov for sep in bright_star_sep]
            asterisms = asterisms[asterism_filter]
            asterism_centres = asterism_centres[asterism_filter]

    if not skip_overlaps and config.asterisms_max_overlap is not None:
        asterism_filter = np.ones((len(asterisms)), dtype=bool)
        asterism_qualities = _get_asterism_quality(config, asterisms, ao_system)
        asterism_pixs = healpix.get_healpix_from_skycoord(fov_level, asterism_centres)

        pixs = np.unique(asterism_pixs)
        N = len(pixs)
        start_time = time.time()
        might_be_slow = len(remove_pixs) > 0
        for i, pix in enumerate(pixs):
            if verbose and might_be_slow and (i+1) % 500 == 0:
                print(f"{i+1}/{N}: {time.time() - start_time:.2f}s", flush=True)
                start_time = time.time()

            neighbour_pixs = healpix.get_neighbours(fov_level, pix)
            search_pixs = np.append(pix, neighbour_pixs[neighbour_pixs >= 0])

            search_asterism_idxs = np.argwhere(asterism_filter & np.isin(asterism_pixs, search_pixs)).flatten()
            if len(search_asterism_idxs) == 0:
                continue
            search_asterism_idxs = search_asterism_idxs[np.argsort(asterism_qualities[search_asterism_idxs], axis=0)[::-1]]

            skip = np.zeros((len(search_asterism_idxs)), dtype=bool)
            for j in range(len(search_asterism_idxs)): # pylint: disable=consider-using-enumerate
                if skip[j]:
                    continue

                idx1 = search_asterism_idxs[j]
                seps = asterism_centres[idx1].separation(asterism_centres[search_asterism_idxs[j+1:]]).to(u.rad).value

                for k in range(len(search_asterism_idxs)-j-1): # pylint: disable=consider-using-enumerate
                    if skip[j+1+k] or seps[k] > fov_rad:
                        continue

                    idx2 = search_asterism_idxs[j+1+k]
                    r1 = fov_rad if asterisms['num_stars'][idx1] > 1 else fov_1ngs_rad
                    r2 = fov_rad if asterisms['num_stars'][idx2] > 1 else fov_1ngs_rad
                    overlap_area = asterism.get_circle_overlap_area(r1, r2, seps[k])
                    overlap = overlap_area / (np.pi * min(r1, r2)**2)

                    if overlap > config.asterisms_max_overlap:
                        skip[j+1+k] = True

            current_pix_filter = asterism_pixs[search_asterism_idxs] == pix
            asterism_filter[search_asterism_idxs[current_pix_filter & skip]] = False

        if verbose and might_be_slow:
            print(f"{N}/{N}: {time.time() - start_time:.2f}s", flush=True)

        asterisms = asterisms[asterism_filter]
        asterism_centres = asterism_centres[asterism_filter]

    if ao_system['max_rel_sep'] > 0:
        asterism_filter = (asterisms['relsep'] >= ao_system['min_rel_sep']) & (asterisms['relsep'] < ao_system['max_rel_sep'])
        asterisms = asterisms[asterism_filter]
        asterism_centres = asterism_centres[asterism_filter]

    if ao_system['max_rel_area'] > 0:
        asterism_filter = (asterisms['relarea'] >= ao_system['min_rel_area']) & (asterisms['relarea'] < ao_system['max_rel_area'])
        asterisms = asterisms[asterism_filter]
        asterism_centres = asterism_centres[asterism_filter]

    # Only include asterisms whose centre is in the outer pixel
    # (do this last so asterisms around the edges are available above)
    asterism_outer_pixs = healpix.get_healpix_from_skycoord(config.outer_level, asterism_centres)
    asterism_filter = asterism_outer_pixs == outer_pix
    asterisms = asterisms[asterism_filter]
    asterism_centres = asterism_centres[asterism_filter]

    asterisms['pix'] = healpix.get_healpix_from_skycoord(config.inner_level, asterism_centres)

    if return_detail:
        return asterisms, stars, ngs, ao_system, fov_level
    else:
        return asterisms

model_cache = {}

def _load_models(ao_systems):
    training = get_training_module()
    model_cache = {}
    for ao_system in ao_systems:
        model_cache[ao_system['name']] = {}
        for key, model_name in ao_system['models'].items():
            model = training.load_model('../data/models', model_name, force_cpu=True)
            model_cache[ao_system['name']][key] = model

def _get_asterisms_EE(config, asterisms, ao_system, batch_size=10000):
    training = get_training_module()

    if 'models' not in ao_system or len(ao_system['models']) == 0:
        raise AOMapException("No models found in ao_system")

    check_angles = ao_system['rot_range'] is not None and ao_system['rot_step'] is not None

    qualities = np.zeros((len(asterisms)))
    best_angles = np.zeros((len(asterisms)))

    for num_stars in range(ao_system['min_wfs'], ao_system['max_wfs'] + 1):
        key = f"{num_stars}star"
        indexes = np.flatnonzero(asterisms['num_stars'] == num_stars)
        num_asterisms = len(indexes)
        if num_asterisms > 0 and key in ao_system['models']:
            if ao_system['name'] in model_cache and key in model_cache[ao_system['name']]:
                model = model_cache[ao_system['name']][key]
            else:
                model = training.load_model('../data/models', ao_system['models'][key], force_cpu=True)

            num_batches = int(np.ceil(num_asterisms/batch_size))
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_asterisms)
                if start_idx >= end_idx:
                    continue
                batch_indexes = indexes[start_idx:end_idx]

                data = {
                    'wavelength': get_prediction_wavelength(config),
                    'lgs': ao_system['lgs'],
                    'ngs': asterism.get_ngs_from_asterisms(asterisms[batch_indexes]),
                    'r': np.zeros((len(batch_indexes),), dtype=np.float64),
                    'theta': np.zeros((len(batch_indexes),), dtype=np.float64),
                    'x': np.zeros((len(batch_indexes),), dtype=np.float64),
                    'y': np.zeros((len(batch_indexes),), dtype=np.float64),
                }

                X = training.get_model_X(data, mean_only=True)
                if check_angles:
                    # Asterism-centered rotation optimization: build mean-model features
                    # in the asterism frame, sweep the sampled instrument angles, and
                    # keep the angle that maximizes EE for catalog ranking. See
                    # _predict_inner_pixel_asterism_performance_batch() for the distinct
                    # inner-pixel-centered rotation search used for on-axis sky mapping.
                    theta_idxs = training.get_ngs_theta_indexes(num_stars)
                    rot_angles = [0.0] + [angle for angle in np.arange(ao_system['rot_range'][0], ao_system['rot_range'][1], ao_system['rot_step']) if angle != 0]

                    batch_ee_max = np.zeros((X.shape[0]))
                    batch_best_angles = np.zeros((X.shape[0]))

                    last_rot_angle = 0.0
                    for rot_angle in rot_angles:
                        delta_theta = np.deg2rad(rot_angle - last_rot_angle)
                        if delta_theta != 0:
                            X[:, theta_idxs] += delta_theta
                            X[:, theta_idxs] = training.wrap_angle_rad(X[:, theta_idxs])
                        Y_pred = training.get_prediction(X, model)
                        ee = Y_pred[:, training.get_ee_index()]
                        batch_best_angles[ee > batch_ee_max] = rot_angle
                        batch_ee_max = np.maximum(batch_ee_max, ee)
                        last_rot_angle = rot_angle

                    qualities[batch_indexes] = batch_ee_max
                    best_angles[batch_indexes] = batch_best_angles
                else:
                    Y_pred = training.get_prediction(X, model)
                    qualities[batch_indexes] = Y_pred[:, training.get_ee_index()]

                training.clear_cache(model)

    asterisms['best_ee'] = qualities
    asterisms['best_angle'] = best_angles

    return qualities

def _get_asterism_quality(config, asterisms, ao_system):
    if 'models' in ao_system:
        return _get_asterisms_EE(config, asterisms, ao_system)

    qualities = np.zeros((len(asterisms)))

    max_separation = ao_system['fov'].to(u.arcsec).value
    radius_1ngs = ao_system['fov_1ngs'].to(u.arcsec).value/2

    min_rel_factor_small = 0.25
    min_rel_factor_large = 0.5
    min_rel_sep = radius_1ngs / max_separation
    mid_rel_sep = 0.5
    max_rel_sep = 1.0

    below_mid_slope = (1.0 - min_rel_factor_small) / (mid_rel_sep - min_rel_sep)
    above_mid_slope = (1.0 - min_rel_factor_large) / (max_rel_sep - mid_rel_sep)

    for i, a in enumerate(asterisms):
        rel_sep = a['relsep'] if a['num_stars'] > 1 else min_rel_sep
        if rel_sep < 0.5:
            rel_factor = max(min_rel_factor_small, 1.0 - below_mid_slope * (mid_rel_sep - rel_sep))
        else:
            rel_factor = max(min_rel_factor_large, 1.0 - above_mid_slope * (rel_sep - mid_rel_sep))

        if ao_system['max_mag'] == ao_system['nom_mag']:
            mag_factor1 = 1
        else:
            mag_factor1 = min(1, (ao_system['max_mag'] - a['star1_mag']) / (ao_system['max_mag'] - ao_system['nom_mag']))

        if a['num_stars'] >= 2:
            if ao_system['max_mag'] == ao_system['nom_mag']:
                mag_factor2 = 1
            else:
                mag_factor2 = min(1, (ao_system['max_mag'] - a['star2_mag']) / (ao_system['max_mag'] - ao_system['nom_mag']))
        else:
            mag_factor2 = 0

        if a['num_stars'] >= 3:
            if ao_system['max_mag'] == ao_system['nom_mag']:
                mag_factor3 = 1
            else:
                mag_factor3 = min(1, (ao_system['max_mag'] - a['star3_mag']) / (ao_system['max_mag'] - ao_system['nom_mag']))
        else:
            mag_factor3 = 0

        mag_factor = (mag_factor1 + mag_factor2 + mag_factor3) / 3

        qualities[i] = rel_factor * mag_factor

    return qualities

def _save_asterisms(config, outer_pix, ao_system_name, asterisms):
    asterisms_filename = _get_asterisms_data_filename(config, outer_pix, ao_system_name)
    os.makedirs(os.path.dirname(asterisms_filename), exist_ok=True)
    with prevent_interruption():
        asterisms.write(asterisms_filename, format='fits', overwrite=True)

def load_asterisms(config, outer_pix, ao_system_name, include_neighbours=False, max_dust_extinction=None, return_none_if_missing=False):
    asterisms_filename = _get_asterisms_data_filename(config, outer_pix, ao_system_name)
    if not os.path.isfile(asterisms_filename):
        if return_none_if_missing:
            return None
        else:
            raise AOMapException(f"Asterisms file not found: {asterisms_filename}")

    asterisms = Table.read(asterisms_filename, format='fits')

    if include_neighbours:
        for pix in healpix.get_neighbours(config.outer_level, outer_pix):
            if pix < 0:
                continue
            neighbour_asterisms = load_asterisms(config, pix, ao_system_name, return_none_if_missing=True)
            if asterisms is None:
                asterisms = neighbour_asterisms
            elif neighbour_asterisms is not None:
                asterisms = vstack([asterisms, neighbour_asterisms])

    if max_dust_extinction is not None and 'Av' in asterisms.colnames:
        asterism_filter = asterisms['Av'] <= max_dust_extinction
        asterisms = asterisms[asterism_filter]

    return asterisms

def _append_asterism_dust_extinction(config, outer_pix, ao_system, dust):
    asterisms = load_asterisms(config, outer_pix, ao_system['name'], return_none_if_missing=True)
    if asterisms is not None:
        if 'Av' not in asterisms.colnames:
            asterisms['Av'] = dust.query(SkyCoord(ra=asterisms['ra'], dec=asterisms['dec'], unit=(u.degree, u.degree)))
            _save_asterisms(config, outer_pix, ao_system['name'], asterisms)

def _get_inner_pixel_asterism_count(config, outer_pix, ao_system):
    asterisms = load_asterisms(config, outer_pix, ao_system['name'], max_dust_extinction=config.asterisms_max_dust_extinction, return_none_if_missing=True)
    if asterisms is not None and len(asterisms) > 0:
        pixs = healpix.get_subpixels(config.outer_level, outer_pix, config.inner_level)
        unique_pixs, counts = np.unique(asterisms['pix'], return_counts=True)
        count_map = dict(zip(unique_pixs, counts))
        asterism_count = np.array([count_map.get(p, 0) for p in pixs])
    else:
        npix = healpix.get_subpixel_npix(config.outer_level, config.inner_level)
        asterism_count = np.zeros((npix), dtype=np.int_)

    return asterism_count

def _get_inner_pixel_asterism_coverage_and_performance(config, outer_pix, ao_system, batch_size=10000):
    npix = healpix.get_subpixel_npix(config.outer_level, config.inner_level)
    seeing_baseline = get_seeing_baseline_performance(config)

    stats = StructType()
    stats.coverage = np.zeros((npix))
    stats.resolved_coverage = np.zeros((npix))
    stats.mean_coverage = np.zeros((npix))
    stats.sr_max = np.full((npix), np.nan if seeing_baseline.sr is None else seeing_baseline.sr)
    stats.ee_max = np.full((npix), np.nan if seeing_baseline.ee is None else seeing_baseline.ee)
    stats.ee_max_field_mean = np.full((npix), np.nan)
    stats.ee_max_id = np.full((npix), np.nan)
    stats.ee_max_distance = np.full((npix), np.nan)
    stats.fwhm_min = np.full((npix), np.nan if seeing_baseline.fwhm is None else seeing_baseline.fwhm)
    stats.ee_max_angle = np.full((npix), np.nan)
    stats.ee_max_asterism_idx = np.full((npix), -1, dtype=np.int_)

    context = _get_inner_pixel_asterism_context(config, outer_pix, ao_system)
    if context.asterisms is None or len(context.asterisms) == 0 or len(context.pixel_idxs) == 0:
        return stats

    stats.coverage[np.unique(context.pixel_idxs)] = 1.0

    if 'point_models' not in ao_system or len(ao_system['point_models']) == 0:
        return stats

    _update_inner_pixel_asterism_performance(
        config,
        stats,
        context,
        ao_system,
        np.full((len(context.pixel_idxs)), True),
        batch_size=batch_size
    )

    if 'models' in ao_system and len(ao_system['models']) > 0:
        _update_inner_pixel_asterism_field_mean(config, stats, context, ao_system, batch_size=batch_size)

    stats.resolved_coverage = (stats.ee_max >= config.coverage_ee_threshold_resolved).astype(np.float64)
    stats.mean_coverage = (stats.ee_max_field_mean >= config.coverage_ee_threshold_mean).astype(np.float64)

    return stats

def _get_inner_pixel_asterism_context(config, outer_pix, ao_system):
    context = StructType()
    context.outer_pix = outer_pix
    context.pixel_idxs = np.array([], dtype=np.int_)
    context.asterism_idxs = np.array([], dtype=np.int_)

    _, context.inner_centres = healpix.get_subpixels_skycoord(config.outer_level, outer_pix, config.inner_level)
    outer_centre = healpix.get_pixel_skycoord(config.outer_level, outer_pix)
    context.inner_x, context.inner_y = _get_plane_offsets_arcsec(outer_centre, context.inner_centres)

    local_asterisms = load_asterisms(
        config,
        outer_pix,
        ao_system['name'],
        max_dust_extinction=config.asterisms_max_dust_extinction,
        return_none_if_missing=True
    )

    tables = []
    locality = []
    if local_asterisms is not None and len(local_asterisms) > 0:
        tables.append(local_asterisms)
        locality.append(np.ones((len(local_asterisms)), dtype=bool))

    for pix in healpix.get_neighbours(config.outer_level, outer_pix):
        if pix < 0:
            continue
        neighbour_asterisms = load_asterisms(
            config,
            pix,
            ao_system['name'],
            max_dust_extinction=config.asterisms_max_dust_extinction,
            return_none_if_missing=True
        )
        if neighbour_asterisms is None or len(neighbour_asterisms) == 0:
            continue
        tables.append(neighbour_asterisms)
        locality.append(np.zeros((len(neighbour_asterisms)), dtype=bool))

    context.asterisms = None if len(tables) == 0 else tables[0] if len(tables) == 1 else vstack(tables)
    context.local_asterism_mask = np.array([], dtype=bool) if len(locality) == 0 else np.concatenate(locality)

    if context.asterisms is None or len(context.asterisms) == 0:
        return context

    asterism_catalog = SkyCoord(ra=context.asterisms['ra'], dec=context.asterisms['dec'], unit=(u.degree, u.degree))
    context.asterism_x, context.asterism_y = _get_plane_offsets_arcsec(outer_centre, asterism_catalog)

    inner_resolution = healpix.get_resolution(config.inner_level)
    context.pixel_idxs, context.asterism_idxs, _, _ = search_around_sky(
        context.inner_centres,
        asterism_catalog,
        (ao_system['fov'] - inner_resolution)/2
    )

    context.star_x = {}
    context.star_y = {}
    for star_idx in range(1, 4):
        star_coords = SkyCoord(
            ra=context.asterisms[f'star{star_idx}_ra'],
            dec=context.asterisms[f'star{star_idx}_dec'],
            unit=(u.degree, u.degree)
        )
        context.star_x[star_idx], context.star_y[star_idx] = _get_plane_offsets_arcsec(outer_centre, star_coords)

    return context

def _get_plane_offsets_arcsec(reference_coord, skycoords):
    lon_offset, lat_offset = reference_coord.spherical_offsets_to(skycoords)
    return lon_offset.to(u.arcsec).value, lat_offset.to(u.arcsec).value

def _get_point_model_cache_key(ao_system_name, num_stars):
    return f"{ao_system_name}:{num_stars}star"

point_model_cache = {}
mean_model_cache = {}

def _get_mean_model(ao_system, num_stars):
    training = get_training_module()

    key = f"{num_stars}star"
    if 'models' not in ao_system or key not in ao_system['models']:
        return None

    cache_key = _get_point_model_cache_key(f"{ao_system['name']}:mean", num_stars)
    if cache_key not in mean_model_cache:
        mean_model_cache[cache_key] = training.load_model('../data/models', ao_system['models'][key], force_cpu=True)

    return mean_model_cache[cache_key]

def _get_point_model(ao_system, num_stars):
    training = get_training_module()

    key = f"{num_stars}star"
    if 'point_models' not in ao_system or key not in ao_system['point_models']:
        return None

    cache_key = _get_point_model_cache_key(ao_system['name'], num_stars)
    if cache_key not in point_model_cache:
        point_model_cache[cache_key] = training.load_model('../data/models', ao_system['point_models'][key], force_cpu=True)

    return point_model_cache[cache_key]

def _get_ao_lgs_xy(lgs):
    lgs_with_xy = []
    for star in lgs:
        lgs_with_xy.append({
            **star,
            'x': star['zd'] * np.cos(np.deg2rad(star['az'])),
            'y': star['zd'] * np.sin(np.deg2rad(star['az']))
        })
    return lgs_with_xy

def _get_rotation_angles(ao_system):
    rot_angles = [0.0]
    if ao_system['rot_range'] is not None and ao_system['rot_step'] is not None:
        rot_angles += [angle for angle in np.arange(ao_system['rot_range'][0], ao_system['rot_range'][1], ao_system['rot_step']) if angle != 0]
    return rot_angles

def _get_valid_ngs_from_context_pair(context, pixel_idx, asterism_idx, field_radius, field_radius_1ngs):
    all_stars = []
    pixel_x = context.inner_x[pixel_idx]
    pixel_y = context.inner_y[pixel_idx]
    num_stars = int(context.asterisms['num_stars'][asterism_idx])

    for star_idx in range(1, num_stars + 1):
        dx = context.star_x[star_idx][asterism_idx] - pixel_x
        dy = context.star_y[star_idx][asterism_idx] - pixel_y
        zd = np.hypot(dx, dy)
        all_stars.append({
            'zd': zd,
            'az': np.rad2deg(np.arctan2(dx, dy)),
            'mag': context.asterisms[f'star{star_idx}_mag'][asterism_idx]
        })

    stars_in_fov = [star for star in all_stars if star['zd'] <= field_radius]
    if len(stars_in_fov) >= 2:
        return stars_in_fov

    stars_in_fov_1ngs = [star for star in all_stars if star['zd'] <= field_radius_1ngs]
    return stars_in_fov_1ngs

def _predict_inner_pixel_asterism_performance_batch(config, ao_system, num_stars, model, ngs):
    training = get_training_module()

    lgs = _get_ao_lgs_xy(ao_system['lgs'])
    theta_idxs = training.get_ngs_theta_indexes(num_stars)
    rot_angles = _get_rotation_angles(ao_system)
    num_pairs = len(ngs)

    metrics = StructType()
    metrics.sr = np.full((num_pairs), np.nan)
    metrics.ee = np.full((num_pairs), np.nan)
    metrics.fwhm = np.full((num_pairs), np.nan)
    metrics.ee_angle = np.full((num_pairs), np.nan)

    data = {
        'wavelength': get_prediction_wavelength(config),
        'lgs': lgs,
        'r': np.zeros((num_pairs)),
        'theta': np.zeros((num_pairs)),
        'ngs': ngs,
        'x': np.zeros((num_pairs)),
        'y': np.zeros((num_pairs))
    }
    X = training.get_model_X(data)

    if len(rot_angles) == 1:
        Y_pred = training.get_prediction(X, model, skip_cache_clear=True)
        metrics.sr = Y_pred[:, training.get_sr_index()]
        metrics.ee = Y_pred[:, training.get_ee_index()]
        metrics.fwhm = Y_pred[:, training.get_fwhm_index()]
        metrics.ee_angle = np.zeros((num_pairs))
        return metrics

    X_rot = np.tile(X, (len(rot_angles), 1))
    # Inner-pixel-centered rotation optimization: recenter the NGS geometry on
    # the science target, evaluate the resolved model over sampled angles, and
    # keep the best per-metric result for that sky position. See
    # _get_asterisms_EE() for the distinct asterism-centered ranking search.
    for rot_idx, rot_angle in enumerate(rot_angles):
        if rot_angle == 0:
            continue
        start_idx = rot_idx * num_pairs
        end_idx = start_idx + num_pairs
        X_rot[start_idx:end_idx, theta_idxs] += np.deg2rad(rot_angle)
        X_rot[start_idx:end_idx, theta_idxs] = training.wrap_angle_rad(X_rot[start_idx:end_idx, theta_idxs])

    Y_pred = training.get_prediction(X_rot, model, skip_cache_clear=True)
    Y_pred = Y_pred.reshape(len(rot_angles), num_pairs, Y_pred.shape[1])
    metrics.sr = np.max(Y_pred[:, :, training.get_sr_index()], axis=0)
    ee_values = Y_pred[:, :, training.get_ee_index()]
    best_ee_rot_idxs = np.argmax(ee_values, axis=0)
    metrics.ee = ee_values[best_ee_rot_idxs, np.arange(num_pairs)]
    metrics.ee_angle = np.array(rot_angles, dtype=np.float64)[best_ee_rot_idxs]
    metrics.fwhm = np.min(Y_pred[:, :, training.get_fwhm_index()], axis=0)

    return metrics

def _predict_inner_pixel_asterism_field_mean_batch(config, ao_system, num_stars, model, ngs, rot_angles):
    training = get_training_module()
    num_pairs = len(ngs)

    data = {
        'wavelength': get_prediction_wavelength(config),
        'lgs': ao_system['lgs'],
        'ngs': ngs,
        'r': np.zeros((num_pairs)),
        'theta': np.zeros((num_pairs)),
        'x': np.zeros((num_pairs)),
        'y': np.zeros((num_pairs))
    }
    X = training.get_model_X(data, mean_only=True)
    theta_idxs = training.get_ngs_theta_indexes(num_stars)
    if len(theta_idxs) > 0:
        X[:, theta_idxs] += np.deg2rad(np.array(rot_angles, dtype=np.float64)).reshape(-1, 1)
        X[:, theta_idxs] = training.wrap_angle_rad(X[:, theta_idxs])

    Y_pred = training.get_prediction(X, model, skip_cache_clear=True)
    return Y_pred[:, training.get_ee_index()]

def _update_inner_pixel_asterism_performance(config, stats, context, ao_system, pair_filter, batch_size=10000):
    field_radius = ao_system['fov'].to(u.arcsec).value / 2.0
    field_radius_1ngs = ao_system['fov_1ngs'].to(u.arcsec).value / 2.0

    candidate_pixel_idxs = context.pixel_idxs[pair_filter]
    candidate_asterism_idxs = context.asterism_idxs[pair_filter]

    num_batches = int(np.ceil(len(candidate_pixel_idxs) / batch_size))
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(candidate_pixel_idxs))
        if start_idx >= end_idx:
            continue

        batch_pixel_idxs = candidate_pixel_idxs[start_idx:end_idx]
        batch_asterism_idxs = candidate_asterism_idxs[start_idx:end_idx]
        grouped_pairs = {}
        for row_idx, (pixel_idx, asterism_idx) in enumerate(zip(batch_pixel_idxs, batch_asterism_idxs)):
            # Recentering on the inner pixel can push some catalog asterism stars
            # outside the patrol radius. Those stars were not part of the model's
            # training domain and must be dropped before choosing the 1/2/3-star
            # inference model for this pixel-centered evaluation.
            ngs = _get_valid_ngs_from_context_pair(context, pixel_idx, asterism_idx, field_radius, field_radius_1ngs)
            surviving_stars = len(ngs)
            if surviving_stars < ao_system['min_wfs']:
                continue

            # Re-bucket by surviving star count so each group can use the
            # matching 1/2/3-star point model in one homogeneous batch.
            if surviving_stars not in grouped_pairs:
                grouped_pairs[surviving_stars] = {
                    'row_idxs': [],
                    'pixel_idxs': [],
                    'asterism_idxs': [],
                    'ngs': []
                }

            grouped_pairs[surviving_stars]['row_idxs'].append(row_idx)
            grouped_pairs[surviving_stars]['pixel_idxs'].append(pixel_idx)
            grouped_pairs[surviving_stars]['asterism_idxs'].append(asterism_idx)
            grouped_pairs[surviving_stars]['ngs'].append(ngs)

        for surviving_stars, group in grouped_pairs.items():
            model = _get_point_model(ao_system, surviving_stars)
            if model is None:
                continue

            # Run one batched inference for all pixel/asterism pairs that share
            # this post-FOV-cut guide-star count.
            metrics = _predict_inner_pixel_asterism_performance_batch(
                config,
                ao_system,
                surviving_stars,
                model,
                group['ngs']
            )

            # Scatter the batched outputs back to individual candidate pairs and
            # update each inner pixel only when this candidate improves the
            # current best metric.
            for result_idx, (row_idx, pixel_idx, asterism_idx, sr, ee, fwhm) in enumerate(zip(
                group['row_idxs'],
                group['pixel_idxs'],
                group['asterism_idxs'],
                metrics.sr,
                metrics.ee,
                metrics.fwhm
            )):
                if np.isfinite(sr) and (np.isnan(stats.sr_max[pixel_idx]) or sr > stats.sr_max[pixel_idx]):
                    stats.sr_max[pixel_idx] = sr
                if np.isfinite(ee) and (np.isnan(stats.ee_max[pixel_idx]) or ee > stats.ee_max[pixel_idx]):
                    stats.ee_max[pixel_idx] = ee
                    stats.ee_max_angle[pixel_idx] = metrics.ee_angle[result_idx]
                    stats.ee_max_asterism_idx[pixel_idx] = asterism_idx
                    stats.ee_max_id[pixel_idx] = context.asterisms['id'][asterism_idx] if context.local_asterism_mask[asterism_idx] else np.nan
                    stats.ee_max_distance[pixel_idx] = np.hypot(
                        context.asterism_x[asterism_idx] - context.inner_x[pixel_idx],
                        context.asterism_y[asterism_idx] - context.inner_y[pixel_idx]
                    )
                if np.isfinite(fwhm) and (np.isnan(stats.fwhm_min[pixel_idx]) or fwhm < stats.fwhm_min[pixel_idx]):
                    stats.fwhm_min[pixel_idx] = fwhm

def _update_inner_pixel_asterism_field_mean(config, stats, context, ao_system, batch_size=10000):
    winner_pixel_idxs = np.flatnonzero(stats.ee_max_asterism_idx >= 0)
    if len(winner_pixel_idxs) == 0:
        return

    field_radius = ao_system['fov'].to(u.arcsec).value / 2.0
    field_radius_1ngs = ao_system['fov_1ngs'].to(u.arcsec).value / 2.0
    grouped_pairs = {}
    for pixel_idx in winner_pixel_idxs:
        asterism_idx = int(stats.ee_max_asterism_idx[pixel_idx])
        ngs = _get_valid_ngs_from_context_pair(context, pixel_idx, asterism_idx, field_radius, field_radius_1ngs)
        surviving_stars = len(ngs)
        if surviving_stars < ao_system['min_wfs']:
            continue

        if surviving_stars not in grouped_pairs:
            grouped_pairs[surviving_stars] = {
                'pixel_idxs': [],
                'ngs': [],
                'angles': []
            }

        grouped_pairs[surviving_stars]['pixel_idxs'].append(pixel_idx)
        grouped_pairs[surviving_stars]['ngs'].append(ngs)
        grouped_pairs[surviving_stars]['angles'].append(stats.ee_max_angle[pixel_idx])

    for surviving_stars, group in grouped_pairs.items():
        model = _get_mean_model(ao_system, surviving_stars)
        if model is None:
            continue

        num_rows = len(group['pixel_idxs'])
        num_batches = int(np.ceil(num_rows / batch_size))
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_rows)
            if start_idx >= end_idx:
                continue

            batch_pixel_idxs = group['pixel_idxs'][start_idx:end_idx]
            batch_ngs = group['ngs'][start_idx:end_idx]
            batch_angles = group['angles'][start_idx:end_idx]
            ee_field_mean = _predict_inner_pixel_asterism_field_mean_batch(
                config,
                ao_system,
                surviving_stars,
                model,
                batch_ngs,
                batch_angles
            )
            stats.ee_max_field_mean[batch_pixel_idxs] = ee_field_mean

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
        case _ if key.startswith('ngs-pix-density'):
            title = 'NGS Pix Density'
        case _ if key.startswith('ngs-pix'):
            title = 'NGS Pix'
        case _ if key.startswith('asterism-coverage') and key.endswith('-resolved'):
            title = 'AO Coverage (Resolved)'
        case _ if key.startswith('asterism-coverage') and key.endswith('-mean'):
            title = 'AO Coverage (Field Mean)'
        case _ if key.startswith('asterism-coverage'):
            title = 'Asterism Coverage'
        case _ if key.startswith('asterism-sr-max'):
            title = 'Best SR'
        case _ if key.startswith('asterism-ee-max'):
            title = 'Best EE'
        case _ if key.startswith('asterism-fwhm-min'):
            title = 'Best FWHM'
        case _ if key.startswith('ao-friendly'):
            title = 'AO-Friendly Areas'
        case _:
            title = f"{key} Map"

    if ao_system is not None:
        title = f"{title} ({ao_system['name']})"

    return title

def _get_map_norm(key, unit): # pylint: disable=unused-argument
    if any(token in key for token in ['sr-max', 'ee-max', 'fwhm-min']):
        return 'linear'
    if unit == 'percent':
        return 'linear'
    return 'log'

def _read_FITS_single_column_values(config, hdu, level, key, ao_system):
    data = hdu.data
    cols = hdu.columns

    level_area = healpix.get_area(level).to(u.arcmin**2)
    inner_area = healpix.get_area(config.inner_level).to(u.arcmin**2)

    match key:
        case _ if key.startswith('ao-friendly'):
            if ao_system is not None:
                fov = ao_system['fov']
                fov_area = np.pi * (fov/2)**2
                wfs = ao_system['max_wfs']
                max_field = 'STAR_COUNT'
                min_field = _get_ngs_pix_field(ao_system)
                field = _get_ngs_pix_field(ao_system)
            else:
                fov_area = np.pi * u.arcsec**2
                wfs = 3
                max_field = 'STAR_COUNT'
                min_field = 'STAR_COUNT'
                field = 'STAR_COUNT'

            max_count = 1/inner_area * level_area
            min_count = wfs/fov_area * level_area
            has_values = (data[max_field] < max_count) & (data[min_field] > min_count)

            if config.max_dust_extinction is not None:
                has_values &= data[FITS_COLUMN_DUST_EXTINCTION] <= config.max_dust_extinction

            if config.min_ecliptic_latitude is not None:
                ecliptic_coords = healpix.get_pixel_skycoord(level, data[FITS_COLUMN_PIX]).barycentrictrueecliptic
                has_values &= np.abs(ecliptic_coords.lat.degree) >= config.min_ecliptic_latitude

            values = np.full((len(data)), np.nan)
            values[has_values] = data[field][has_values] / level_area
            is_density = True
        case _:
            field = _get_field_from_key(key)
            if 'DENSITY' in field and key != 'model-density':
                if 'PIX' in field:
                    field = field.replace('_DENSITY', '')
                else:
                    field = field.replace('DENSITY', 'COUNT')
                factor = 1/level_area
                is_density = True
            else:
                if cols[field].format == 'K':
                    factor = 1
                else:
                    factor = 1.0
                is_density = False

            values = (factor.value if isinstance(factor, u.Quantity) else factor) * data[field]

    if is_density and 'asterism' in key:
        values *= 60**2  # Convert to 1/deg^2

    FITS_format = cols[field].format # pylint: disable=no-member
    unit = cols[field].unit # pylint: disable=no-member
    if is_density:
        FITS_format = 'D'
        if 'asterism' in key:
            unit = f"{unit}/deg^2"
        else:
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
        match tmp_FITS_format:
            case 'K':
                tmp_values = tmp_values.astype(np.int_)
            case 'L':
                tmp_values = tmp_values.astype(np.bool_)
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

def _get_inner_values(config, outer_pix, keys, ao_system, use_cache=False, return_details=False):
    inner_data = _load_inner(config, outer_pix, use_cache=use_cache)
    retval = _read_FITS_column_values(config, inner_data[1], config.inner_level, keys, ao_system, return_details=return_details)
    inner_data.close()
    return retval

def _get_data_values(config, level, keys, ao_system, return_details=False):
    hdul = _load_data(config, level)
    retval = _read_FITS_column_values(config, hdul[1], level, keys, ao_system, return_details=return_details)
    hdul.close()
    return retval

def get_dummy_map_data(config, map_level, dummy_value=1.0, level=None, pixs=None, coords=(), ra_limit=None, dec_limit=None, survey=None, allow_slow=False):
    map_data = get_map_data(config, map_level, 'star-count', level=level, pixs=pixs, coords=coords, ra_limit=ra_limit, dec_limit=dec_limit, survey=survey, allow_slow=allow_slow)
    map_data.key = 'dummy'
    map_data.values = np.full_like(map_data.values, dummy_value, dtype=np.float64)
    map_data.FITS_format = 'D'
    map_data.unit = None
    map_data.num_format = None
    map_data.norm = None
    map_data.title = None
    return map_data

def get_map_data(config, map_level, key, level=None, pixs=None, coords=(), ra_limit=None, dec_limit=None, survey=None, allow_slow=False, nan_below=None, use_cache=False):
    if pixs is not None and level is None:
        raise AOMapException('level required if pixs is provided')

    if level is not None and pixs is None:
        raise AOMapException('pixs required if level is provided')

    if level is not None and level > map_level:
        raise AOMapException('level must be less than map_level')

    if survey is not None and (map_level > config.max_data_level or map_level < config.outer_level):
        raise AOMapException('survey not supported at this level')

    if pixs is not None and not (isinstance(pixs, list) or isinstance(pixs, np.ndarray)):
        pixs = [pixs]

    ao_key = next((prefix for prefix in ['ao-friendly', 'ngs-density', 'ngs-pix-density', 'ngs-count', 'ngs-pix', 'asterism-coverage', 'asterism-sr-max', 'asterism-ee-max', 'asterism-fwhm-min'] if f"{prefix}-" in key), None)
    if ao_key is not None:
        if ao_key == 'asterism-coverage' and key.endswith('-resolved'):
            ao_system_name = key.replace(f"{ao_key}-", '').rsplit('-resolved', 1)[0]
        elif ao_key == 'asterism-coverage' and key.endswith('-mean'):
            ao_system_name = key.replace(f"{ao_key}-", '').rsplit('-mean', 1)[0]
        else:
            ao_system_name = key.replace(f"{ao_key}-",'')
        ao_system = get_ao_system(config, ao_system_name)
    else:
        ao_system = None

    if map_level > config.max_data_level:
        if pixs is None:
            raise AOMapException('pixs required as full sky maps at this level are not supported')

        if map_level > config.inner_level:
            raise AOMapException(f"map_level must be less than or equal to {config.inner_level}")

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

        ([map_pixs, map_values], [_, FITS_format], [_, unit]) = _get_inner_values(
            config,
            outer_pixs[0],
            ['pix', key],
            ao_system,
            use_cache=use_cache,
            return_details=True,
        )

        if len(pixs) > 1:
            for outer_pix in outer_pixs[1:]:
                [tmp_map_pixs, tmp_map_values] = _get_inner_values(
                    config,
                    outer_pix,
                    ['pix', key],
                    ao_system,
                    use_cache=use_cache,
                )
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
        if survey is not None:
            survey_column = _get_survey_column(survey)
            ([map_pixs, map_values, map_filter], [_, FITS_format, _], [_, unit, _]) = _get_data_values(config, max(map_level, config.outer_level), ['pix', key, survey_column], ao_system, return_details=True)
        else:
            ([map_pixs, map_values], [_, FITS_format], [_, unit]) = _get_data_values(config, max(map_level, config.outer_level), ['pix', key], ao_system, return_details=True)

            if map_level < config.outer_level:
                map_pixs = _decrease_pix_level(map_pixs, config.outer_level - map_level)
                map_values = _decrease_values_level(map_values, _get_field_aggregate_method(key), config.outer_level - map_level)

        if pixs is not None:
            if level < map_level:
                map_pixs = healpix.get_subpixels(level, pixs, map_level)
            else:
                map_pixs = pixs

            map_values = map_values[map_pixs]
            if survey is not None:
                map_filter = map_filter[map_pixs]

        if survey is not None:
            map_values = map_values[map_filter]
            map_pixs = map_pixs[map_filter]

    map_coords = healpix.get_pixel_skycoord(map_level, map_pixs)

    if len(coords) > 0:
        row_filter = healpix.filter_skycoords(map_coords, coords)
        map_pixs = map_pixs[row_filter]
        map_values = map_values[row_filter]
        map_coords = map_coords[row_filter]

    if ra_limit is not None:
        ra_filter = (map_coords.ra.degree >= ra_limit[0]) & (map_coords.ra.degree <= ra_limit[1])
        map_pixs = map_pixs[ra_filter]
        map_values = map_values[ra_filter]
        map_coords = map_coords[ra_filter]

    if dec_limit is not None:
        dec_filter = (map_coords.dec.degree >= dec_limit[0]) & (map_coords.dec.degree <= dec_limit[1])
        map_pixs = map_pixs[dec_filter]
        map_values = map_values[dec_filter]
        map_coords = map_coords[dec_filter]

    if nan_below is not None:
        map_values[map_values < nan_below] = np.nan

    if FITS_format == 'K' or FITS_format == 'L':
        num_format = '{x:.0f}'
    else:
        if unit == 'percent':
            num_format = '{x:.0%}'
        else:
            num_format = '{x:.1f}'

    map_data = StructType()
    map_data.key = key
    map_data.level = map_level
    map_data.pixs = map_pixs
    map_data.values = map_values
    map_data.coords = map_coords
    map_data.FITS_format = FITS_format
    map_data.unit = unit if unit != 'percent' else None
    map_data.num_format = num_format
    map_data.norm = _get_map_norm(key, unit)
    map_data.title = _get_map_title(key, ao_system)
    return map_data

def mask_map_data(map_data, mask):
    if mask is None:
        return map_data

    if len(mask) != len(map_data.pixs):
        raise AOMapException('mask length does not match map data length')

    masked_map_data = StructType()
    masked_map_data.key = map_data.key
    masked_map_data.level = map_data.level
    masked_map_data.pixs = map_data.pixs[mask]
    masked_map_data.values = map_data.values[mask]
    masked_map_data.coords = map_data.coords[mask]
    masked_map_data.FITS_format = map_data.FITS_format
    masked_map_data.unit = map_data.unit
    masked_map_data.num_format = map_data.num_format
    masked_map_data.norm = map_data.norm
    masked_map_data.title = map_data.title
    return masked_map_data

def get_map_table(config, map_level, key, level=None, pixs=None, coords=(), ra_limit=None, dec_limit=None, survey=None):
    map_data = get_map_data(config, map_level, key, level=level, pixs=pixs, coords=coords, ra_limit=ra_limit, dec_limit=dec_limit, survey=survey)

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
            contours=None,                 # plot countours using additional map_data
            galactic=False,                # rotate map from celestial to galactic coordinates
            projection='astro',            # astro | cart [hours | degrees | longitude]
            zoom=None,                     # zoom in on the map
            rotation=None,                 # longitudinal rotation angle in degrees
            width=None,                    # figure width
            height=None,                   # figure height
            dpi=None,                      # figure dpi
            xsize=None,                    # x-axis pixel dimensions
            grid=True,                     # display grid lines
            cmap=None,                     # colormap
            norm=None,                     # normalization
            vmin=None,                     # minimum value for color normalization
            vmax=None,                     # maximum value for color normalization
            contour_cmap=None,             # contour colormap
            contour_norm=None,             # contour normalization
            contour_levels=None,           # contour levels
            contour_filled=False,          # countour is filled
            contour_colors=None,           # contour colors
            contour_alpha=None,            # contour transparency
            surveys=None,                  # plot survey outlines
            lines=None,                    # plot lines over map
            points=None,                   # plot points over map
            stars=None,                    # plot stars over map
            asterisms=None,                # plot asterisms over map
            cbar=True,                     # display the colorbar
            cbar_orientation=None,         # colorbar orientation
            cbar_ticks=None,               # colorbar ticks
            cbar_format=None,              # colorbar number format
            cbar_unit=None,                # colorbar unit
            cbar_pad=None,                 # colorbar padding
            boundaries_level=None,         # HEALpix level for boundaries
            boundaries_pixs=None,          # HEALpix pixels for boundaries
            title=None,                    # plot title
            tissot=False,                  # display Tissot's indicatrices
            milkyway=False,                # display outline of Milky Way
            milkyway_width=None,           # number of degrees north/south to draw dotted line
            ecliptic=False,                # display outline of the ecliptic
            ecliptic_width=None,           # number of degrees north/south to draw dotted line
            colors=None,
            alphas=None,
            fontsize=None,
            hide_title=False,
            hide_cbar=False,
            return_fig = False
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

    if contours is not None:
        if isinstance(contours, bool):
            if map_data is not None:
                contour_values = map_data.values

                if contour_norm is None:
                    contour_norm = map_data.norm
        else:
            contour_values = contours.values

            if contour_norm is None:
                contour_norm = contours.norm
    else:
        contour_values = None

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
        includes_poles = skycoords is not None and np.any(np.abs(skycoords.dec.degree) > 89)
        if zoom and not includes_poles:
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

    if isinstance(norm, str) and 'log' in norm and values is not None and np.any(values <= 0):
        values = values.astype(float, copy=True)
        values[values <= 0.0] = np.nan

    if contour_values is not None and isinstance(contour_norm, str) and 'log' in contour_norm and contour_values is not None and np.any(contour_values <= 0):
        contour_values = contour_values.astype(float, copy=True)
        contour_values[contour_values <= 0.0] = np.nan

    if surveys is not None:
        for i, survey in enumerate(surveys):
            if isinstance(survey, str):
                name_or_filename = survey
            elif isinstance(survey, list):
                name_or_filename = survey[0]
            else:
                continue

            if not os.path.isfile(name_or_filename):
                survey_file = f'../data/surveys/{name_or_filename}-poly.txt'
                if os.path.isfile(survey_file):
                    if isinstance(survey, str):
                        surveys[i] = survey_file
                    else:
                        survey[0] = survey_file
                else:
                    raise AOMapException('Unrecognized survey')

    fig = healpix.plot(values, level=level, pixs=pixs, skycoords=skycoords, contour_values=contour_values, plot_properties={
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
        'contour_cmap': contour_cmap,
        'contour_norm': contour_norm,
        'contour_levels': contour_levels,
        'contour_filled': contour_filled,
        'contour_colors': contour_colors,
        'contour_alpha': contour_alpha,
        'surveys': surveys,
        'lines': lines,
        'points': points,
        'stars': stars,
        'asterisms': asterisms,
        'grid': grid,
        'grid_longitude': grid_longitude,
        'cbar': cbar,
        'cbar_orientation': cbar_orientation,
        'cbar_ticks': cbar_ticks,
        'cbar_format': cbar_format,
        'cbar_unit': cbar_unit,
        'cbar_pad': cbar_pad,
        'boundaries_level': boundaries_level,
        'boundaries_pixs': boundaries_pixs,
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'tissot': tissot,
        'milkyway': milkyway,
        'milkyway_width': milkyway_width,
        'ecliptic': ecliptic,
        'ecliptic_width': ecliptic_width,
        'colors': colors,
        'alphas': alphas,
        'fontsize': fontsize,
        'hide_title': hide_title,
        'hide_cbar': hide_cbar
    })

    if return_fig:
        return fig

    plt.show()

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

def plot_inner_pixel_asterism(config_or_filename,
                              outer_pix,
                              inner_pix,
                              asterism_id,
                              ao_system_name,
                              field_radius=60.0,
                              rotation_angle=0.0,
                              title=None,
                              filename=None,
                              return_fig=False):
    config = read_config(config_or_filename)
    ao_system = get_ao_system(config, ao_system_name)
    if ao_system is None:
        raise AOMapException(f"AO system not found: {ao_system_name}")

    context = _get_inner_pixel_asterism_context(config, outer_pix, ao_system)
    if context.asterisms is None or len(context.asterisms) == 0:
        raise AOMapException(f"No asterisms found for outer pixel {outer_pix}")

    inner_pixs = healpix.get_subpixels(config.outer_level, outer_pix, config.inner_level)
    row_matches = np.where(inner_pixs == inner_pix)[0]
    if len(row_matches) == 0:
        raise AOMapException(f"Inner pixel {inner_pix} not found in outer pixel {outer_pix}")
    row_idx = row_matches[0]

    context_ids = np.array(context.asterisms['id'], dtype=np.int_)
    local_mask = np.array(context.local_asterism_mask, dtype=bool)
    asterism_matches = np.where((context_ids == asterism_id) & local_mask)[0]
    if len(asterism_matches) == 0:
        raise AOMapException(f"Asterism {asterism_id} not found as a local asterism in outer pixel {outer_pix}")
    asterism_idx = asterism_matches[0]

    pixel_x = float(context.inner_x[row_idx])
    pixel_y = float(context.inner_y[row_idx])
    rotation_angle_rad = np.deg2rad(rotation_angle)

    def rotate_offsets(x, y):
        return (
            x * np.cos(rotation_angle_rad) - y * np.sin(rotation_angle_rad),
            x * np.sin(rotation_angle_rad) + y * np.cos(rotation_angle_rad)
        )

    asterism_x, asterism_y = rotate_offsets(
        float(context.asterism_x[asterism_idx] - pixel_x),
        float(context.asterism_y[asterism_idx] - pixel_y)
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.add_patch(plt.Circle((0, 0), field_radius, fill=False, lw=1.6, color='0.35'))
    ax.scatter([0], [0], s=70, marker='+', color='black', label='Inner pixel center')
    ax.scatter([asterism_x], [asterism_y], s=55, marker='x', color='tab:red', label='Asterism center')

    lgs = _get_ao_lgs_xy(ao_system['lgs'])
    for star in lgs:
        ax.scatter(
            [star['x']],
            [star['y']],
            s=120,
            marker='s',
            color='gold',
            edgecolor='black',
            zorder=2
        )

    star_points = []
    valid_star_count = 0
    num_stars = int(context.asterisms['num_stars'][asterism_idx])
    star_mags = np.array([float(context.asterisms[f'star{star_idx}_mag'][asterism_idx]) for star_idx in range(1, num_stars + 1)])
    min_mag = np.min(star_mags)
    max_mag = np.max(star_mags)
    for star_idx in range(1, num_stars + 1):
        star_x, star_y = rotate_offsets(
            float(context.star_x[star_idx][asterism_idx] - pixel_x),
            float(context.star_y[star_idx][asterism_idx] - pixel_y)
        )
        star_mag = float(context.asterisms[f'star{star_idx}_mag'][asterism_idx])
        star_radius = float(np.hypot(star_x, star_y))
        is_valid = star_radius <= field_radius
        if is_valid:
            valid_star_count += 1
        star_points.append((star_x, star_y))
        alpha = 1.0 if is_valid else 0.35
        if max_mag > min_mag:
            mag_scale = 1.0 - (star_mag - min_mag) / (max_mag - min_mag)
        else:
            mag_scale = 0.5
        red_level = 0.25 + 0.65 * mag_scale
        star_color = (1.0, 1.0 - red_level, 1.0 - red_level)
        ax.scatter([star_x], [star_y], s=120, color=star_color, edgecolor='black', zorder=3, alpha=alpha)
        ax.text(star_x + 2.0, star_y + 2.0, f'S{star_idx}  m={star_mag:.1f}', fontsize=9, alpha=1.0 if is_valid else 0.5)
        ax.plot([0, star_x], [0, star_y], color=star_color, alpha=0.25 if is_valid else 0.12, lw=1.0)

    ax.set_aspect('equal')
    ax.set_xlabel('x offset from inner pixel center (arcsec)')
    ax.set_ylabel('y offset from inner pixel center (arcsec)')
    ax.set_title(title or f'Inner pixel {inner_pix} with asterism {asterism_id} ({valid_star_count}/{num_stars} in FOV, rot={rotation_angle:.1f} deg)')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right')

    margin = max(
        field_radius,
        np.max(np.abs(np.array(star_points + [(asterism_x, asterism_y)])))
    ) + 10
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    fig.tight_layout()

    if filename is not None:
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(filename, dpi=170)

    if return_fig:
        return fig

    if filename is None:
        plt.show()
    else:
        plt.close(fig)

def plot_asterism_trend_diagnostics(config_or_filename,
                                    outer_pix,
                                    ao_system_name,
                                    star_counts=(2, 3),
                                    global_ranges=True,
                                    filename_prefix=None,
                                    return_figs=False):
    config = read_config(config_or_filename)
    ao_system = get_ao_system(config, ao_system_name)
    if ao_system is None:
        raise AOMapException(f"AO system not found: {ao_system_name}")

    inner_filename = _get_inner_pixel_data_filename(config, outer_pix)
    asterism_filename = _get_asterisms_data_filename(config, outer_pix, ao_system_name)
    if not os.path.isfile(inner_filename):
        raise AOMapException(f"Inner-pixel file not found: {inner_filename}")
    if not os.path.isfile(asterism_filename):
        raise AOMapException(f"Asterism file not found: {asterism_filename}")

    with fits.open(inner_filename) as hdul:
        inner = hdul[1].data
    with fits.open(asterism_filename) as hdul:
        asterisms = hdul[1].data

    asterism_num_stars = np.array(asterisms['num_stars'], dtype=np.int_)
    best_ee = np.array(asterisms['best_ee'], dtype=np.float64)
    mean_mag = np.vstack([
        np.array(asterisms['star1_mag'], dtype=np.float64),
        np.array(asterisms['star2_mag'], dtype=np.float64),
        np.array(asterisms['star3_mag'], dtype=np.float64),
    ])
    mean_mag[mean_mag <= 0] = np.nan
    mean_mag = np.nanmean(mean_mag, axis=0)

    centre_ra = np.array(asterisms['ra'], dtype=np.float64)
    centre_dec = np.array(asterisms['dec'], dtype=np.float64)
    star_ras = [
        np.array(asterisms['star1_ra'], dtype=np.float64),
        np.array(asterisms['star2_ra'], dtype=np.float64),
        np.array(asterisms['star3_ra'], dtype=np.float64),
    ]
    star_decs = [
        np.array(asterisms['star1_dec'], dtype=np.float64),
        np.array(asterisms['star2_dec'], dtype=np.float64),
        np.array(asterisms['star3_dec'], dtype=np.float64),
    ]
    star_mags = [
        np.array(asterisms['star1_mag'], dtype=np.float64),
        np.array(asterisms['star2_mag'], dtype=np.float64),
        np.array(asterisms['star3_mag'], dtype=np.float64),
    ]

    min_dist_catalog = np.full(len(asterisms), np.nan, dtype=np.float64)
    for idx in range(len(asterisms)):
        dists = []
        for star_ra, star_dec, star_mag in zip(star_ras, star_decs, star_mags):
            if star_mag[idx] <= 0:
                continue
            dx = (star_ra[idx] - centre_ra[idx]) * np.cos(np.deg2rad(centre_dec[idx])) * 3600.0
            dy = (star_dec[idx] - centre_dec[idx]) * 3600.0
            dists.append(np.hypot(dx, dy))
        if len(dists) > 0:
            min_dist_catalog[idx] = np.min(dists)

    value_field = _get_asterism_ee_max_field_name(ao_system_name)
    id_field = _get_asterism_ee_max_id_field_name(ao_system_name)
    if value_field not in inner.names or id_field not in inner.names:
        raise AOMapException(f"Required inner columns not found: {value_field}, {id_field}")

    winner_ids = np.array(inner[id_field], dtype=np.float64)
    winner_ee = np.array(inner[value_field], dtype=np.float64)
    local_mask = np.isfinite(winner_ids) & np.isfinite(winner_ee)
    winner_ids = winner_ids[local_mask].astype(np.int_)
    winner_ee = winner_ee[local_mask]
    winner_pixs = np.array(inner[FITS_COLUMN_PIX], dtype=np.int64)[local_mask]

    inner_coords = healpix.get_pixel_skycoord(config.inner_level, pixs=winner_pixs)
    inner_ra = np.array(inner_coords.ra.degree, dtype=np.float64)
    inner_dec = np.array(inner_coords.dec.degree, dtype=np.float64)

    asterism_ids = np.array(asterisms['id'], dtype=np.int_)
    asterism_id_to_idx = {int(asterism_id): idx for idx, asterism_id in enumerate(asterism_ids)}
    winner_idxs = np.array([asterism_id_to_idx[int(asterism_id)] for asterism_id in winner_ids], dtype=np.int_)
    winner_num_stars = asterism_num_stars[winner_idxs]
    winner_mean_mag = mean_mag[winner_idxs]

    field_radius = ao_system['fov'].to(u.arcsec).value / 2.0
    winner_min_dist = np.full(len(winner_idxs), np.nan, dtype=np.float64)
    for row_idx, asterism_idx in enumerate(winner_idxs):
        dists = []
        for star_ra, star_dec, star_mag in zip(star_ras, star_decs, star_mags):
            if star_mag[asterism_idx] <= 0:
                continue
            dx = (star_ra[asterism_idx] - inner_ra[row_idx]) * np.cos(np.deg2rad(inner_dec[row_idx])) * 3600.0
            dy = (star_dec[asterism_idx] - inner_dec[row_idx]) * 3600.0
            dist = np.hypot(dx, dy)
            if dist <= field_radius + 1e-6:
                dists.append(dist)
        if len(dists) > 0:
            winner_min_dist[row_idx] = np.min(dists)

    def _corrcoef(xvals, yvals):
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        if np.count_nonzero(mask) < 2:
            return np.nan
        return float(np.corrcoef(xvals[mask], yvals[mask])[0, 1])

    all_counts_mask_catalog = np.isin(asterism_num_stars, star_counts)
    all_counts_mask_winner = np.isin(winner_num_stars, star_counts)
    catalog_valid_mask = all_counts_mask_catalog & np.isfinite(best_ee) & np.isfinite(mean_mag) & np.isfinite(min_dist_catalog)
    winner_valid_mask = all_counts_mask_winner & np.isfinite(winner_ee) & np.isfinite(winner_mean_mag) & np.isfinite(winner_min_dist)
    if np.count_nonzero(catalog_valid_mask) == 0 and np.count_nonzero(winner_valid_mask) == 0:
        raise AOMapException(f"No valid diagnostic points found for {ao_system_name}")

    def _limits(values):
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            return (0.0, 1.0)
        return (float(np.nanmin(finite_values)), float(np.nanmax(finite_values)))

    if global_ranges:
        xmag_min, xmag_max = _limits(np.concatenate([mean_mag[catalog_valid_mask], winner_mean_mag[winner_valid_mask]]))
        xdist_min, xdist_max = _limits(np.concatenate([min_dist_catalog[catalog_valid_mask], winner_min_dist[winner_valid_mask]]))
        ee_min, ee_max = _limits(np.concatenate([best_ee[catalog_valid_mask], winner_ee[winner_valid_mask]]))
    else:
        xmag_min = xmag_max = xdist_min = xdist_max = ee_min = ee_max = None

    mag_pad = 0.05 * (xmag_max - xmag_min if global_ranges and xmag_max > xmag_min else 1.0)
    dist_pad = 0.05 * (xdist_max - xdist_min if global_ranges and xdist_max > xdist_min else 1.0)
    ee_pad = 0.05 * (ee_max - ee_min if global_ranges and ee_max > ee_min else 1.0)

    diagnostics = {}
    figures = {}
    for star_count in star_counts:
        catalog_mask = (asterism_num_stars == star_count) & np.isfinite(best_ee) & np.isfinite(mean_mag) & np.isfinite(min_dist_catalog)
        winner_mask = (winner_num_stars == star_count) & np.isfinite(winner_ee) & np.isfinite(winner_mean_mag) & np.isfinite(winner_min_dist)

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharey=True, constrained_layout=True)
        fig.suptitle(f'Outer Pixel {outer_pix}: {star_count}-Star Asterisms', fontsize=14)

        panels = [
            (0, 0, mean_mag[catalog_mask], best_ee[catalog_mask], 'Catalog Mean Model', 'Mean NGS Magnitude'),
            (0, 1, winner_mean_mag[winner_mask], winner_ee[winner_mask], 'Resolved Inner-Pixel Winners', 'Mean NGS Magnitude'),
            (1, 0, min_dist_catalog[catalog_mask], best_ee[catalog_mask], 'Catalog Mean Model', 'Min NGS Distance (arcsec)'),
            (1, 1, winner_min_dist[winner_mask], winner_ee[winner_mask], 'Resolved Inner-Pixel Winners', 'Min In-FOV NGS Distance (arcsec)'),
        ]

        for row_idx, col_idx, xvals, yvals, title, xlabel in panels:
            ax = axes[row_idx, col_idx]
            ax.scatter(xvals, yvals, s=10, alpha=0.55, color='#b22222', edgecolors='none')
            ax.set_title(f"{title}\nN={len(xvals)}, corr={_corrcoef(xvals, yvals):.3f}")
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.25)

        axes[0, 0].set_ylabel('EE')
        axes[1, 0].set_ylabel('EE')

        if global_ranges:
            axes[0, 0].set_xlim(xmag_min - mag_pad, xmag_max + mag_pad)
            axes[0, 1].set_xlim(xmag_min - mag_pad, xmag_max + mag_pad)
            axes[1, 0].set_xlim(xdist_min - dist_pad, xdist_max + dist_pad)
            axes[1, 1].set_xlim(xdist_min - dist_pad, xdist_max + dist_pad)
            for ax in axes.flat:
                ax.set_ylim(ee_min - ee_pad, ee_max + ee_pad)

        diagnostics[star_count] = {
            'catalog_n': int(np.count_nonzero(catalog_mask)),
            'winner_n': int(np.count_nonzero(winner_mask)),
            'catalog_mag_corr': _corrcoef(mean_mag[catalog_mask], best_ee[catalog_mask]),
            'winner_mag_corr': _corrcoef(winner_mean_mag[winner_mask], winner_ee[winner_mask]),
            'catalog_dist_corr': _corrcoef(min_dist_catalog[catalog_mask], best_ee[catalog_mask]),
            'winner_dist_corr': _corrcoef(winner_min_dist[winner_mask], winner_ee[winner_mask]),
        }

        if filename_prefix is not None:
            filename = f"{filename_prefix}-{star_count}star.png"
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fig.savefig(filename, dpi=180)
            diagnostics[star_count]['filename'] = filename

        if return_figs:
            figures[star_count] = fig
        elif filename_prefix is None:
            plt.show()
        else:
            plt.close(fig)

    if return_figs:
        return diagnostics, figures
    return diagnostics

#endregion
