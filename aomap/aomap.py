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
import warnings
import yaml
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import healpy
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
FITS_COLUMN_STAR_COUNT = 'STAR_COUNT'

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

    start_time = time.time()
    last_time = start_time

    for i in range(num_chunks):
        outer_start_pix = i * chunk_size
        outer_end_pix = min((i+1)*chunk_size, npix)

        # NOTE: parallelizing the following isn't effective as it is disk IO limited
        for outer_pix in range(outer_start_pix, outer_end_pix):
            _set_data_pixel_values(config, levels, level_data, outer_pix)

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
    cols.append(fits.Column(name=FITS_COLUMN_STAR_COUNT, format='K', array=np.zeros((npix), dtype=np.int_), unit='stars'))

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

def _set_data_pixel_values(config, levels, level_data, outer_pix):
    inner_filename = _get_inner_pixel_data_filename(config, outer_pix)
    if not os.path.isfile(inner_filename) or not _is_good_FITS(inner_filename):
        raise AOMapException(f"Inner data not found for outer pixel {outer_pix}")

    inner_data = _load_inner(config, outer_pix)
    aggregate_data = _get_initial_aggregate_pixel_values(config, inner_data)
    inner_data.close()

    level_done = np.zeros((len(levels)), dtype=bool)
    for level in range(config.inner_level, 0, -1):
        if level in levels:
            level_index = levels.index(level)

            level_data[level_index][1].data[FITS_COLUMN_MODEL_DENSITY][aggregate_data.pix] = aggregate_data.mean_model_density
            level_data[level_index][1].data[FITS_COLUMN_STAR_COUNT][aggregate_data.pix] = aggregate_data.sum_star_count

            level_done[level_index] = True

            if np.all(level_done):
                break

        _aggregate_pixel_values_decrease_level(config, aggregate_data)

def _get_initial_aggregate_pixel_values(config, inner_data):
    aggregate_data = StructType()
    aggregate_data.pix = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_PIX)
    aggregate_data.mean_model_density = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_MODEL_DENSITY)
    aggregate_data.sum_star_count = _get_inner_pixel_data_column(inner_data, FITS_COLUMN_STAR_COUNT)
    return aggregate_data

def _aggregate_pixel_values_decrease_level(config, aggregate_data):
    aggregate_data.pix = aggregate_data.pix[::4] // 4
    aggregate_data.mean_model_density = aggregate_data.mean_model_density.reshape(-1, 4).mean(axis=1)
    aggregate_data.sum_star_count = aggregate_data.sum_star_count.reshape(-1, 4).sum(axis=1)

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
            # Exclusions that DO require inner_data go here:
            # TODO: excluded = ...

            inner_data.close()

    return np.array([True, excluded])

def _get_inner_pixel_data_column(inner_data, column_name):
    return inner_data[1].data[column_name]

def _create_inner(config, outer_pix, num_retries=3, force_reload_gaia=False):
    # Compute Galaxy Density Model
    pixs, coords = healpix.get_subpixels_skycoord(config.outer_level, outer_pix, config.inner_level)
    galaxy_model = _compute_galaxy_model(coords.galactic)

    # Get Stars from Gaia
    gaia_stars = _get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=num_retries, force_reload=force_reload_gaia)

    # Determine Inner Pixel of Gaia Stars
    gaia_pixs = healpix.get_healpix(config.inner_level, SkyCoord(ra=gaia_stars['gaia_ra'], dec=gaia_stars['gaia_dec'], unit=(u.degree, u.degree)))

    # Count Gaia Stars per Inner Pixel
    unique, counts = np.unique(gaia_pixs, return_counts=True)
    count_map = dict(zip(unique, counts))
    star_count = np.array([count_map.get(p, 0) for p in pixs])

    # Build FITS table
    cols = []
    cols.append(fits.Column(name=FITS_COLUMN_PIX, format='K', array=pixs))
    cols.append(fits.Column(name=FITS_COLUMN_MODEL_DENSITY, format='D', array=galaxy_model, unit='density'))
    cols.append(fits.Column(name=FITS_COLUMN_STAR_COUNT, format='K', array=star_count, unit='stars'))

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

def _get_map_values(config, level, key):
    field = key.replace('-','_').upper()
    if 'DENSITY' in field and key != 'model-density':
        field = field.replace('DENSITY', 'COUNT')
        factor = 1/healpix.get_area(level).to(u.arcmin**2).value
    else:
        factor = 1.0

    data = _load_data(config, level)
    values = factor * data[1].data[field] # pylint: disable=no-member
    FITS_format = data[1].columns[field].format # pylint: disable=no-member
    unit = data[1].columns[field].unit # pylint: disable=no-member
    data.close()

    return values, FITS_format, unit

def get_map_data(config, level, key, pixs=None, coords=()):
    values, FITS_format, unit = _get_map_values(config, level, key)

    if pixs is not None:
        values = values[pixs] # pylint: disable=no-member
    else:
        pixs = np.arange(healpix.get_npix(level))

    skycoords = healpix.get_pixel_skycoord(level, pixs)

    if len(coords) > 0:
        row_filter = healpix.filter_skycoords(skycoords, coords)
        pixs = pixs[row_filter]
        values = values[row_filter]
        skycoords = skycoords[row_filter]

    if 'count' in key or key.endswith('pix'):
        num_format = '%.0f'
    else:
        num_format = '%.1f'

    norm = 'log'

    match key:
        case 'model-density':
            title = 'Galaxy Model'
        case 'star-count':
            title = 'Star Count'
        case 'star-density':
            title = 'Stellar Density'
        case _:
            title = f"{key} Map"

    map_data = StructType()
    map_data.key = key
    map_data.level = level
    map_data.pixs = pixs
    map_data.values = values
    map_data.coords = skycoords
    map_data.FITS_format = FITS_format
    map_data.unit = unit
    map_data.num_format = num_format
    map_data.norm = norm
    map_data.title = title
    return map_data

def get_map_table(config, level, key, pixs=None, coords=()):
    map_data = get_map_data(config, level, key, pixs, coords)

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

def plot_map(config, map_data,
            projection='default',        # mollweide, cart, aitoff, lambert, hammer, 3d, polar
            coordsys='default',          # celstial, galactic, ecliptic
            direction='default',         # longitude direction ('astro' = east towards left, 'geo' = east towards right)
            rotation=None,               # longitudinal rotation angle in degrees
            grid=True,                   # display grid lines
            longitude_grid_spacing=30,   # set x axis grid spacing in degrees
            latitude_grid_spacing=30,    # set y axis grid spacing in degrees
            cbar=True,                   # display the colorbar
            cmap='viridis',              # specify the colormap
            norm='default',              # color normalization
            vmin=None,                   # minimum value for color normalization
            vmax=None,                   # maximum value for color normalization
            unit=None,                   # text describing the data unit
            num_format='%.1f',          # number format of data unit
            hours=True,                  # display RA in hours
            title=None,                  # set title of the plot
            xsize=800,                   # size of the image
            width=None                   # figure width
    ):

    if projection == 'default':
        if hasattr(config, 'plot_projection'):
            projection = config.plot_projection
        else:
            projection = 'mollweide'

    if coordsys == 'default':
        if hasattr(config, 'plot_coordsys'):
            coordsys = config.plot_coordsys
        else:
            coordsys = 'C'

    if direction == 'default':
        if hasattr(config, 'plot_direction'):
            direction = config.plot_direction
        else:
            direction = 'astro'

    if rotation is None:
        if hasattr(config, 'plot_rotation'):
            rotation = config.plot_rotation
        else:
            rotation = 0.0

    match projection:
        case 'cart' | 'cartesian':
            projection = 'cart'
            xtick_label_color = 'black'
            ra_offset = -longitude_grid_spacing
        case _:
            xtick_label_color = (0.9,0.9,0.9)
            ra_offset = longitude_grid_spacing

    match coordsys:
        case 'celestial' | 'cel' | 'c' | 'C':
            coord = 'C'
            xlabel = 'RA'
            ylabel = 'DEC'
        case 'galactic' | 'gal' | 'g' | 'G':
            coord = 'G'
            xlabel = 'GLON'
            ylabel = 'GLAT'
            hours = False
        case 'ecliptic' | 'ecl' | 'e' | 'E':
            coord = 'E'
            xlabel = 'ELON'
            ylabel = 'ELAT'
            hours = False

    if norm == 'default':
        norm = map_data.norm

    if title is None:
        title = map_data.title

    if unit is None:
        unit = map_data.unit

    if num_format is None:
        num_format = map_data.num_format

    cb_orientation = 'vertical' if projection == 'cart' else 'horizontal'

    values = map_data.values.copy()
    if 'log' in norm:
        values[values <= 0] = np.nan

    healpy.projview(
        values,
        nest=True,
        coord=['C', coord],
        flip=direction,
        projection_type=projection,
        graticule=grid,
        graticule_labels=grid,
        phi_convention='symmetrical',
        rot=rotation,
        longitude_grid_spacing=longitude_grid_spacing,
        latitude_grid_spacing=latitude_grid_spacing,
        xlabel=xlabel,
        ylabel=ylabel,
        unit=unit,
        format=num_format,
        title=title,
        xsize=xsize,
        width=width,
        cbar=cbar,
        cmap=cmap,
        norm=norm,
        min=vmin,
        max=vmax,
        cb_orientation=cb_orientation,
        xtick_label_color=xtick_label_color
    )

    ax = plt.gca()

    ax.set_axisbelow(False) # hack to show grid lines on top of the image

    if hours:
        if rotation % longitude_grid_spacing != 0.0:
            raise AOMapException(f"rot {rotation} must be a multiple of {longitude_grid_spacing}")

        tick_hours = np.linspace(-12+ra_offset/15+rotation/15, 12-ra_offset/15+rotation/15, len(ax.xaxis.majorTicks))
        tick_hours[tick_hours < 0] += 24

        if direction == 'astro':
            tick_hours = np.flip(tick_hours)

        warnings.simplefilter('ignore', UserWarning)
        ax.set_xticklabels([f"{t:.0f}h" for t in tick_hours])
        warnings.resetwarnings()

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
