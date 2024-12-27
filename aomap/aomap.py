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

ALLOWED_MAPS = [
    'galaxy-model',
    'star-count',
    'stellar-density'
]

FITS_COLUMN_MAP_VALUE = 'VALUE'
FITS_COLUMN_MAP_EXCLUDED = 'EXCLUDED'
FITS_COLUMN_PIX = 'PIX'
FITS_COLUMN_MODEL_DENSITY = 'MODEL_DENSITY'
FITS_COLUMN_STAR_COUNT = 'STAR_COUNT'

#endregion

class Maps:
    def __init__(self, config, level, update=False):
        self.config = config

        if level < self.config.outer_level:
            raise AOMapException(f"Level must be greater than or equal to {config.outer_level}")
        if level > self.config.max_map_level:
            raise AOMapException(f"Level must be less than or equal to {config.max_map_level}")

        self.level = level
        self._maps = {}
        self._load_maps(level, update=update)

    @staticmethod
    def load(config_or_filename, level=None):
        config = Maps._read_config(config_or_filename)

        if level is None:
            level = config.outer_level

        return Maps(config, level=level)

#region Config

    @staticmethod
    def _read_config(config_or_filename):
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

        if not hasattr(config, 'max_map_level'):
            config.max_map_level = config.outer_level
            if config.max_map_level < config.outer_level:
                raise AOMapException('max_map_level must be greater than or equal to outer_level')
            if config.max_map_level > config.inner_level:
                raise AOMapException('max_map_level must be less than or equal to inner_level')

        if not hasattr(config, 'maps'):
            config.maps = ['star-count']

        for key in config.maps:
            if key not in ALLOWED_MAPS:
                raise AOMapException(f"Map '{key}' is not supported")

        if not hasattr(config, 'exclude_min_galactic_latitude'):
            config.exclude_min_galactic_latitude = 0.0

        return config

#endregion

#region Dict-like

    def __getitem__(self, key):
        return self._maps[key]

    def __setitem__(self, key, value):
        self._maps[key] = value

    def __delitem__(self, key):
        del self._maps[key]

    def __len__(self):
        return len(self._maps)

    def __iter__(self):
        return iter(self._maps)

    def __contains__(self, key):
        return key in self._maps

    def keys(self):
        return self._maps.keys()

    def values(self):
        return self._maps.values()

    def items(self):
        return self._maps.items()

    def __repr__(self):
        return repr(self._maps)

#endregion

#region Building

    @staticmethod
    def build(config_or_filename, mode='recalc', pixs=None, force_reload_gaia=False, max_pixels=None, verbose=False):
        config = Maps._read_config(config_or_filename)

        force_create = mode.startswith('re')
        Maps._create_maps(config, config.outer_level, add_excluded=True, force_create=force_create, verbose=verbose)
        maps = Maps(config, config.outer_level, update=True)

        npix = healpix.get_npix(config.outer_level)

        if config.chunk_multiple == 0:
            chunk_size = npix
        else:
            chunk_size = (config.cores if config.cores >= 1 else os.cpu_count()) * config.chunk_multiple

        todo = np.zeros((npix), dtype=bool)
        if pixs is not None:
            todo[pixs] = True
        else:
            for k in maps:
                todo = todo | (np.isnan(maps.get_map_values(k)) & ~maps.get_map_excluded(k))

        if np.all(todo):
            todo_pix = np.arange(npix)
        else:
            todo_pix = np.where(todo)[0]

        if len(todo_pix) == 0:
            if verbose:
                print(f"Building maps for level {config.outer_level} already done")
            return False

        if max_pixels is not None and max_pixels < len(todo_pix):
            todo_pix = todo_pix[:max_pixels]

        num_todo = len(todo_pix)
        chunk_size = min(chunk_size, num_todo)
        num_chunks = int(np.ceil(num_todo / chunk_size))

        if verbose:
            print(f"Building maps for level {config.outer_level}:")
            if todo_pix[0] > 0:
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"\r  {current_time}: {todo_pix[0]+1}/{npix} (start)", end='', flush=True)

            start_time = time.time()
            last_time = start_time

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i+1)*chunk_size, num_todo)

            if config.cores == 1:
                excluded_buffer = np.zeros((end_idx-start_idx), dtype=bool)
                values_buffer = np.full((end_idx-start_idx, len(maps)), np.nan)

                j = 0
                for outer_pix in todo_pix[start_idx:end_idx]:
                    results = Maps._get_outer_pixel_excluded_and_values(config, mode, outer_pix, force_reload_gaia)
                    excluded_buffer[j] = np.bool(results[0])
                    values_buffer[j,:] = results[1:]
                    j += 1
            else:
                results = np.array(Parallel(n_jobs=config.cores)(delayed(Maps._get_outer_pixel_excluded_and_values)(config, mode, outer_pix, force_reload_gaia) for outer_pix in todo_pix[start_idx:end_idx]))
                excluded_buffer = np.bool(results[:,0])
                values_buffer = results[:,1:]

            maps.update_map_data(todo_pix[start_idx:end_idx], values_buffer, excluded_buffer)
            maps.flush()

            if verbose:
                elapsed_time = time.time() - last_time
                last_time = time.time()
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"\r  {current_time}: {todo_pix[end_idx-1]+1}/{npix} ({end_idx-start_idx}px in {elapsed_time:.2f}s, {np.sum(excluded_buffer)} excluded)           ", end='', flush=True)

        if verbose:
            total_time = time.time() - start_time
            print(f"\n  done: {num_todo}px in {total_time:.1f}s")

        maps.close()

        return True

    @staticmethod
    def build_extra_map_levels(config_or_filename, verbose = False):
        config = Maps._read_config(config_or_filename)

        if config.max_map_level == config.outer_level:
            if verbose:
                print("Nothing to build")
            return

        levels = []
        level_maps = []
        for level in range(config.outer_level+1, config.max_map_level+1):
            Maps._create_maps(config, level, force_create=True, verbose=verbose)
            levels.append(level)
            level_maps.append(Maps(config, level, update=True))

        npix = healpix.get_npix(config.outer_level)

        if config.chunk_multiple == 0:
            chunk_size = npix
        else:
            chunk_size = (config.cores if config.cores >= 1 else os.cpu_count()) * config.chunk_multiple

        num_chunks = int(np.ceil(npix / chunk_size))

        if verbose:
            if len(levels) == 1:
                print(f"Building maps for level {levels}:")
            else:
                print(f"Building maps for levels {levels[0]}-{levels[-1]}:")

        start_time = time.time()
        last_time = start_time

        for i in range(num_chunks):
            outer_start_pix = i * chunk_size
            outer_end_pix = min((i+1)*chunk_size, npix)

            # NOTE: parallelizing the following isn't effective as it is disk IO limited
            for outer_pix in range(outer_start_pix, outer_end_pix):
                (start_pix, start_index, values_buffer) = Maps._set_outer_pixel_aggregated_pixel_values(config, levels, outer_pix)

                for j, level in enumerate(levels):
                    level_npix = healpix.get_subpixel_npix(config.outer_level, level)
                    level_maps[j].update_map_data(start_pix[j], values_buffer[start_index[j]:start_index[j]+level_npix])

            for maps in level_maps:
                maps.flush()

            if verbose:
                elapsed_time = time.time() - last_time
                last_time = time.time()
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"\r  {current_time}: {outer_end_pix}/{npix} ({outer_end_pix-outer_start_pix}px in {elapsed_time:.2f}s)          ", end='', flush=True)

        if verbose:
            total_time = time.time() - start_time
            print(f"\n  done: {npix}px in {total_time:.1f}s")

        for maps in level_maps:
            maps.close()

    def flush(self):
        for k in self:
            self._flush_map(k)

    def close(self):
        for k in self:
            self._close_map(k)

#endregion

#region Plotting

    def plot(self,
             key=None,                    # key of the map to plot (None=all maps)
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
             unit_format='%.1f',          # number format of data unit
             hours=True,                  # display RA in hours
             title=None,                  # set title of the plot
             xsize=800,                   # size of the image
             width=None                   # figure width
        ):

        if key is not None:
            self._plot_map(key,
                       projection,
                       coordsys,
                       direction,
                       rotation,
                       grid,
                       longitude_grid_spacing,
                       latitude_grid_spacing,
                       cbar,
                       cmap,
                       norm,
                       vmin,
                       vmax,
                       unit,
                       unit_format,
                       hours,
                       title,
                       xsize,
                       width
            )
        else:
            for k in self:
                self._plot_map(k,
                           projection,
                           coordsys,
                           direction,
                           rotation,
                           grid,
                           longitude_grid_spacing,
                           latitude_grid_spacing,
                           cbar,
                           cmap,
                           norm,
                           vmin,
                           vmax,
                           unit,
                           unit_format,
                           hours,
                           title,
                           xsize,
                           width
                )

    def _plot_map(self, key,
                  projection,
                  coordsys,
                  direction,
                  rotation,
                  grid,
                  longitude_grid_spacing,
                  latitude_grid_spacing,
                  cbar,
                  cmap,
                  norm,
                  vmin,
                  vmax,
                  unit,
                  unit_format,
                  hours,
                  title,
                  xsize,
                  width
        ):

        if projection == 'default':
            if hasattr(self.config, 'plot_projection'):
                projection = self.config.plot_projection
            else:
                projection = 'mollweide'

        if coordsys == 'default':
            if hasattr(self.config, 'plot_coordsys'):
                coordsys = self.config.plot_coordsys
            else:
                coordsys = 'C'

        if direction == 'default':
            if hasattr(self.config, 'plot_direction'):
                direction = self.config.plot_direction
            else:
                direction = 'astro'

        if rotation is None:
            if hasattr(self.config, 'plot_rotation'):
                rotation = self.config.plot_rotation
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
            norm = Maps._get_default_norm(key)

        if title is None:
            title = Maps._get_default_title(key)

        if unit is None:
            unit, unit_format = Maps._get_default_unit(key)

        cb_orientation = 'vertical' if projection == 'cart' else 'horizontal'

        values = self.get_map_values(key).copy()
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
            format=unit_format,
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

#endregion

#region Paths

    @staticmethod
    def _get_map_filename(config, key, level):
        return f"{config.folder}/{key}-hpx{level}.fits"

    @staticmethod
    def _get_outer_pixel_path(config, outer_pix):
        coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix)
        return f"{config.folder}/inner/hpx{config.outer_level}-{config.inner_level}/{int(coord.ra.degree/15)}h/{'+' if coord.dec.degree >= 0 else '-'}{int(np.abs(coord.dec.degree/10))*10:02}/{outer_pix}"

    @staticmethod
    def _get_gaia_path(config, outer_pix):
        coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix)
        return f"{config.folder}/gaia/hpx{config.outer_level}/{int(coord.ra.degree/15)}h/{'+' if coord.dec.degree >= 0 else '-'}{int(np.abs(coord.dec.degree/10))*10:02}/{outer_pix}"

    @staticmethod
    def _get_inner_pixel_data_filename(config, outer_pix):
        return f"{Maps._get_outer_pixel_path(config, outer_pix)}/inner.fits"

#endregion

#region Outer Pixel FITS

    @staticmethod
    def _create_maps(config, level, add_excluded=False, force_create=False, verbose=False):
        npix = healpix.get_npix(level)

        for key in config.maps:
            filename = Maps._get_map_filename(config, key, level)
            if force_create or not os.path.isfile(filename):
                if verbose:
                    print(f"Creating {filename}")

                os.makedirs(os.path.dirname(filename), exist_ok=True)

                values = np.full((npix), np.nan) # np.ma.masked_all((npix))
                excluded = np.zeros((npix), dtype=bool)

                cols = []

                col0 = fits.Column(name=FITS_COLUMN_MAP_VALUE, format='D', array=values, unit=Maps._get_default_unit(key)[0])
                cols.append(col0)

                if add_excluded:
                    col1 = fits.Column(name=FITS_COLUMN_MAP_EXCLUDED, format='L', array=excluded, unit=None)
                    cols.append(col1)

                hdu = fits.BinTableHDU.from_columns(cols)
                header = hdu.header
                header['PIXTYPE'] = 'HEALPIX'
                header['ORDERING'] = 'NESTED'
                header['COORDSYS'] = 'C'
                header['EXTNAME'] = f"{key} Map"
                header['NSIDE'] = 2**level
                header['FIRSTPIX'] = 0
                header['LASTPIX'] = npix - 1
                header['INDXSCHM'] = 'IMPLICIT'
                header['OBJECT'] = 'FULLSKY'
                hdu.writeto(filename, overwrite=force_create)

    def _load_maps(self, level, update=False):
        for key in self.config.maps:
            filename = Maps._get_map_filename(self.config, key, level)
            if not os.path.isfile(filename):
                raise AOMapException(f"Map file not found: {filename}")
            hdul = fits.open(filename, mode='update' if update else 'readonly')
            self[key] = hdul

    def get_map_values(self, key, pixs=None, coords=(), return_details=False):
        if pixs is not None:
            values = self[key][1].data[FITS_COLUMN_MAP_VALUE][pixs]

            if not return_details:
                return values

            coords = healpix.get_pixel_skycoord(self.config.outer_level, pixs)

            return Table([
                pixs,
                values,
                coords.ra.degree,
                coords.dec.degree
            ], names=[
                'Pix',
                'Value',
                'RA',
                'Dec'
            ])

        if len(coords) > 0:
            npix = healpix.get_npix(self.config.outer_level) # pylint: disable=no-member
            pixs = np.arange(npix)
            skycoords = healpix.get_pixel_skycoord(self.config.outer_level, pixs)
            row_filter = healpix.filter_skycoords(skycoords, coords)

            values = self[key][1].data[FITS_COLUMN_MAP_VALUE][row_filter]

            if not return_details:
                return values

            pixs = pixs[row_filter]
            skycoords = skycoords[row_filter]

            return Table([
                pixs,
                skycoords.ra.degree,
                skycoords.dec.degree,
                values,
            ], names=[
                'Pix',
                'RA',
                'Dec',
                'Value'
            ])

        return self[key][1].data[FITS_COLUMN_MAP_VALUE]

    def get_map_excluded(self, key):
        return self[key][1].data[FITS_COLUMN_MAP_EXCLUDED]

    def get_map_unit(self, key):
        return self[key][1].columns[FITS_COLUMN_MAP_VALUE].unit

    def update_map_data(self, start_pix_or_pixs, values, excluded=None):
        for j, k in enumerate(self):
            if start_pix_or_pixs is None:
                np.copyto(self.get_map_values(k), values[:,j])
                if excluded is not None:
                    np.copyto(self.get_map_excluded(k), excluded)
            else:
                if np.size(start_pix_or_pixs) > 1:
                    pixs = start_pix_or_pixs
                else:
                    pixs = np.arange(start_pix_or_pixs, start_pix_or_pixs+len(values))

                for i, pix in enumerate(pixs):
                    self.get_map_values(k)[pix] = values[i,j]
                    if excluded is not None:
                        self.get_map_excluded(k)[pix] = excluded[i]

    def _flush_map(self, key):
        with prevent_interruption():
            self[key].flush()

    def _close_map(self, key):
        self[key].close()

    @staticmethod
    def _is_good_FITS(filename):
        try:
            with fits.open(filename) as hdul:
                hdul.verify('exception')
        except fits.VerifyError:
            return False

        return True

#endregion

#region Pixel Values

    ############################################################################
    # The following are implemented as static methods for parallel processing. #
    ############################################################################

    @staticmethod
    def _get_map_info(key, level):
        match key:
            case 'galaxy-model':
                return ('mean_model_density', 1.0)
            case 'star-count':
                return ('sum_star_count', 1.0)
            case 'stellar-density':
                return ('sum_star_count', 1/healpix.get_area(level).to(u.arcmin**2).value)

        return False

    @staticmethod
    def _get_default_unit(key):
        match key:
            case 'galaxy-model':
                return 'density', '%.1f'
            case 'star-count':
                return 'stars', '%.0f'
            case 'stellar-density':
                return 'stars/arcmin^2', '%.1f'

        return ''

    @staticmethod
    def _get_default_norm(key):
        match key:
            case 'galaxy-model':
                return 'log'
            case 'star-count':
                return 'log'
            case 'stellar-density':
                return 'log'

        return None

    @staticmethod
    def _get_default_title(key):
        match key:
            case 'galaxy-model':
                return 'Galaxy Model'
            case 'star-count':
                return 'Star Count'
            case 'stellar-density':
                return 'Stellar Density'

        return f"{key} Map"

    @staticmethod
    def _get_outer_pixel_excluded_and_values(config, mode, outer_pix, force_reload_gaia):
        galactic_coord = healpix.get_pixel_skycoord(config.outer_level, outer_pix).galactic
        excluded = False

        # Exclusions that DO NOT require inner_data go here:
        if config.exclude_min_galactic_latitude > 0:
            excluded = np.abs(galactic_coord.b.degree) < config.exclude_min_galactic_latitude

        if not excluded:
            skip = False

            match mode:
                case 'build':
                    inner_filename = Maps._get_inner_pixel_data_filename(config, outer_pix)
                    use_existing = os.path.isfile(inner_filename) and Maps._is_good_FITS(inner_filename)
                case 'rebuild':
                    use_existing = False
                case 'recalc':
                    use_existing = True

            if use_existing:
                inner_data = Maps._load_inner_pixel_data(config, outer_pix)
            else:
                try:
                    inner_data = Maps._build_inner_pixel_data(config, outer_pix, num_retries=3, force_reload_gaia=force_reload_gaia)
                    Maps._save_inner_pixel_data(config, outer_pix, inner_data)
                except (ConnectionResetError, FileNotFoundError) as e:
                    skip = True
                    print(f"Error building inner data for {outer_pix}:\n{e}")
                    traceback.print_exc()

            if skip:
                # Leave this outer pixel to do in the future
                excluded_and_values = np.full((1+len(config.maps)), np.nan)
                excluded_and_values[0] = False
                return excluded_and_values

            model_density = Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_MODEL_DENSITY)
            star_count = Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_STAR_COUNT)

            aggregate_data = StructType()
            aggregate_data.mean_model_density = np.mean(model_density)
            aggregate_data.sum_star_count = np.sum(star_count)
            values = Maps._get_aggregated_pixel_values(config, config.outer_level, aggregate_data)

            if inner_data is not None:
                # Exclusions that DO require inner_data go here:
                # TODO: excluded = ...

                inner_data.close()

        excluded_and_values = np.full((1+len(config.maps)), np.nan)
        excluded_and_values[0] = excluded
        if not excluded:
            np.copyto(excluded_and_values[1:], values)

        return excluded_and_values

    @staticmethod
    def _get_aggregated_pixel_values(config, level, aggregate_data):
        values = None

        for i, k in enumerate(config.maps):
            (field, factor) = Maps._get_map_info(k, level)

            if values is None:
                num_rows = np.size(getattr(aggregate_data, field))
                values = np.full((num_rows, len(config.maps)), np.nan)

            values[:,i] = factor * getattr(aggregate_data, field)

        return values

    @staticmethod
    def _set_outer_pixel_aggregated_pixel_values(config, levels, outer_pix):
        inner_filename = Maps._get_inner_pixel_data_filename(config, outer_pix)
        if not os.path.isfile(inner_filename) or not Maps._is_good_FITS(inner_filename):
            raise AOMapException(f"Inner data not found for outer pixel {outer_pix}")

        inner_data = Maps._load_inner_pixel_data(config, outer_pix)

        pix = Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_PIX)[0]

        aggregate_data = StructType()
        aggregate_data.mean_model_density = Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_MODEL_DENSITY)
        aggregate_data.sum_star_count = Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_STAR_COUNT)

        start_pix = np.zeros((len(levels)), dtype=int)
        start_index = np.cumsum([0] + [healpix.get_subpixel_npix(config.outer_level, level) for level in levels[0:-1]])
        num_values = np.sum(healpix.get_subpixel_npix(config.outer_level, level) for level in levels)
        values_buffer = np.full((num_values, len(config.maps)), np.nan)

        for level in range(config.inner_level, config.outer_level, -1):
            if level <= config.max_map_level:
                aggregated_values = Maps._get_aggregated_pixel_values(config, level, aggregate_data)
                npix = healpix.get_subpixel_npix(config.outer_level, level)
                j = levels.index(level)
                start_pix[j] = pix
                values_buffer[start_index[j]:start_index[j]+npix] = aggregated_values

                if j == 0:
                    break

            pix = pix // 4
            aggregate_data.mean_model_density = aggregate_data.mean_model_density.reshape(-1, 4).mean(axis=1)
            aggregate_data.sum_star_count = aggregate_data.sum_star_count.reshape(-1, 4).sum(axis=1)

        return (start_pix, start_index, values_buffer)

    @staticmethod
    def _get_inner_pixel_data(inner_data):
        return inner_data[1]

    @staticmethod
    def _get_inner_pixel_data_column(inner_data, column_name):
        return Maps._get_inner_pixel_data(inner_data).data[column_name]

    @staticmethod
    def _build_inner_pixel_data(config, outer_pix, num_retries=3, force_reload_gaia=False):
        cols = []

        # Compute Galaxy Density Model
        pixs, coords = healpix.get_subpixels_skycoord(config.outer_level, outer_pix, config.inner_level)
        galaxy_model = Maps._compute_galaxy_model(coords.galactic)

        # Get Stars from Gaia
        gaia_stars = Maps._get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=num_retries, force_reload=force_reload_gaia)

        # Determine Inner Pixel of Gaia Stars
        gaia_pixs = healpix.get_healpix(config.inner_level, SkyCoord(ra=gaia_stars['gaia_ra'], dec=gaia_stars['gaia_dec'], unit=(u.degree, u.degree)))

        # Count Gaia Stars per Inner Pixel
        unique, counts = np.unique(gaia_pixs, return_counts=True)
        count_map = dict(zip(unique, counts))
        star_count = np.array([count_map.get(p, 0) for p in pixs])

        # Build FITS table
        cols.append(fits.Column(name=FITS_COLUMN_PIX, format='K', array=pixs))
        cols.append(fits.Column(name=FITS_COLUMN_MODEL_DENSITY, format='D', array=galaxy_model, unit='density'))
        cols.append(fits.Column(name=FITS_COLUMN_STAR_COUNT, format='K', array=star_count, unit='stars'))
        pixel_data_hdu = fits.BinTableHDU.from_columns(cols)
        return fits.HDUList([fits.PrimaryHDU(), pixel_data_hdu])

    @staticmethod
    def _save_inner_pixel_data(config, outer_pix, inner_data):
        filename = Maps._get_inner_pixel_data_filename(config, outer_pix)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with prevent_interruption():
            inner_data.writeto(filename, overwrite=True)

    @staticmethod
    def _load_inner_pixel_data(config, outer_pix):
        filename = Maps._get_inner_pixel_data_filename(config, outer_pix)
        if not os.path.isfile(filename):
            raise AOMapException(f"Inner data not found for outer pixel {outer_pix}")
        return fits.open(filename, mode='readonly')

    @staticmethod
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

#endregion

#region Gaia

    @staticmethod
    def _get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=1, force_reload=False):
        filename = f"{Maps._get_gaia_path(config, outer_pix)}/gaia.fits"

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
