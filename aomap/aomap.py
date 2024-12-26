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
FITS_COLUMN_GALAXY_MODEL = 'MODEL_DENSITY'
FITS_COLUMN_STAR_COUNT = 'STAR_COUNT'

#endregion

class Maps:
    def __init__(self, config, read_only=False, force_create=False, verbose=False):
        self.config = config
        self._check_config()
        self._add_config()

        self._maps = {}
        self._load_maps(read_only=read_only, force_create=force_create, verbose=verbose)

    @staticmethod
    def load(config_or_filename, mode='readonly', verbose=False):
        if isinstance(config_or_filename, str):
            config = Maps._read_config(config_or_filename)
        else:
            config = config_or_filename

        read_only = mode == 'readonly'
        force_create = not read_only and mode.startswith('re')
        return Maps(config, read_only = read_only, force_create=force_create, verbose=verbose)

#region Config

    @staticmethod
    def _read_config(filename):
        if not os.path.isfile(filename):
            raise AOMapException(f"Config file not found: {filename}")

        with open(filename, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        # convert dict to struct (personal preference)
        config = StructType()
        for key, value in config_data.items():
            setattr(config, key, value)

        return config

    def _check_config(self):
        if not hasattr(self.config, 'cores') or self.config.cores < -1 or self.config.cores == 0:
            self.config.cores = 1

        if not hasattr(self.config, 'chunk_multiple'):
            self.config.chunk_multiple = 0

        if not hasattr(self.config, 'outer_level'):
            raise AOMapException("outer_level not defined in config")

        if not hasattr(self.config, 'inner_level'):
            raise AOMapException("inner_level not defined in config")

        if not hasattr(self.config, 'maps'):
            self.config.maps = ['simple']

        for key in self.config.maps:
            if key not in ALLOWED_MAPS:
                raise AOMapException(f"Map '{key}' is not supported")

        if not hasattr(self.config, 'exclude_min_galactic_latitude'):
            self.config.exclude_min_galactic_latitude = 0.0

    def _add_config(self):
        self.config.outer_npix = healpix.get_npix(self.config.outer_level)
        self.config.inner_npix = healpix.get_npix(self.config.inner_level)

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

    def build(self, mode='build', pixs=None, max_pixels=None, verbose=False):
        num_maps = len(self)

        if self.config.chunk_multiple == 0:
            chunk_size = self.config.outer_npix
        else:
            chunk_size = (self.config.cores if self.config.cores >= 1 else os.cpu_count()) * self.config.chunk_multiple

        todo = np.zeros((self.config.outer_npix), dtype=bool)
        if pixs is not None:
            todo[pixs] = True
        else:
            for k in self:
                todo = todo | (np.isnan(self.get_map_values(k)) & ~self.get_map_excluded(k))

        if np.all(todo):
            todo_pix = np.arange(self.config.outer_npix)
        else:
            todo_pix = np.where(todo)[0]

        if len(todo_pix) == 0:
            if verbose:
                print('Building maps already done')
            return

        if max_pixels is not None and max_pixels < len(todo_pix):
            todo_pix = todo_pix[:max_pixels]

        num_todo = len(todo_pix)

        if chunk_size > num_todo:
            chunk_size = num_todo

        num_chunks = int(np.ceil(num_todo / chunk_size))

        if verbose:
            print('Building maps:')
            start_time = time.time()
            last_time = start_time

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i+1)*chunk_size, num_todo)

            if self.config.cores == 1:
                excluded = np.zeros((end_idx-start_idx), dtype=bool)
                values = np.full((end_idx-start_idx, num_maps), np.nan)

                j = 0
                for outer_pix in todo_pix[start_idx:end_idx]:
                    results = Maps._get_outer_pixel_values(self.config, mode, outer_pix)
                    excluded[j] = np.bool(results[0])
                    values[j,:] = results[1:]
                    j += 1
            else:
                results = np.array(Parallel(n_jobs=self.config.cores)(delayed(Maps._get_outer_pixel_values)(self.config, mode, outer_pix) for outer_pix in todo_pix[start_idx:end_idx]))
                excluded = np.bool(results[:,0])
                values = results[:,1:]

            self._update_map_data(values, excluded, todo_pix[start_idx:end_idx])

            for k in self:
                self._flush_map(k)

            if verbose:
                elapsed_time = time.time() - last_time
                last_time = time.time()
                print(f"\r  {todo_pix[end_idx-1]+1}/{self.config.outer_npix} ({end_idx-start_idx}px in {elapsed_time:.2f}s, {np.sum(excluded)} excluded)          ", end='', flush=True)

        if verbose:
            total_time = time.time() - start_time
            print(f"\n  done: {num_todo}px in {total_time:.1f}s")

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
            graticule=True,
            graticule_labels=True,
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
    def _get_map_filename(config, key):
        return f"{config.folder}/{key}-hpx{config.outer_level}.fits"

    @staticmethod
    def _get_outer_pixel_path(config, outer_pix):
        coord = healpix.get_skycoord(config.outer_level, outer_pix)
        return f"{config.folder}/inner/hpx{config.outer_level}-{config.inner_level}/{int(coord.ra.degree/15)}h/{'+' if coord.dec.degree >= 0 else '-'}{int(np.abs(coord.dec.degree/10))*10:02}/{outer_pix}"

    @staticmethod
    def _get_gaia_path(config, outer_pix):
        coord = healpix.get_skycoord(config.outer_level, outer_pix)
        return f"{config.folder}/gaia/hpx{config.outer_level}/{int(coord.ra.degree/15)}h/{'+' if coord.dec.degree >= 0 else '-'}{int(np.abs(coord.dec.degree/10))*10:02}/{outer_pix}"

    @staticmethod
    def _get_inner_pixel_data_filename(config, outer_pix):
        return f"{Maps._get_outer_pixel_path(config, outer_pix)}/inner.fits"

#endregion

#region Outer Pixel FITS

    def _load_maps(self, read_only=False, force_create=False, verbose=False):
        for key in self.config.maps:
            filename = Maps._get_map_filename(self.config, key)

            if force_create or not os.path.isfile(filename):
                if verbose:
                    print(f"Creating {filename}")

                os.makedirs(os.path.dirname(filename), exist_ok=True)

                values = np.full((self.config.outer_npix), np.nan) # np.ma.masked_all((maps.outer_npix))
                excluded = np.zeros((self.config.outer_npix), dtype=bool)

                col0 = fits.Column(name=FITS_COLUMN_MAP_VALUE, format='D', array=values, unit=Maps._get_default_unit(key)[0])
                col1 = fits.Column(name=FITS_COLUMN_MAP_EXCLUDED, format='L', array=excluded, unit=None)

                hdu = fits.BinTableHDU.from_columns([col0, col1])
                header = hdu.header
                header['PIXTYPE'] = 'HEALPIX'
                header['ORDERING'] = 'NESTED'
                header['COORDSYS'] = 'C'
                header['EXTNAME'] = f"{key} Map"
                header['NSIDE'] = 2**self.config.outer_level
                header['FIRSTPIX'] = 0
                header['LASTPIX'] = self.config.outer_npix - 1
                header['INDXSCHM'] = 'IMPLICIT'
                header['OBJECT'] = 'FULLSKY'
                hdu.writeto(filename, overwrite=force_create)
            else:
                if verbose:
                    print(f"Opening {filename}")

            if read_only:
                hdul = fits.open(filename, mode='readonly')
            else:
                hdul = fits.open(filename, mode='update')
            self[key] = hdul

    def get_map_values(self, key, pixs=None, coords=(), return_details=False):
        if pixs is not None:
            values = self[key][1].data[FITS_COLUMN_MAP_VALUE][pixs]

            if not return_details:
                return values

            coords = healpix.get_skycoord(self.config.outer_level, pixs)

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
            if len(coords) == 4:
                frame = 'icrs'
                (min_lon, max_lon, min_lat, max_lat) = coords
            elif len(coords) == 5:
                (frame, min_lon, max_lon, min_lat, max_lat) = coords
            else:
                raise AOMapException("Invalid number of arguments")

            pixs = np.arange(self.config.outer_npix)
            coords = healpix.get_skycoord(self.config.outer_level, pixs, frame=frame)

            if frame == 'galactic':
                lon = coords.l.degree
                lat = coords.b.degree
            else:
                lon = coords.ra.degree
                lat = coords.dec.degree

            if min_lon > max_lon:
                lon_filter = (lon >= min_lon) | (lon < max_lon)
            else:
                lon_filter = (lon >= min_lon) & (lon < max_lon)

            lat_filter = (lat >= min_lat) & (lat < max_lat)

            row_filter = lon_filter & lat_filter

            values = self[key][1].data[FITS_COLUMN_MAP_VALUE][row_filter]

            if not return_details:
                return values

            pixs = pixs[row_filter]
            coords = coords[row_filter]

            return Table([
                pixs,
                coords.ra.degree,
                coords.dec.degree,
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

    def _update_map_data(self, values, excluded, pixs=None):
        for j, k in enumerate(self):
            if pixs is None:
                np.copyto(self.get_map_values(k), values[:,j])
                np.copyto(self.get_map_excluded(k), excluded)
            else:
                for i, pix in enumerate(pixs):
                    self.get_map_values(k)[pix] = values[i,j]
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
    def _get_outer_pixel_values(config, mode, outer_pix):
        galactic_coord = healpix.get_skycoord(config.outer_level, outer_pix).galactic
        pix_area = healpix.get_area(config.outer_level)
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
                    inner_data = Maps._build_inner_pixel_data(config, outer_pix, num_retries=3)
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

            values = np.full((len(config.maps)), np.nan)

            for i, k in enumerate(config.maps):
                match k:
                    case 'galaxy-model':
                        values[i] = np.mean(Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_GALAXY_MODEL))
                    case 'star-count':
                        values[i] = np.sum(Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_STAR_COUNT))
                    case 'stellar-density':
                        values[i] = np.sum(Maps._get_inner_pixel_data_column(inner_data, FITS_COLUMN_STAR_COUNT)) / pix_area.to(u.arcmin**2).value

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
    def _get_inner_pixel_data(inner_data):
        return inner_data[1]

    @staticmethod
    def _get_inner_pixel_data_column(inner_data, column_name):
        return Maps._get_inner_pixel_data(inner_data).data[column_name]

    @staticmethod
    def _build_inner_pixel_data(config, outer_pix, num_retries=3):
        cols = []

        # Compute Galaxy Density Model
        pixs, coords, _ = healpix.get_subpixels_skycoord(config.outer_level, outer_pix, config.inner_level)
        galaxy_model = Maps._compute_galaxy_model(coords.galactic)

        # Get Stars from Gaia
        gaia_stars = Maps._get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=num_retries)

        # Determine Inner Pixel of Gaia Stars
        gaia_pixs = healpix.get_healpix(config.inner_level, SkyCoord(ra=gaia_stars['gaia_ra'], dec=gaia_stars['gaia_dec'], unit=(u.degree, u.degree)))

        # Count Gaia Stars per Inner Pixel
        unique, counts = np.unique(gaia_pixs, return_counts=True)
        count_map = dict(zip(unique, counts))
        star_count = np.array([count_map.get(p, 0) for p in pixs])

        # Build FITS table
        cols.append(fits.Column(name=FITS_COLUMN_GALAXY_MODEL, format='D', array=galaxy_model, unit='density'))
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
    def _get_gaia_stars_in_outer_pixel(config, outer_pix, num_retries=1):
        filename = f"{Maps._get_gaia_path(config, outer_pix)}/gaia.fits"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if os.path.isfile(filename):
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
