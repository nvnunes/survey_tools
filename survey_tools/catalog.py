#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from copy import deepcopy
from datetime import datetime
import glob
import os
import pathlib
import pickle
import re
import tarfile
from compress_pickle import dump, load
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.io
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u
from astropy.wcs import WCS
from dust_extinction.parameter_averages import F99
from regions import RectangleSkyRegion
from scipy.special import gamma, gammainc # pylint: disable=no-name-in-module
from survey_tools import healpix, sky
from survey_tools.utility import files, table

class StructType:
    pass

class CatalogException(Exception):
    pass

#region Paths

def get_default_data_path():
    data_path = f"{pathlib.Path(__file__).parent.parent.resolve()}/data"

    if not os.path.exists(data_path):
        raise CatalogException('Data files missing. See "data/README.txt" for more details.')

    return data_path

def _get_default_catalog_path(catalog_name = None):
    if catalog_name is not None:
        return f"{get_default_data_path()}/catalogs/{catalog_name}"
    else:
        return f"{get_default_data_path()}/catalogs"

def _get_default_image_path():
    return f"{get_default_data_path()}/images"

#endregion

#region Parameters

def get_params(catalog_name, field_name = None, filter_name = None):
    catalog_params = StructType()
    catalog_params.catalog = catalog_name

    match catalog_name:
        case 'UVISTA' | 'UVISTA-PLUS' | 'ZCOSMOS-BRIGHT' | 'ZCOSMOS-DEEP' | 'DEIMOS' | 'FMOS' | 'LEGAC':
            catalog_params.field = 'COSMOS'
        case _:
            catalog_params.field = field_name

    catalog_params.catalog_path  = _get_default_catalog_path(catalog_name)
    catalog_params.catalog_image_path = _get_default_image_path()

    if field_name is not None:
        catalog_params.field_file_prefix = field_name.upper().replace('-', '')

        match catalog_name:
            case '3D-HST' | '3D-HST-PLUS':
                match field_name:
                    case 'UDS':
                        catalog_params.field_version = 4.2
                    case _:
                        catalog_params.field_version = 4.1

        if filter_name is not None:
            match catalog_name:
                case 'UVISTA' | 'UVISTA-PLUS':
                    catalog_params.catalog_image_folder = f"{catalog_params.catalog_image_path}/UVISTA"
                    match filter_name:
                        case 'J':
                            catalog_params.catalog_image_file = 'ADP.2024-05-27T16_34_32.521.fits'
                        case 'H':
                            catalog_params.catalog_image_file = 'ADP.2024-05-27T16_34_32.523.fits'
                        case 'K':
                            catalog_params.catalog_image_file = 'ADP.2024-05-27T16_34_32.525.fits'
                case '3D-HST' | '3D-HST-PLUS':
                    catalog_params.catalog_image_folder = f"{catalog_params.catalog_image_path}/3D-HST"
                    catalog_params.catalog_image_file = f"{catalog_params.field_file_prefix}_3dhst.v4.0.{filter_name}_orig_sci.fits"

            if table.has_field(catalog_params, 'catalog_image_file'):
                catalog_params.catalog_image_path = f"{catalog_params.catalog_image_folder}/{catalog_params.catalog_image_file}"

    # Other Parameters
    catalog_params.max_star_mag = 22.0  # catalog mag (to reduce number of cross-matches performed)
    catalog_params.min_line_snr = 2.0

    if table.has_field(catalog_params, 'catalog_image_file'):
        image_hdul = open_image(catalog_params)
        catalog_params.wcs = WCS(image_hdul['PRIMARY'].header) # pylint: disable=no-member
        catalog_params.pixel_scale = WCS.proj_plane_pixel_scales(catalog_params.wcs)
        image_hdul.close()

    return catalog_params

def open_image(catalog_params):
    image_hdul = fits.open(catalog_params.catalog_image_path)
    return image_hdul

#endregion

#region Read

class CatalogData:
    def __init__(self, catalog_params, force_tables = False, region = None, region_wcs = None):
        self.params = catalog_params
        self.catalog = catalog_params.catalog
        self.count = 0
        self.field = catalog_params.field
        self.hduls = []

        self.redshift_is_copy = False
        self.spp_is_copy = False
        self.lines_is_copy = False
        self.clumps_is_copy = False

        match catalog_params.catalog:
            case 'UVISTA':
                self.source = 'UVISTA'
                self.name = 'COSMOS/UltraVISTA'
                self.date = datetime(2015, 8, 20)
                self.frame = 'fk5'

                # Columns: id ra dec xpix ypix Ks_tot eKs_tot Ks eKs H eH J eJ Y eY ch4 ech4 ch3 ech3 ch2 ech2 ch1 ech1 zp ezp ip eip rp erp V eV gp egp B eB u eu IA484 eIA484 IA527 eIA527 IA624 eIA624 IA679 eIA679 IA738 eIA738 IA767 eIA767 IB427 eIB427 IB464 eIB464 IB505 eIB505 IB574 eIB574 IB709 eIB709 IB827 eIB827 fuv efuv nuv enuv mips24 emips24 K_flag K_star K_Kron apcor z_spec z_spec_cc z_spec_id star contamination nan_contam orig_cat_id orig_cat_field USE
                file_path = f"{catalog_params.catalog_path}/UVISTA_final_v4.1.cat"
                include_names = ['id', 'ra', 'dec', 'xpix', 'ypix', 'Ks_tot', 'eKs_tot', 'Ks', 'eKs', 'H', 'eH', 'J', 'eJ', 'K_flag', 'K_star', 'K_Kron', 'z_spec', 'z_spec_cc', 'z_spec_id', 'star', 'contamination', 'USE']
                self.sources = astropy.io.ascii.read(file_path, include_names=include_names)
                self.count = len(self.sources) # needed in the code below so cannot wait until the end

                # Columns: id z_spec z_a z_m1 chi_a z_p chi_p z_m2 odds l68 u68 l95 u95 l99 u99 nfilt q_z z_peak peak_prob z_mc
                file_path = f"{catalog_params.catalog_path}/UVISTA_final_v4.1.zout"
                self.redshift = astropy.io.ascii.read(file_path)
                self.redshift['z_spec_cc'] = self.sources['z_spec_cc']

                # Columns: id z tau metal lage Av lmass lsfr lssfr la2t chi2
                file_path = f"{catalog_params.catalog_path}/UVISTA_final_BC03_v4.1.fout"
                self.spp = astropy.io.ascii.read(file_path, header_start=16)
                self.spp['lmass'][np.isnan(self.spp['lmass'])] = -99
                self.spp['lsfr'][np.isnan(self.spp['lsfr'])] = -99
                no_fit_filter = self.spp['chi2'] == -1
                self.spp['ltau'][no_fit_filter]  = -99
                self.spp['metal'][no_fit_filter] = -99
                self.spp['lage'][no_fit_filter]  = -99
                self.spp['Av'][no_fit_filter]    = -99
                self.spp['lmass'][no_fit_filter] = -99
                self.spp['lsfr'][no_fit_filter]  = -99
                self.spp['lssfr'][no_fit_filter] = -99
                self.spp['la2t'][no_fit_filter]  = -99

                #,id,ra,dec,z_spec,z_phot,log_mass,log_sfr,Av,is_mass_clumpy,is_UV_clumpy,UV_frac_clump
                file_path = f"{catalog_params.catalog_path}/Viz_COSMOS_deconv_clumpy_catalog.csv"
                if os.path.exists(file_path):
                    clump_data = astropy.io.ascii.read(file_path)

                    id = np.zeros((self.count), dtype=np.int_) # pylint: disable=redefined-builtin
                    is_mass_clumpy = np.zeros((self.count), dtype=np.bool)
                    is_UV_clumpy = np.zeros((self.count), dtype=np.bool)
                    UV_frac_clump = -1 * np.ones((self.count))

                    for i in np.arange(len(clump_data)):
                        indexes = np.where(self.sources['id'] == clump_data['id'][i])[0]
                        if len(indexes) == 0:
                            continue
                        idx = indexes[0]

                        if self.sources['z_spec'][idx] != clump_data['z_spec'][i]:
                            raise Exception(f"z_spec doesn't match: {clump_data['id'][i]}")

                        id[idx] = clump_data['id'][i]
                        is_mass_clumpy[idx] = clump_data['is_mass_clumpy'][i] == 'True'
                        is_UV_clumpy[idx] = clump_data['is_UV_clumpy'][i] == 'True'
                        UV_frac_clump[idx] = clump_data['UV_frac_clump'][i]

                    self.clumps = Table([
                            id, is_mass_clumpy, is_UV_clumpy, UV_frac_clump
                        ], names=[
                            'id', 'is_mass_clumpy', 'is_UV_clumpy', 'UV_frac_clump',
                        ], dtype=[
                            np.int_, np.bool, np.bool, np.float64,
                        ]
                    )

            case 'ZCOSMOS-BRIGHT':
                self.source = 'ZCB'
                self.name = 'zCOSMOS Bright'
                self.date = datetime(2016, 1, 19)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/zcosmos3.dat"
                readme_path = f"{catalog_params.catalog_path}/ReadMe.txt"

                self.sources = astropy.io.ascii.read(file_path, readme=readme_path, include_names=['zCOSMOS','RAdeg','DEdeg','z','CC','Imag','FileName'])
                self.sources.rename_column('zCOSMOS', 'id')

                source_filter = (self.sources['CC'] > 0) & (self.sources['z'] > 0.0)
                self.sources = self.sources[source_filter]

            case 'ZCOSMOS-DEEP':
                self.source = 'ZCD'
                self.name = 'zCOSMOS Deep'
                self.date = datetime(2016, 1, 19) # Not known
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/cosmos_zspec_zgt2.txt" # zCOSMOS Deep (not public, unreleased)
                if os.path.exists(file_path):
                    self.sources = astropy.io.ascii.read(file_path, include_names=['ra','dec','z_spec','Q_f'])
                    self.sources.add_column(np.arange(len(self.sources))+1, name='id', index=0)
                    self.sources.rename_column('Q_f'  , 'z_spec_cc')

            case '3D-HST':
                self.source = '3DHST'
                self.name = '3D-HST'
                self.date = datetime(2014, 9, 3)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version}.cats/Catalog/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version}.cat.FITS"
                if force_tables:
                    self.sources = Table.read(file_path)
                else:
                    sources_hdul = fits.open(file_path)
                    self.sources = sources_hdul[1].data # pylint: disable=no-member
                    self.hduls.append(sources_hdul)

                file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst_v4.1.5_catalogs/{catalog_params.field_file_prefix}_3dhst.v4.1.5.zfit.linematched.fits"
                if force_tables:
                    self.redshift = Table.read(file_path)
                else:
                    redshift_hdul = fits.open(file_path)
                    self.redshift = redshift_hdul[1].data # pylint: disable=no-member
                    self.hduls.append(redshift_hdul)

                file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version:.1f}.cats/Fast/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version:.1f}.fout.FITS"
                if force_tables:
                    self.spp = Table.read(file_path)
                else:
                    spp_hdul = fits.open(file_path)
                    self.spp = spp_hdul[1].data # pylint: disable=no-member
                    self.hduls.append(spp_hdul)

                file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst_v4.1.5_catalogs/{catalog_params.field_file_prefix}_3dhst.v4.1.5.linefit.linematched.fits"
                if force_tables:
                    self.lines = Table.read(file_path)
                else:
                    line_hdul = fits.open(file_path)
                    self.lines = line_hdul[1].data # pylint: disable=no-member
                    self.hduls.append(line_hdul)

            case 'VUDS':
                self.source = 'VUDS'
                self.name = 'VUDS'
                self.date = datetime(2015, 8, 12)
                self.frame = 'fk5'

                match self.field:
                    case 'COSMOS':
                        file_path = f"{catalog_params.catalog_path}/cosmos.dat"
                    case 'GOODS-S':
                        file_path = f"{catalog_params.catalog_path}/ecdfs.dat"

                readme_path = f"{catalog_params.catalog_path}/ReadMe"

                self.sources = astropy.io.ascii.read(file_path, readme=readme_path, include_names=['VUDS', 'RAdeg', 'DEdeg', 'zspec', 'zflags', 'Mask', 'slit', 'obj'])
                self.sources.rename_column('VUDS', 'id')

                table.add_fields(self.sources, 'index', np.arange(len(self.sources))+1)

            case 'Casey':
                self.source = 'Casey'
                self.name = 'Casey DSFG'
                self.date = datetime(2012, 12, 1)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/apj449592t1_mrt.txt"
                self.sources = astropy.io.ascii.read(file_path)

                table.add_fields(self.sources, 'index', np.arange(len(self.sources))+1)

                self.sources['ra'] = Angle([f"{self.sources['Name'][i][-18:-16]}h{self.sources['Name'][i][-16:-14]}m{self.sources['Name'][i][-14:-9]}s" for i in np.arange(len(self.sources))]).degree
                self.sources['dec'] = Angle([f"{self.sources['Name'][i][-9:-6]}d{self.sources['Name'][i][-6:-4]}m{self.sources['Name'][i][-4:]}s" for i in np.arange(len(self.sources))]).degree

                # Filter to selected field
                match self.field:
                    # U = UDS;
                    # S = CDFS;
                    # C = COSMOS;
                    # L = LHN;
                    # G = GOODS-N;
                    # E = Elais-N1
                    case 'UDS':
                        field_flag = 'U'
                    case 'COSMOS':
                        field_flag = 'C'
                    case 'GOODS-N':
                        field_flag = 'G'
                    case _:
                        field_flag = 'X'

                sources_filter = self.sources['f_zphot'] == field_flag
                self.sources = self.sources[sources_filter]

            case 'DEIMOS':
                self.source = 'DEIMOS'
                self.name = 'DEIMOS 10K'
                self.date = datetime(2017, 12, 27)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/deimos_redshifts.tbl"
                self.sources = astropy.io.ascii.read(file_path)

                table.add_fields(self.sources, 'index', np.arange(len(self.sources))+1)

                source_filter = (self.sources['Remarks'] != 'star') & ~np.isnan(self.sources['zspec'] > 0.0) & (self.sources['zspec'] > 0.0)
                self.sources = self.sources[source_filter]

            case 'MOSDEF':
                self.source = 'MOSDEF'
                self.name = 'MOSDEF'
                self.date = datetime(2018, 3, 11)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/mosdef_zcat.final_slitap.fits"
                sources_hdul = fits.open(file_path)
                self.sources = sources_hdul[1].data # pylint: disable=no-member

                # Filter to selected field
                sources_filter = (self.sources['FIELD'] == self.field) & (self.sources['ID_V4'] >= 0) & (self.sources['Z_MOSFIRE'] >= 0.0)
                self.sources = self.sources[sources_filter]

                # Remove Duplicates
                dup_filter = np.zeros((len(self.sources)), dtype=np.bool)
                unique_ids, counts = np.unique(self.sources['ID_V4'], return_counts=True)
                dupped_ids = unique_ids[counts > 1]

                for i in np.arange(len(dupped_ids)):
                    dup_id = dupped_ids[i]
                    current_filter = self.sources['ID_V4'] == dup_id
                    dup_filter[current_filter] = True
                    keep_idx = np.argmax(current_filter)
                    dup_filter[keep_idx] = False
                    self.sources['z_MOSFIRE'][keep_idx] = np.mean(self.sources['z_MOSFIRE'][current_filter])

                self.sources = self.sources[~dup_filter]

                # Emission Lines (NOTE: these are not line matched, so do so manually)
                file_path = f"{catalog_params.catalog_path}/linemeas_nocor.fits"
                lines_hdul = fits.open(file_path)
                lines = lines_hdul[1].data # pylint: disable=no-member

                line_prefixes = get_MOSDEF_line_prefixes()
                empty_idx = np.argmax(lines['ID'] == -9999) # HACK: use the first row we're going to skip anyway
                lines[empty_idx] = np.zeros((1), dtype=lines.columns.dtype)[0]
                line_indexes = empty_idx * np.ones((len(self.sources)), dtype=np.int_)
                indexes = np.arange(len(lines))
                for i in np.arange(len(self.sources)):
                    row_filter = (lines['FIELD'] == self.sources['FIELD'][i]) & (lines['ID'] == self.sources['ID_V4'][i])
                    num_rows = np.sum(row_filter)
                    keep_idx = np.argmax(row_filter) # first row

                    if num_rows == 0:
                        continue
                    elif num_rows > 1:
                        for j in np.arange(len(line_prefixes)):
                            flux_field_name = f"{line_prefixes[j]}_PreferredFlux"
                            error_field_name = f"{line_prefixes[j]}_PreferredFlux_err"
                            sky_flag_field_name = f"{line_prefixes[j]}_slflag"

                            fluxes = lines[flux_field_name][row_filter]
                            errors = lines[error_field_name][row_filter]
                            slflags = lines[sky_flag_field_name][row_filter]

                            value_filter = (fluxes > 0.0) & (errors > 0.0)
                            if np.sum(value_filter) > 0:
                                weights = 1/np.power(errors[value_filter], 2)
                                lines[keep_idx][flux_field_name] = np.average(fluxes[value_filter], weights=weights)
                                lines[keep_idx][error_field_name] = np.sqrt(1/np.sum(weights))
                                lines[keep_idx][sky_flag_field_name] = min(slflags[value_filter])

                    line_indexes[i] = keep_idx

                self.lines = lines[line_indexes]

                self.hduls = [sources_hdul, lines_hdul]

            case 'FMOS':
                self.source = 'FMOS'
                self.name = 'FMOS-COSMOS'
                self.date = datetime(2019, 1, 25)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/fmos-cosmos_catalog_2019.fits"
                sources_hdul = fits.open(file_path)
                self.sources = sources_hdul[1].data # pylint: disable=no-member
                self.lines = self.sources # copy of sources to access emission line data
                self.lines_is_copy = True
                self.hduls = [sources_hdul]

            case 'KMOS3D':
                self.source = 'KMOS3D'
                self.name = 'KMOS3D'
                self.date = datetime(2019, 7, 12)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/k3d_fnlsp_table_v3.fits"
                sources_hdul = fits.open(file_path)
                self.sources = sources_hdul[1].data # pylint: disable=no-member

                file_path = f"{catalog_params.catalog_path}/k3d_fnlsp_table_hafits_v3.fits"
                lines_hdul = fits.open(file_path)
                self.lines = lines_hdul[1].data # pylint: disable=no-member

                # Filter to selected field
                match self.field:
                    case 'COSMOS':
                        field_code = 'COS'
                    case 'GOODS-S':
                        field_code = 'GS'
                    case 'UDS':
                        field_code = 'U'

                sources_filter = self.sources['FIELD'] == field_code
                self.sources = self.sources[sources_filter]

                sources_filter = self.lines['FIELD'] == field_code
                self.lines = self.lines[sources_filter]

                # Sources and Lines are line-matched but sources has more objects, drop them
                if len(self.sources) > len(self.lines):
                    self.sources = self.sources[0:len(self.lines)]

                self.spp = self.sources # copy of sources to spp data
                self.spp_is_copy = True

                self.hduls = [sources_hdul, lines_hdul]

            case 'C3R2':
                self.source = 'C3R2'
                self.name = 'C3R2 DR1+DR2+DR3'
                self.date = datetime(2021, 6, 18)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/c3r2_DR1+DR2_2019april11.txt"
                self.sources12 = astropy.io.ascii.read(file_path)

                file_path = f"{catalog_params.catalog_path}/C3R2-DR3-18june2021.txt"
                self.sources3 = astropy.io.ascii.read(file_path)

                self.sources = vstack([self.sources12, self.sources3])

                source_filter = np.char.startswith(self.sources['ID'], self.field)
                self.sources = self.sources[source_filter]

                table.add_fields(self.sources, 'index', np.arange(len(self.sources))+1)

                self.sources['ra'] = Angle([f"{self.sources['RAh'][i]}h{self.sources['RAm'][i]}m{self.sources['RAs'][i]}s" for i in np.arange(len(self.sources))]).degree
                self.sources['dec'] = Angle([f"{(self.sources['DE-'][i] if self.sources['DE-'][i] != '' else '+')}{self.sources['DEd'][i]}d{self.sources['DEm'][i]}m{self.sources['DEs'][i]}s" for i in np.arange(len(self.sources))]).degree

            case 'LEGAC':
                self.source = 'LEGAC'
                self.name = 'LEGA-C'
                self.date = datetime(2021, 8, 2)
                self.frame = 'fk5'

                file_path = f"{catalog_params.catalog_path}/legac_dr3_cat.fits"
                sources_hdul = fits.open(file_path)
                self.sources = sources_hdul[1].data # pylint: disable=no-member
                self.sources['ID'] = self.sources['ID'].astype(int)
                self.sources['ID_LEGAC'] = self.sources['ID_LEGAC'].astype(int)

                self.lines = self.sources # copy to access emission line data
                self.lines_is_copy = True
                self.hduls = [sources_hdul]

            case 'HSCSSP':
                self.source = 'HSCSSP'
                self.name = 'HSC SSP Public'
                self.date = datetime(2000, 1, 1) # So all other z-specs of the same file take precidence (since this is a cross-matched catalog)
                self.frame = 'fk5'

                match self.field:
                    case 'AEGIS':
                        file_path = f"{catalog_params.catalog_path}/EGS-specz-v2.3.fits"
                    case 'COSMOS':
                        file_path = f"{catalog_params.catalog_path}/COSMOS-specz-v2.8-public.fits"

                sources_hdul = fits.open(file_path)
                self.sources = sources_hdul[1].data # pylint: disable=no-member
                self.hduls = [sources_hdul]

                self.sources = fits.FITS_rec.from_columns(self.sources.columns.add_col(fits.Column(name='index', array=np.arange(len(self.sources))+1, format='K')))

            case 'DESI':
                self.source = 'DESI'
                self.name = 'DESI EDR'
                self.date = datetime(2023, 6, 9)
                self.frame = 'icrs'

                file_path = f"{catalog_params.catalog_path}/edr_galaxy_stellarmass_lineinfo_v1.0.fits"
                sources_hdul = fits.open(file_path)
                self.sources = sources_hdul[1].data # pylint: disable=no-member
                self.spp = self.sources # copy to access spp line data
                self.spp_is_copy = True
                self.lines = self.sources # copy to access emission line data
                self.lines_is_copy = True
                self.hduls = [sources_hdul]

                self.sources = fits.FITS_rec.from_columns(self.sources.columns.add_col(fits.Column(name='index', array=np.arange(len(self.sources))+1, format='K')))

                kron_radius = np.zeros((len(self.sources)))
                value_filter = (self.sources['SERSIC'] > 0.0) & (self.sources['SHAPE_R'] > 0.0)
                kron_radius[value_filter] = compute_kron_radius(self.sources['SERSIC'][value_filter], Re=self.sources['SHAPE_R'][value_filter])
                self.sources = fits.FITS_rec.from_columns(self.sources.columns.add_col(fits.Column(name='kron_radius', array=kron_radius, format='D')))

            case 'UVISTA-PLUS':
                catalog_file_path = f"{catalog_params.catalog_path}/UVISTA-PLUS_COSMOS.pkl.gz"
                with open(catalog_file_path, 'rb') as f:
                    loaded_data = load(f)
                    self.__dict__.update(loaded_data.__dict__)

                self.params = catalog_params
                self.source = 'UVISTAP'

            case '3D-HST-PLUS':
                catalog_file_path = f"{catalog_params.catalog_path}/3D-HST-PLUS_{catalog_params.field}.pkl.gz"
                with open(catalog_file_path, 'rb') as f:
                    loaded_data = load(f)
                    self.__dict__.update(loaded_data.__dict__)

                self.params = catalog_params
                self.source = '3DHSTP'

        if not table.has_field(self, 'count'):
            self.count = len(self.sources)
        if not table.has_field(self, 'all_sources'):
            self.all_sources = [self.source]
        if not table.has_field(self, 'all_sources_date'):
            self.all_sources_date = [self.date]

        if table.has_field(self, 'spp') and isinstance(self.spp, Table) and not table.has_field(self.spp, 'spp_flag'):
            _, spp_flag, _ = get_spp(self)
            table.add_fields(self.spp, 'spp_flag', spp_flag)

        if region is not None:
            if region_wcs is None:
                if table.has_field(self.params, 'wcs'):
                    region_wcs = self.params.wcs
                else:
                    region_wcs = WCS(naxis=2)                                  # 2D WCS for an image
                    region_wcs.wcs.crpix = [180, 180]*3600                     # Reference pixel (center of the image)
                    region_wcs.wcs.cdelt = np.array([-0.00027778, 0.00027778]) # Pixel scale in degrees (i.e., 1 arcsec/pixel)
                    region_wcs.wcs.crval = [180.0, 0.0]                        # Reference world coordinate (RA, Dec in degrees)
                    region_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]            # Projection type (TAN for tangent-plane)

            sources_filter = region.contains(SkyCoord(ra=self.sources[get_ra_field(self)], dec=self.sources[get_dec_field(self)], unit=(u.degree, u.degree)), region_wcs)

            self.sources = self.sources[sources_filter]
            self.count = len(self.sources)

            if table.has_field(self, 'redshift'):
                if self.redshift_is_copy:
                    self.redshift = self.sources
                else:
                    self.redshift = self.redshift[sources_filter]

            if table.has_field(self, 'spp'):
                if self.spp_is_copy:
                    self.spp = self.sources
                else:
                    self.spp = self.spp[sources_filter]

            if table.has_field(self, 'lines'):
                if self.lines_is_copy:
                    self.lines = self.sources
                else:
                    self.lines = self.lines[sources_filter]

            if table.has_field(self, 'clumps'):
                if self.clumps_is_copy:
                    self.clumps = self.sources
                else:
                    self.clumps = self.clumps[sources_filter]

    def close(self):
        for i in np.arange(len(self.hduls)):
            self.hduls[i].close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def id_field(self):
        return get_id_field(self)

    @property
    def ra_field(self):
        return get_ra_field(self)

    @property
    def dec_field(self):
        return get_dec_field(self)

    def get_index(self, ids):
        if np.size(ids) <= 1:
            return np.where(self.sources[self.id_field] == ids)[0][0]
        else:
            return np.where(np.isin(self.sources[self.id_field], ids))[0]

#endregion

#region Identifier Info

def get_id_field(catalog_data_or_params, table_name=None):
    match catalog_data_or_params.catalog:
        case '3D-HST' | '3D-HST-PLUS':
            match table_name:
                case 'redshift':
                    return 'phot_id'
                case 'lines':
                    return 'number'
                case _:
                    return 'id'
        case 'VUDS' | 'Casey' | 'DEIMOS' | 'C3R2' | 'HSCSSP':
            return 'index'
        case 'MOSDEF':
            return 'ID_V4'
        case 'FMOS':
            return 'INDEX'
        case 'KMOS3D':
            return 'ID_SKELTON'
        case 'LEGAC':
            return 'ID_LEGAC'
        case 'DESI':
            return 'TARGETID'
        case _:
            return 'id'

def get_ra_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'VUDS' | 'ZCOSMOS-BRIGHT':
            return 'RAdeg'
        case 'DEIMOS':
            return 'Ra'
        case 'MOSDEF' | 'FMOS' | 'KMOS3D' | 'LEGAC' | 'HSCSSP':
            return 'RA'
        case 'DESI':
            return 'TARGET_RA'
        case _:
            return 'ra'

def get_dec_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'VUDS' | 'ZCOSMOS-BRIGHT':
            return 'DEdeg'
        case 'DEIMOS':
            return 'Dec'
        case 'MOSDEF' | 'FMOS' | 'KMOS3D' | 'LEGAC' | 'HSCSSP':
            return 'DEC'
        case 'DESI':
            return 'TARGET_DEC'
        case _:
            return 'dec'

def get_x_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA' | 'UVISTA-PLUS':
            return 'xpix'
        case _:
            return 'x'

def get_y_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA' | 'UVISTA-PLUS':
            return 'ypix'
        case _:
            return 'y'

#endregion

#region Flag Info

def get_use_phot_sources():
    return np.array(['UVISTA','3DHST','KMOS3D','DESI'])

def get_use_phot_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA':
            return 'USE'
        case '3D-HST':
            return 'use_phot'
        case 'KMOS3D' | 'DESI':
            return True
    return None

def get_has_use_phot(catalog_data):
    return np.any(get_use_phot_sources() == catalog_data.source)

def get_star_flag_sources():
    return np.array(['UVISTA','3DHST'])

def get_star_flag_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA':
            return 'star'
        case '3D-HST':
            return 'star_flag'
    return None

def get_has_star_flag(catalog_data):
    return np.any(get_star_flag_sources() == catalog_data.source)

#endregion

#region Flux Info

def get_flux_sources():
    return np.array(['UVISTA','3DHST','KMOS3D','DESI'])

def get_flux_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA':
            return 'Ks_tot'
        case '3D-HST':
            return 'f_f160w'
        case 'KMOS3D':
            return 'M_KS'
        case 'DESI':
            return 'FLUX_Z'
    return None

def get_flux_zero_point(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA':
            return 25
        case '3D-HST':
            return 25
        case 'KMOS3D':
            return 25
        case 'DESI':
            return 22.5
    return 25

def get_has_flux(catalog_data):
    return np.any(get_flux_sources() == catalog_data.source)

#endregion

#region Flux Radius Info

def get_flux_radius_sources():
    return np.array(['UVISTA','3DHST','DESI','UVISTAP','3DHSTP'])

def get_flux_radius_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA' | 'UVISTA-PLUS':
            return 'K_Kron'
        case '3D-HST' | '3D-HST-PLUS':
            return 'kron_radius'
        case 'DESI':
            return 'kron_radius'
    return None

def get_has_flux_radius(catalog_data):
    return np.any(get_flux_radius_sources() == catalog_data.source)

def get_flux_radius_factor(catalog_data):
    match catalog_data.source:
        case 'UVISTA' | '3DHST' | 'UVISTAP' | '3DHSTP':
            return catalog_data.params.pixel_scale[1].value * 3600.0 # pixels -> arcsec
        case 'DESI':
            return 1.0 # arcsec
    return 1.0

def get_is_flux_radius_kron(catalog_data):
    flux_radius_field = get_flux_radius_field(catalog_data)
    if flux_radius_field is None:
        return False
    return 'kron' in flux_radius_field.lower()

#endregion

#region Redshift Info

def get_redshift_spec(catalog_data, idx_or_filter = None, force_catalog_name = None):
    if force_catalog_name is not None:
        catalog_name = force_catalog_name
    else:
        catalog_name = catalog_data.catalog

    if idx_or_filter is None:
        N = catalog_data.count
        idx_or_filter = np.ones((N), dtype=np.bool)
    elif np.size(idx_or_filter) == 1:
        if idx_or_filter == -1:
            return -99.0, 99
        N = 1
    elif issubclass(idx_or_filter.dtype.type, np.integer):
        N = len(idx_or_filter)
    else:
        N = np.sum(idx_or_filter)

    ##############################
    # Redshift
    ##############################

    if N == 1:
        z_spec = -99.0
    else:
        z_spec = -99.0 * np.ones(N)

    match catalog_name:
        case 'UVISTA':
            z_spec = catalog_data.redshift['z_spec'][idx_or_filter]

        case 'ZCOSMOS-BRIGHT':
            z_spec = catalog_data.sources['z'][idx_or_filter]

        case 'ZCOSMOS-DEEP':
            z_spec = catalog_data.sources['z_spec'][idx_or_filter]

        case '3D-HST':
            if N == 1:
                z_spec = -99.0
                if catalog_data.redshift['z_best_s'][idx_or_filter] <= 2:
                    z_spec = catalog_data.redshift['z_best'][idx_or_filter]
                elif catalog_data.redshift['z_spec'][idx_or_filter] > 0.0:
                    z_spec = catalog_data.redshift['z_spec'][idx_or_filter]
                elif catalog_data.redshift['use_zgrism'][idx_or_filter]:
                    z_spec = catalog_data.redshift['z_max_grism'][idx_or_filter]
            else:
                value_filter = catalog_data.redshift['z_best_s'][idx_or_filter] <= 2
                z_spec[value_filter] = catalog_data.redshift['z_best'][idx_or_filter][value_filter]
                value_filter = (z_spec < 0) & (catalog_data.redshift['z_spec'][idx_or_filter] > 0.0)
                z_spec[value_filter] = catalog_data.redshift['z_spec'][idx_or_filter][value_filter]
                value_filter = (z_spec < 0) & (catalog_data.redshift['use_zgrism'][idx_or_filter] == 1)
                z_spec[value_filter] = catalog_data.redshift['z_max_grism'][idx_or_filter][value_filter]

        case 'VUDS':
            z_spec = catalog_data.sources['zspec'][idx_or_filter]

        case 'Casey':
            z_spec = catalog_data.sources['zspec'][idx_or_filter]

        case 'DEIMOS':
            z_spec = catalog_data.sources['zspec'][idx_or_filter]

        case 'MOSDEF':
            z_spec = catalog_data.sources['Z_MOSFIRE'][idx_or_filter]

        case 'FMOS':
            z_spec = catalog_data.sources['ZBEST'][idx_or_filter]

        case 'KMOS3D':
            z_spec = catalog_data.sources['Z'][idx_or_filter]

        case 'C3R2':
            z_spec = catalog_data.sources['zspec'][idx_or_filter]

        case 'LEGAC':
            z_spec = catalog_data.sources['Z_SPEC'][idx_or_filter]

        case 'DESI':
            z_spec = catalog_data.sources['Z'][idx_or_filter]

        case 'HSCSSP':
            z_spec = catalog_data.sources['Z_SPEC'][idx_or_filter]

        case 'UVISTA-PLUS' | '3D-HST-PLUS':
            if N == 1:
                if catalog_data.best['z_best_flag'][idx_or_filter] < 10:
                    z_spec = catalog_data.best['z_best'][idx_or_filter]
                else:
                    z_spec = catalog_data.sources['z_spec'][idx_or_filter]
            else:
                value_filter = catalog_data.best['z_best_flag'][idx_or_filter] < 10
                z_spec[value_filter] = catalog_data.best['z_best'][idx_or_filter][value_filter]
                z_spec[~value_filter] = catalog_data.sources['z_spec'][idx_or_filter][~value_filter]

    ##############################
    # Redshift Quality Flag
    ##############################

    # z_flag =
    #    1: Best Spectroscopic Redshift
    #    2: Usable Spectroscopic Redshift
    #    3: Best Grism Redshift
    #    4: Usable Grism Redshift
    #    7: Less Reliable Spectroscopic
    #    8: Inconsistent Spectroscopic Redshifts
    #    9: Unreliable Spectroscopic Redshift
    #   11: Reliable Photometric Redshift
    #   19: Unreliable Photometric Redshift
    #   81: Possible AGN
    #   89: Contamination from multiple sources
    #   99: Not provided

    if N == 1:
        z_spec_flag = 99
    else:
        z_spec_flag = 99 * np.ones((N), dtype=np.int_)

    match catalog_name:
        case 'UVISTA' | 'ZCOSMOS-BRIGHT' | 'ZCOSMOS-DEEP' | 'VUDS' | 'DEIMOS':
            match catalog_name:
                case 'ZCOSMOS-BRIGHT':
                    z_field_name = 'z'
                    z_flag_field_name = 'CC'
                case 'VUDS':
                    z_field_name = 'zspec'
                    z_flag_field_name = 'zflags'
                case 'DEIMOS':
                    z_field_name = 'zspec'
                    z_flag_field_name = 'Qf'
                case _:
                    z_field_name = 'z_spec'
                    z_flag_field_name = 'z_spec_cc'

            # Definition of Confidence Classes Class (z_spec_cc)
            #   9   A securely detected line which is believed to be either [OII] 3727 or Ha 6563
            #   4   A very secure redshift with an exhibition-quality spectrum
            #   3   A very secure redshift
            #   2   A likely redshift about which there is some doubt
            #   1   An insecure redshift
            #   0   No redshift measurement attempted
            #   +10 As above but for broad line AGN, with 18 instead of 9, reflecting the greater range of possibilities
            #   +20 or +200 As above but for a target only observed as a secondary target in a slit centered on another object
            # Decimal place modifiers :
            #   .5 The spectroscopic and photometric redshifts are consistent to within 0.08(1+z)
            #   .4 No photometric redshift is available for some reason
            #   .3 For Class 9 and 18 one-line redshifts only, the spectroscopic redshift is consistent only after
            #      the spectroscopic redshift is changed to the alternate redshift
            #   .1 The spectroscopic and photometric redshifts differ by more than 0.08(1+z)
            #
            # Note. The set of objects defined as Classes 1.5, 2.4, 2.5, 9.3, 9.5 and all Class 3.x and 4.x comprise
            #       88% of the sample (95% within 0.5 <z< 0.8) and are estimated to be 99% reliable.
            if N == 1:
                if catalog_data.sources[z_field_name][idx_or_filter] > 0.0:
                    if catalog_data.sources[z_flag_field_name][idx_or_filter] >= 4 and catalog_data.sources[z_flag_field_name][idx_or_filter] < 5:
                        z_spec_flag = 1
                    elif catalog_data.sources[z_flag_field_name][idx_or_filter] >= 3 and catalog_data.sources[z_flag_field_name][idx_or_filter] < 4:
                        z_spec_flag = 2
                    elif catalog_data.sources[z_flag_field_name][idx_or_filter] > 0 and catalog_data.sources[z_flag_field_name][idx_or_filter] < 10:
                        z_spec_flag = 9
                    elif catalog_data.sources[z_flag_field_name][idx_or_filter] >= 10 and catalog_data.sources[z_flag_field_name][idx_or_filter] < 20:
                        z_spec_flag = 81
                    elif catalog_data.sources[z_flag_field_name][idx_or_filter] < 0 or catalog_data.sources[z_flag_field_name][idx_or_filter] >= 20:
                        z_spec_flag = 89
            else:
                value_filter =                       (catalog_data.sources[z_field_name][idx_or_filter] > 0.0) & (catalog_data.sources[z_flag_field_name][idx_or_filter] >=  4) & (catalog_data.sources[z_flag_field_name][idx_or_filter] <  5)
                z_spec_flag[value_filter] = 1

                value_filter = (z_spec_flag == 99) & (catalog_data.sources[z_field_name][idx_or_filter] > 0.0) & (catalog_data.sources[z_flag_field_name][idx_or_filter] >=  3) & (catalog_data.sources[z_flag_field_name][idx_or_filter] <  4)
                z_spec_flag[value_filter] = 2

                value_filter = (z_spec_flag == 99) & (catalog_data.sources[z_field_name][idx_or_filter] > 0.0) & (catalog_data.sources[z_flag_field_name][idx_or_filter] >   0) & (catalog_data.sources[z_flag_field_name][idx_or_filter] < 10)
                z_spec_flag[value_filter] = 9

                value_filter = (z_spec_flag == 99) & (catalog_data.sources[z_field_name][idx_or_filter] > 0.0) & (catalog_data.sources[z_flag_field_name][idx_or_filter] >= 10) & (catalog_data.sources[z_flag_field_name][idx_or_filter] < 20)
                z_spec_flag[value_filter] = 81

                value_filter = (z_spec_flag == 99) & (catalog_data.sources[z_field_name][idx_or_filter] > 0.0) & ((catalog_data.sources[z_flag_field_name][idx_or_filter] < 0) | (catalog_data.sources[z_flag_field_name][idx_or_filter] >= 20))
                z_spec_flag[value_filter] = 89

        case '3D-HST':
            flux_Ha = catalog_data.lines['Ha_FLUX'][idx_or_filter]
            e_Ha = catalog_data.lines['Ha_FLUX_ERR'][idx_or_filter]
            flux_OIII = catalog_data.lines['OIII_FLUX'][idx_or_filter]
            e_OIII = catalog_data.lines['OIII_FLUX_ERR'][idx_or_filter]
            flux_Hb = catalog_data.lines['Hb_FLUX'][idx_or_filter]
            e_Hb = catalog_data.lines['Hb_FLUX_ERR'][idx_or_filter]
            flux_OII = catalog_data.lines['OII_FLUX'][idx_or_filter]
            e_OII = catalog_data.lines['OII_FLUX_ERR'][idx_or_filter]

            if N == 1:
                snr_Ha = (flux_Ha / e_Ha) if e_Ha > 0.0 else 0.0
                snr_OIII = (flux_OIII / e_OIII) if e_OIII > 0.0 else 0.0
                snr_Hb = (flux_Hb / e_Hb) if e_Hb > 0.0 else 0.0
                snr_OII = (flux_OII / e_OII) if e_OII > 0.0 else 0.0
            else:
                snr_Ha = np.zeros((N))
                value_filter = e_Ha > 0
                snr_Ha[value_filter] = flux_Ha[value_filter] / e_Ha[value_filter]
                snr_OIII = np.zeros((N))
                value_filter = e_OIII > 0
                snr_OIII[value_filter] = flux_OIII[value_filter] / e_OIII[value_filter]
                snr_Hb = np.zeros((N))
                value_filter = e_Hb > 0
                snr_Hb[value_filter] = flux_Hb[value_filter] / e_Hb[value_filter]
                snr_OII = np.zeros((N))
                value_filter = e_OII > 0
                snr_OII[value_filter] = flux_OII[value_filter] / e_OII[value_filter]

            # z_best_s:
            #   1 = Spectroscopic
            #   2 = Grism
            #   3 = Photometric
            if N == 1:
                match catalog_data.redshift['z_best_s'][idx_or_filter]:
                    case 1:
                        z_spec_flag = 1
                    case 2:
                        if catalog_data.redshift['use_zgrism'][idx_or_filter] != 1:
                            z_spec_flag = 9
                        # Grism redshift has at least one strong emission line (Ha, [OIII], Hb, [OII])
                        elif (snr_Ha   >= catalog_data.params.min_line_snr) \
                         | (snr_OIII >= catalog_data.params.min_line_snr) \
                         | (snr_Hb   >= catalog_data.params.min_line_snr) \
                         | (snr_OII  >= catalog_data.params.min_line_snr):
                            z_spec_flag = 3
                        else:
                            z_spec_flag = 4
            else:
                value_filter = catalog_data.redshift['z_best_s'][idx_or_filter] == 1
                z_spec_flag[value_filter] = 1

                value_filter = (z_spec_flag == 99) \
                             & (catalog_data.redshift['z_best_s'][idx_or_filter] == 2) \
                             & (catalog_data.redshift['use_zgrism'][idx_or_filter] != 1)
                z_spec_flag[value_filter] = 9

                value_filter = (z_spec_flag == 99) \
                             & (catalog_data.redshift['z_best_s'][idx_or_filter] == 2) \
                             & (catalog_data.redshift['use_zgrism'][idx_or_filter] == 1) & ( \
                                    (snr_Ha   >= catalog_data.params.min_line_snr) \
                                  | (snr_OIII >= catalog_data.params.min_line_snr) \
                                  | (snr_Hb   >= catalog_data.params.min_line_snr) \
                                  | (snr_OII  >= catalog_data.params.min_line_snr) \
                               )
                z_spec_flag[value_filter] = 3

                value_filter = (z_spec_flag == 99) \
                             & (catalog_data.redshift['z_best_s'][idx_or_filter] == 2) \
                             & (catalog_data.redshift['use_zgrism'][idx_or_filter] == 1)
                z_spec_flag[value_filter] = 4

        case 'Casey':
            # 43 I1     ---      Conf   Confidence in zspec identification (4)
            # Note (4): Ranges from 1 to 5, 5 being the most confident (further
            # identifications at Conf < 1 have been excluded from this paper).

            if N == 1:
                if catalog_data.sources['Conf'][idx_or_filter] == 5:
                    z_spec_flag = 1
                elif catalog_data.sources['Conf'][idx_or_filter] >= 3:
                    z_spec_flag = 2
                elif catalog_data.sources['Conf'][idx_or_filter] < 3:
                    z_spec_flag = 9
            else:
                value_filter =                        catalog_data.sources['Conf'][idx_or_filter] == 5
                z_spec_flag[value_filter] = 1
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Conf'][idx_or_filter] >= 3)
                z_spec_flag[value_filter] = 2
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Conf'][idx_or_filter] <  3)
                z_spec_flag[value_filter] = 9

        case 'MOSDEF':
            # Z_MOSFIRE_ZQUAL:
            # 7: Redshift is based on multiple emission features detected with S/N>=2, or else robust absorption-line redshift.
            # 6: Redshift is based on single emission feature detected with S/N>=3, and is within 95% confidence interval of photo-z or within delta(z)=0.05 of pre-MOSFIRE spec-z (if it exists)
            # 5: Redshift is based on single emission feature detected with 2<=S/N<3, and is within 95% confidence interval of photo-z or within delta(z)=0.05 of pre-MOSFIRE spec-z (if it exists)
            # 4: Redshift is based on single emission feature detected with S/N>=3, and is neither within 95% confidence interval of photo-z nor within delta(z)=0.05 of pre-MOSFIRE spec-z (if it exists)
            # 3: Redshift is based on single emission feature detected with 2<=S/N<3, and is neither within 95% confidence interval of photo-z nor or within delta(z)=0.05 of pre-MOSFIRE spec-z (if it exists)
            # 2: Redshift is based on visual inspection, but no emission feature formally detected with S/N>=2, and is within 95% confidence interval of photo-z or within delta(z)=0.05 of pre-MOSFIRE spec-z (if it exists)
            # 1: Redshift is based on visual inspection, but no emission feature formally detected with S/N>=2, and is neither within 95% confidence interval of photo-z nor within delta(z)=0.05 of pre-MOSFIRE spec-z (if it exists)
            # 0: No redshift measured, either based on formal line detections or visual inspection (corresponds to Z_MOSFIRE=-1).
            if N == 1:
                if   catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] == 7:
                    z_spec_flag = 1
                elif catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] >= 5:
                    z_spec_flag = 2
                elif catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] >= 4:
                    z_spec_flag = 7
                elif catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] < 4:
                    z_spec_flag = 9
            else:
                value_filter =                        catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] == 7
                z_spec_flag[value_filter] = 1
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] >= 5)
                z_spec_flag[value_filter] = 2
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] >= 4)
                z_spec_flag[value_filter] = 7
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Z_MOSFIRE_ZQUAL'][idx_or_filter] <  4)
                z_spec_flag[value_filter] = 9

        case 'FMOS':
            # zFlag=0: No emission line detected.
            # zFlag=1: Presence of a single emission line detected at 1.5 <= S/N < 3.
            # zFlag=2: One emission line detected at 3 <= S/N < 5.
            # zFlag=3: One emission line detected at S/N >= 5.
            # zFlag=4: One emission line at S/N >= 5 and a second line at S/N >= 3 that confirms the redshift.
            # zFlag=-1 for flux calibration stars.
            # zFlag=-99 if the spectroscopy failed.
            if N == 1:
                if catalog_data.sources['ZFLAG'][idx_or_filter] >= 1:
                    z_spec_flag = 1
            else:
                value_filter = catalog_data.sources['ZFLAG'][idx_or_filter] >= 1
                z_spec_flag[value_filter] = 1

        case 'KMOS3D':
            # FLAG_ZQUALITY : [-1] non-detection [0] redshift is secure [1] redshift/detection is uncertain
            if N == 1:
                if catalog_data.sources['FLAG_ZQUALITY'][idx_or_filter] == 0:
                    z_spec_flag = 1
                elif catalog_data.sources['FLAG_ZQUALITY'][idx_or_filter] == 1:
                    z_spec_flag = 9
            else:
                value_filter = catalog_data.sources['FLAG_ZQUALITY'][idx_or_filter] == 0
                z_spec_flag[value_filter] = 1
                value_filter = catalog_data.sources['FLAG_ZQUALITY'][idx_or_filter] == 1
                z_spec_flag[value_filter] = 9

        case 'C3R2':
            # 1. Qual = 4: A quality flag of 4 indicates an unambiguous redshift identified with
            # multiple features or the presence of the split [O II]λ3727 doublet.
            #
            # 2. Qual = 3.5: A quality flag of 3.5 indicates a high-confidence redshift based
            # on a single line, with a remote possibility of an incorrect identification. An example
            # might be a strong, isolated emission line identified as Hα, where other identifications
            # of the line are highly improbable due to the lack of associated lines or continuum
            # breaks. This flag is typically only adopted for LRIS and MOSFIRE spectra; single line
            # redshifts in DEIMOS spectra are usually the OII doublet, which is split by the DEIMOS
            # spectral resolution.
            #
            # 3. Qual = 3: A quality flag of 3 indicates a high-confidence redshift with a low
            # probability of an incorrect identifica- tion. An example might be the low
            # signal-to-noise ratio (S/N) detection of an emission line, possibly corrupted by
            # telluric emission or absorption, identified as [OII] λ3727, but where the data quality
            # is insufficient to clearly resolve the doublet.
            #
            # 4. Qual = 2/1: A quality flag of 2 indicates a reasonable guess, while a quality flag
            # of 1 indicates a highly uncertain guess. Sources with these low-confidence redshifts
            # are not included in the data release.
            #
            # 5. Qual = 0: A quality flag of 0 indicates that no redshift could be identified.
            # As described above, a code indicating the cause of the redshift failure is assigned
            # in place of the redshift:
            #   1. Code = -91: Insufficient S/N;
            #   2. Code = -92: Well-detected but no discernible features;
            #   3. Code = -93: Problem with the reduction;
            #   4. Code = -94: Missing slit (an extreme case of -93).

            if N == 1:
                if   catalog_data.sources['Qual'][idx_or_filter] >= 4.0:
                    z_spec_flag = 1
                elif catalog_data.sources['Qual'][idx_or_filter] >= 3.5:
                    z_spec_flag = 2
                elif catalog_data.sources['Qual'][idx_or_filter] >= 3.0:
                    z_spec_flag = 7
                elif catalog_data.sources['Qual'][idx_or_filter]  < 3.0:
                    z_spec_flag = 9
            else:
                value_filter =                        catalog_data.sources['Qual'][idx_or_filter] >= 4.0
                z_spec_flag[value_filter] = 1
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Qual'][idx_or_filter] >= 3.5)
                z_spec_flag[value_filter] = 2
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Qual'][idx_or_filter] >= 3.0)
                z_spec_flag[value_filter] = 7
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Qual'][idx_or_filter]  < 3.0)
                z_spec_flag[value_filter] = 9

        case 'LEGAC':
            # We have created two flags that should be kept in mind when assigning meaning to
            # the measured quantities.
            #
            # FLAG_SPEC: a subset of spectra show clear evidence of
            # AGNs affecting the continuum shape, compromising the interpretation of the index
            # measurements. Upon visual inspection, we decided to flag all 107 galaxies with
            # mid-infrared and/or X-ray AGNs even if the majority do not show an obvious issue;
            # the catalog entry FLAG_SPEC = 1 indicates such AGNs. Narrow-line and radio AGNs
            # do not present a problem for our spectral analysis and are not flagged. However,
            # in 25 cases, we found that the photometry-based flux calibration showed significant
            # imperfections, potentially compromising the measurement of absorption and emission
            # indices. These are indicated by FLAG_SPEC = 2.
            #
            # FLAG_MORPH: in most cases, the light coming through the slit is from a single galaxy
            # with a regular morphology, but in a significant minority of cases, this is not the
            # case. In order to address this, we devise the catalog parameter FLAG_MORPH with
            # value 0, 1, or 2. Single galaxies with regular morphologies have a value 0. In spectra
            # for which the light coming through the slit does not come from single, regular galaxies
            # get a flag value 1; these are often irregular galaxies such as merger remnants, but
            # also multiple galaxies that are separated in the HST image, but not in the spectrum.
            # The guiding principle when assigning flag values of 1 is that the combination of the
            # stellar velocity dispersion (from the spectrum) and structural parameters (from the
            # HST image) cannot be included in an analysis of the fundamental plane or other scaling
            # relations. Spectra that have, on top of this, the problem that the light comes from
            # galaxies at different redshifts receive a flag value 2. Again, the guiding principle
            # is that the presence of a secondary object at the different redshift prevents us from
            # using the stellar velocity dispersion and the structural parameters for a scaling
            # relation analysis. In total, 257 are flagged, 14 of which have flag value 2. Many
            # more galaxies have low-level “contamination” from secondary sources in the slit, but
            # not to the extent that the measurements presented here are ambiguous in their
            # interpretation.
            if N == 1:
                if catalog_data.sources['FLAG_SPEC'][idx_or_filter] == 0 and catalog_data.sources['FLAG_MORPH'][idx_or_filter] == 0:
                    z_spec_flag = 1
                elif catalog_data.sources['FLAG_SPEC'][idx_or_filter] == 0 and catalog_data.sources['FLAG_MORPH'][idx_or_filter] > 0:
                    z_spec_flag = 89
                elif catalog_data.sources['FLAG_SPEC'][idx_or_filter] > 0:
                    z_spec_flag = 81
            else:
                value_filter =                       (catalog_data.sources['FLAG_SPEC'][idx_or_filter] == 0) & (catalog_data.sources['FLAG_MORPH'][idx_or_filter] == 0)
                z_spec_flag[value_filter] = 1
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['FLAG_SPEC'][idx_or_filter] == 0) & (catalog_data.sources['FLAG_MORPH'][idx_or_filter] > 0)
                z_spec_flag[value_filter] = 89
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['FLAG_SPEC'][idx_or_filter] > 0)
                z_spec_flag[value_filter] = 81

        case 'HSCSSP':
            # Q=1 means no redshift could be estimated;
            # Q=2 means a possible, but doubtful, redshift estimate;
            # Q=3 means a ‘probable’ redshift (notionally 90% confidence);
            # Q=4 means a reliable redshift (notionally 99% confidence);
            # Q=5 means a reliable redshift and a high-quality spectrum.
            # Note that this quality parameter is determined entirely by
            # the subjective judgement of the user, and is independent of
            # the automatic quality parameter Qb. Quality classes 1 and
            # 2 are considered failures, so the redshift completeness is the
            # number of Q=3,4,5 redshifts divided by the total number of
            # galaxies in the field. The standard redshift sample comprises
            # objects with Q≥3, but (for applications where redshift
            # reliability is more important than completeness) one may
            # prefer to use the set of objects with Q≥4.
            if N == 1:
                if   catalog_data.sources['Z_QUAL'][idx_or_filter] == 5:
                    z_spec_flag = 1
                elif catalog_data.sources['Z_QUAL'][idx_or_filter] >= 4:
                    z_spec_flag = 2
                elif catalog_data.sources['Z_QUAL'][idx_or_filter] >= 3:
                    z_spec_flag = 7
                elif catalog_data.sources['Z_QUAL'][idx_or_filter]  < 3:
                    z_spec_flag = 9
            else:
                value_filter =                        catalog_data.sources['Z_QUAL'][idx_or_filter] == 5
                z_spec_flag[value_filter] = 1
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Z_QUAL'][idx_or_filter] >= 4)
                z_spec_flag[value_filter] = 2
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Z_QUAL'][idx_or_filter] >= 3)
                z_spec_flag[value_filter] = 7
                value_filter = (z_spec_flag == 99) & (catalog_data.sources['Z_QUAL'][idx_or_filter]  < 3)
                z_spec_flag[value_filter] = 9

        case 'DESI':
            # edr_galaxy_stellarmass_lineinfo_v1.0.fits: Stellar mass and emission line information for
            # all galaxies in DESI EDR with reliable redshifts, defined as SPECTYPE==GALAXY and ZWARN==0.
            # THEREFORE only galaxies with best redshifts included.
            if N == 1:
                z_spec_flag = 1
            else:
                z_spec_flag[:] = 1

        case 'UVISTA-PLUS' | '3D-HST-PLUS':
            z_spec_flag = catalog_data.best['z_best_flag'][idx_or_filter]

    # Fix datatype issues
    if not np.isscalar(z_spec) and np.isscalar(z_spec_flag):
        z_spec_flag = np.array([z_spec_flag])

    return z_spec, z_spec_flag

def get_redshift_any(catalog_data, idx_or_filter = None, force_catalog_name = None, skip_z_merged = False):
    if force_catalog_name is not None:
        catalog_name = force_catalog_name
    else:
        catalog_name = catalog_data.catalog

    if idx_or_filter is None:
        N = catalog_data.count
        idx_or_filter = np.ones((N), dtype=np.bool)
    elif np.size(idx_or_filter) == 1:
        if idx_or_filter == -1:
            return -99.0, 99
        N = 1
    elif issubclass(idx_or_filter.dtype.type, np.integer):
        N = len(idx_or_filter)
    else:
        N = np.sum(idx_or_filter)

    z, z_flag = get_redshift_spec(catalog_data, idx_or_filter, force_catalog_name)

    # z_flag =
    #   11: Reliable Photometric Redshift
    #   19: Unreliable Photometric Redshift
    #   99: No redshift

    match catalog_name:
        case 'UVISTA':
            if N == 1:
                if z < 0.0 and catalog_data.redshift['z_peak'][idx_or_filter] >= 0.0:
                    z = catalog_data.redshift['z_peak'][idx_or_filter]
                    if catalog_data.sources['USE'][idx_or_filter] == 1:
                        z_flag = 11
                    else:
                        z_flag = 19
            else:
                phot_redshift_filter = (z < 0.0) & (catalog_data.redshift['z_peak'][idx_or_filter] >= 0.0)
                z[phot_redshift_filter] = catalog_data.redshift['z_peak'][idx_or_filter][phot_redshift_filter]

                value_filter = phot_redshift_filter & (catalog_data.sources['USE'][idx_or_filter] == 1)
                z_flag[value_filter] = 11

                value_filter = phot_redshift_filter & (catalog_data.sources['USE'][idx_or_filter] == 0)
                z_flag[value_filter] = 19

        case '3D-HST':
            if N == 1:
                if z < 0 and catalog_data.redshift['z_peak_phot'][idx_or_filter] >= 0.0: #catalog_data.redshift['z_best_s'][idx_or_filter] == 3
                    z = catalog_data.redshift['z_peak_phot'][idx_or_filter]
                    if catalog_data.redshift['use_phot'][idx_or_filter] == 1:
                        z_flag = 11
                    else:
                        z_flag = 19
            else:
                phot_redshift_filter = (z < 0) & (catalog_data.redshift['z_peak_phot'][idx_or_filter] >= 0.0) #catalog_data.redshift['z_best_s'][idx_or_filter] == 3
                z[phot_redshift_filter] = catalog_data.redshift['z_peak_phot'][idx_or_filter][phot_redshift_filter]

                value_filter = phot_redshift_filter & (catalog_data.redshift['use_phot'][idx_or_filter] == 1) #& (catalog_data.redshift['z_best_s'][idx_or_filter] == 3) \
                z_flag[value_filter] = 11

                value_filter = phot_redshift_filter & (catalog_data.redshift['use_phot'][idx_or_filter] == 0)
                z_flag[value_filter] = 19

    if not skip_z_merged and table.has_field(catalog_data, 'redshift') and table.has_field(catalog_data.redshift, 'z_merged'):
        if N == 1:
            if z < 0.0 and catalog_data.redshift['z_merged'][idx_or_filter] >= 0.0:
                z = catalog_data.redshift['z_merged'][idx_or_filter]
                z_flag = catalog_data.redshift['z_merged_flag'][idx_or_filter]
        else:
            merged_redshift_filter = (z < 0.0) & (catalog_data.redshift['z_merged'][idx_or_filter] >= 0.0)
            z[merged_redshift_filter] = catalog_data.redshift['z_merged'][idx_or_filter][merged_redshift_filter]
            z_flag[merged_redshift_filter] = catalog_data.redshift['z_merged_flag'][idx_or_filter][merged_redshift_filter]

    # Fix datatype issues
    if not np.isscalar(z) and np.isscalar(z_flag):
        z_flag = np.array([z_flag])

    return z, z_flag

#endregion

#region SPP Info

def get_spp_sources():
    return np.array(['UVISTA','3DHST','KMOS3D','DESI','UVISTAP','3DHSTP'])

def get_has_spp(catalog_data):
    return np.any(get_spp_sources() == catalog_data.source)

def get_spp_names():
    return np.array(['lmass', 'lsfr', 'Av', 'chi2'])

def get_spp(catalog_data, idx_or_filter = None, suffix = None):
    spp_names = get_spp_names()
    num_spp_properties = len(spp_names)

    if idx_or_filter is None:
        N = catalog_data.count
        idx_or_filter = np.ones((N), dtype=np.bool)
    elif np.size(idx_or_filter) == 1:
        if idx_or_filter == -1:
            return -99.0 * np.ones((num_spp_properties))
        N = 1
    elif issubclass(idx_or_filter.dtype.type, np.integer):
        N = len(idx_or_filter)
    else:
        N = np.sum(idx_or_filter)

    if N == 1:
        spp = -99.0 * np.ones((num_spp_properties))
        spp_flags = 99
    else:
        spp = -99.0 * np.ones((N,num_spp_properties))
        spp_flags = 99 * np.ones((N), dtype=np.int_)

    match catalog_data.catalog:
        case 'UVISTA' | 'UVISTA-PLUS':
            field_names = np.array(spp_names, dtype='<U20')
            do_log = [False, False, False, False]
        case '3D-HST' | '3D-HST-PLUS':
            field_names = np.array(['lmass', 'lsfr', 'av', 'chi2'], dtype='<U20')
            do_log = [False, False, False, False]
        case 'KMOS3D':
            field_names = np.array(['LMSTAR', 'SFR', 'SED_AV', ''], dtype='<U20')
            do_log = [False, True, False, False]
        case 'DESI':
            field_names = np.array(['SED_MASS', 'SED_SFR', 'SED_AV', ''], dtype='<U20')
            do_log = [True, True, False, False]

    if (catalog_data.catalog == 'UVISTA-PLUS' or catalog_data.catalog == '3D-HST-PLUS') and suffix is None:
        spp_table = catalog_data.best

        # Hack to temporarily fix inconsistency in field name Av vs av
        if not table.has_field(spp_table, field_names[2]) and field_names[2] == 'av' and table.has_field(spp_table, 'Av'):
            field_names[2] = 'Av'
        elif not table.has_field(spp_table, field_names[2]) and field_names[2] == 'Av' and table.has_field(spp_table, 'av'):
            field_names[2] = 'av'
    else:
        spp_table = catalog_data.spp

    if suffix is not None:
        for i in np.arange(len(field_names)):
            if field_names[i] != '':
                field_names[i] = f"{spp_names[i]}_{suffix}"

    ##############################
    # SPP Values
    ##############################

    for i in np.arange(num_spp_properties):
        if field_names[i] != '':
            if N == 1:
                if do_log[i]:
                    if spp_table[field_names[i]][idx_or_filter] > 0:
                        spp[i] = np.log10(spp_table[field_names[i]][idx_or_filter])
                else:
                    spp[i] = spp_table[field_names[i]][idx_or_filter]
            else:
                if do_log[i]:
                    value_filter = spp_table[field_names[i]][idx_or_filter] > 0
                    spp[value_filter,i] = np.log10(spp_table[field_names[i]][idx_or_filter][value_filter])
                else:
                    spp[:,i] = spp_table[field_names[i]][idx_or_filter]

    # Fix special cases
    if N == 1:
        for i in np.arange(num_spp_properties):
            if np.isnan(spp[i]):
                spp[i] = -99
    else:
        for i in np.arange(num_spp_properties):
            value_filter = np.isnan(spp[:,i])
            spp[value_filter,i] = -99

    ##############################
    # SPP Redshift
    ##############################

    if suffix is None:
        match catalog_data.catalog:
            case '3D-HST':
                z_spp = catalog_data.spp['z'][idx_or_filter]
            case 'UVISTA':
                z_spp = catalog_data.spp['z'][idx_or_filter]
            case 'KMOS3D' | 'DESI':
                z_spp = catalog_data.spp['Z'][idx_or_filter]
            case 'UVISTA-PLUS' | '3D-HST-PLUS':
                z_spp = catalog_data.best['z_spp'][idx_or_filter]
    else:
        z_spp = catalog_data.spp[f"z_spp_{suffix}"]

    if N == 1:
        if np.isnan(z_spp) or (z_spp < 0.0):
            z_spp = -99.0
    else:
        value_filter = np.isnan(z_spp) | (z_spp < 0.0)
        z_spp[value_filter] = -99.0

    if N == 1:
        spp = np.append(spp, z_spp)
    else:
        spp = np.hstack([spp, z_spp.reshape(-1,1)])

    spp_names = np.append(spp_names, 'z_spp')
    num_spp_properties += 1

    ##############################
    # Quality Flag
    ##############################

    # spp_flag =
    #    1: Reliable
    #    2: Less Reliable
    #    8: Inconsistent
    #    9: Unreliable
    #   98: Incomplete
    #   99: Not provided

    if suffix is None:
        match catalog_data.catalog:
            case 'UVISTA':
                # Also available: l68_lsfr, u68_lsfr, chi2
                if N == 1:
                    if spp_table['lmass'][idx_or_filter] > -99 or spp_table['lsfr'][idx_or_filter] > -99 or spp_table['Av'][idx_or_filter] > -99:
                        if catalog_data.sources['USE'][idx_or_filter] == 0:
                            spp_flags = 9
                        elif spp_table['lmass'][idx_or_filter] == -99 or spp_table['lsfr'][idx_or_filter] == -99 or spp_table['Av'][idx_or_filter] == -99:
                            spp_flags = 98
                        else:
                            spp_flags = 2
                else:
                    has_value = (spp_table['lmass'][idx_or_filter] > -99) | (spp_table['lsfr'][idx_or_filter] > -99) | (spp_table['Av'][idx_or_filter] > -99)
                    spp_flags[has_value & (catalog_data.sources['USE'][idx_or_filter] == 0)] = 9
                    spp_flags[has_value & (catalog_data.sources['USE'][idx_or_filter] == 1) & ((spp_table['lmass'][idx_or_filter] == -99) | (spp_table['lsfr'][idx_or_filter] == -99) | (spp_table['Av'][idx_or_filter] == -99))] = 98
                    spp_flags[has_value & (catalog_data.sources['USE'][idx_or_filter] == 1) & (spp_table['lmass'][idx_or_filter] > -99) & (spp_table['lsfr'][idx_or_filter] > -99) & (spp_table['Av'][idx_or_filter] > -99)] = 2

            case '3D-HST':
                # Also available: chi2
                if N == 1:
                    if spp_table['lmass'][idx_or_filter] > -99 or spp_table['lsfr'][idx_or_filter] > -99 or spp_table['av'][idx_or_filter] > -99:
                        if catalog_data.sources['use_phot'][idx_or_filter] == 0:
                            spp_flags = 9
                        elif spp_table['lmass'][idx_or_filter] == -99 or spp_table['lsfr'][idx_or_filter] == -99 or spp_table['av'][idx_or_filter] == -99:
                            spp_flags = 98
                        elif catalog_data.redshift['use_zgrism'][idx_or_filter] <= 0:
                            spp_flags = 2
                        else:
                            spp_flags = 1
                else:
                    has_value = (spp_table['lmass'][idx_or_filter] > -99) | (spp_table['lsfr'][idx_or_filter] > -99) | (spp_table['av'][idx_or_filter] > -99)
                    spp_flags[has_value & (catalog_data.sources['use_phot'][idx_or_filter] == 0)] = 9
                    spp_flags[has_value & (catalog_data.sources['use_phot'][idx_or_filter] == 1) & ((spp_table['lmass'][idx_or_filter] == -99) | (spp_table['lsfr'][idx_or_filter] == -99) | (spp_table['av'][idx_or_filter] == -99))] = 98
                    spp_flags[has_value & (catalog_data.sources['use_phot'][idx_or_filter] == 1) & (spp_table['lmass'][idx_or_filter] > -99) & (spp_table['lsfr'][idx_or_filter] > -99) & (spp_table['av'][idx_or_filter] > -99) & (catalog_data.redshift['use_zgrism'][idx_or_filter] <= 0)] = 2
                    spp_flags[has_value & (catalog_data.sources['use_phot'][idx_or_filter] == 1) & (spp_table['lmass'][idx_or_filter] > -99) & (spp_table['lsfr'][idx_or_filter] > -99) & (spp_table['av'][idx_or_filter] > -99) & (catalog_data.redshift['use_zgrism'][idx_or_filter] == 1)] = 1

            case 'KMOS3D':
                if N == 1:
                    spp_flags = 1
                else:
                    spp_flags[:] = 1

            case 'DESI':
                if N == 1:
                    if spp_table['SED_MASS'][idx_or_filter] > -99 or spp_table['SED_SFR'][idx_or_filter] > -99 or spp_table['SED_AV'][idx_or_filter] > -99:
                        if spp_table['SED_MASS'][idx_or_filter] == -99 or spp_table['SED_SFR'][idx_or_filter] == -99 or spp_table['SED_AV'][idx_or_filter] == -99:
                            spp_flags = 98
                        else:
                            spp_flags = 1
                else:
                    has_value = (spp_table['SED_MASS'][idx_or_filter] > -99) | (spp_table['SED_SFR'][idx_or_filter] > -99) | (spp_table['SED_AV'][idx_or_filter] > -99)
                    spp_flags[has_value & ((spp_table['SED_MASS'][idx_or_filter] == -99) | (spp_table['SED_SFR'][idx_or_filter] == -99) | (spp_table['SED_AV'][idx_or_filter] == -99))] = 98
                    spp_flags[has_value & (spp_table['SED_MASS'][idx_or_filter] > -99) & (spp_table['SED_SFR'][idx_or_filter] > -99) & (spp_table['SED_AV'][idx_or_filter] > -99)] = 1

            case 'UVISTA-PLUS' | '3D-HST-PLUS':
                spp_flags = catalog_data.best['spp_flag'][idx_or_filter]
    else:
        spp_flags = spp_table[f"flag_{suffix}"][idx_or_filter]

    # Fix special cases
    match catalog_data.catalog:
        case '3D-HST':
            if N == 1:
                if spp[3] == -1.0:
                    spp[:] = -99
                    spp_flags = 99
            else:
                value_filter = spp[:,3] == -1.0 # chi2 = -1
                spp[value_filter,:] = -99
                spp_flags[value_filter] = 99

    # Fix datatype issues
    if np.isscalar(spp_flags):
        spp_flags = np.array([spp_flags])

    return spp, spp_flags, spp_names

def get_spp_values(catalog_data, ids_or_filter):
    spp, _, spp_names = get_spp(catalog_data, ids_or_filter)

    lmass_idx = np.where(spp_names == 'lmass')[0]
    lsfr_idx = np.where(spp_names == 'lsfr')[0]
    Av_idx = np.where(spp_names == 'Av')[0]

    if np.ndim(spp) == 1:
        return spp[lmass_idx], spp[lsfr_idx], spp[Av_idx]
    else:
        return spp[:,lmass_idx], spp[:,lsfr_idx], spp[:,Av_idx]

#endregion

#region Lines Info

def get_line_sources():
    return np.array(['3DHST', 'MOSDEF', 'FMOS', 'KMOS3D', 'LEGAC', 'DESI','3DHSTP','UVISTAP'])

def get_line_names():
    return np.array(['SIIb', 'SIIa', 'NIIb', 'Ha', 'NIIa', 'OIIIb', 'OIIIa', 'Hb', 'OIIb', 'OIIa'])

def get_has_lines(catalog_data):
    return np.any(get_line_sources() == catalog_data.source)

def get_lines(catalog_data, idx_or_filter = None, suffix = None):
    line_names = get_line_names()
    num_lines = len(line_names)
    num_line_properties = 3

    if idx_or_filter is None:
        N = catalog_data.count
        idx_or_filter = np.ones((N), dtype=np.bool)
    elif np.size(idx_or_filter) == 1:
        if idx_or_filter == -1:
            return np.zeros((num_lines,num_line_properties)), 99 * np.ones((num_lines)), line_names
        N = 1
    elif issubclass(idx_or_filter.dtype.type, np.integer):
        N = len(idx_or_filter)
    else:
        N = np.sum(idx_or_filter)

    if N == 1:
        lines = np.zeros((num_lines,num_line_properties))
        line_flags = 99 * np.ones((num_lines))
    else:
        lines = np.zeros((N,num_lines,num_line_properties))
        line_flags = 99 * np.ones((N,num_lines))

    # Note:
    # Flux/Err: 10^-17 erg/s/cm^2
    # FWHM    : angstrom

    c = 299792.458 # km/s
    rest_lambda = sky.get_emission_line_rest_wavelengths()
    line_lambdas = sky.get_vacuum_to_air_wavelength([rest_lambda[line_name] for line_name in line_names])
    field_names = np.empty((num_lines, num_line_properties), dtype="<U30")
    field_unit_factors = np.ones((num_lines, num_line_properties))
    aperature_correction_fields = np.empty((num_lines), dtype="<U30")

    match catalog_data.catalog:
        case '3D-HST':
            line_prefixes = ['', '', '', 'Ha', '', 'OIII', '', 'Hb', 'OII', '']
            for i in np.arange(num_lines):
                if line_prefixes[i] != '':
                    field_names[i,0] = f"{line_prefixes[i]}_FLUX"
                    field_names[i,1] = f"{line_prefixes[i]}_FLUX_ERR"

                    field_unit_factors[i,0] = 1.0 # 10^-17 erg/s/cm^2
                    field_unit_factors[i,1] = 1.0 # 10^-17 erg/s/cm^2

        case 'MOSDEF':
            line_prefixes = get_MOSDEF_line_prefixes()
            for i in np.arange(num_lines):
                if line_prefixes[i] != '':
                    field_names[i,0] = f"{line_prefixes[i]}_PreferredFlux"
                    field_names[i,1] = f"{line_prefixes[i]}_PreferredFlux_Err"
                    field_names[i,2] = f"{line_prefixes[i]}_FWHM"

                    field_unit_factors[i,0] = 1e17 # erg/s/cm^2
                    field_unit_factors[i,1] = 1e17 # erg/s/cm^2
                    field_unit_factors[i,2] = 1.0 # angstrom

        case 'FMOS':
            #   - FLUX_*****: Emission-line fluxes are given in units of erg/s/cm2.
            #     Emission-line fluxes represent `in-fiber` observed values, and the associated errors
            #     are `formal` errors returned from MPFIT, but not including the uncertainties on
            #     the absolute flux calibration.
            #
            #   - The total fluxes may be obtained by multiplying the `in-fiber` values by the aperture
            #     correction factors:
            #     APERCORR_BEST_L1 for Halpha, [NII], [SII]
            #     APERCORR_BEST_L2 for Hbeta+[OIII]

            line_suffixes = ['', '', 'NII6584', 'HALPHA', '', 'OIII5007', '', 'HBETA', '', '']
            for i in np.arange(num_lines):
                if line_suffixes[i] != '':
                    field_names[i,0] = f"FLUX_{line_suffixes[i]}"
                    field_names[i,1] = f"FLUX_ERR_{line_suffixes[i]}"
                    field_names[i,2] = f"FWHMV_OBS_{line_suffixes[i]}"

                    field_unit_factors[i,0] = 1e17 # erg/s/cm^2
                    field_unit_factors[i,1] = 1e17 # erg/s/cm^2
                    field_unit_factors[i,2] = line_lambdas[i] / c # km/s

                    if i < 5:
                        aperature_correction_fields[i] = 'APERCORR_BEST_L1'
                    else:
                        aperature_correction_fields[i] = 'APERCORR_BEST_L2'

        case 'KMOS3D':
            line_suffixes = ['', '', '', 'HA', '', '', '', '', '', '']
            for i in np.arange(num_lines):
                if line_suffixes[i] != '':
                    field_names[i,0] = f"FLUX_{line_suffixes[i]}"
                    field_names[i,1] = f"FLUX_{line_suffixes[i]}_ERR"
                    field_names[i,2] = 'SIG'

                    field_unit_factors[i,0] = 1 # 10^-17? erg/s/cm^2
                    field_unit_factors[i,1] = 1 # 10^-17? erg/s/cm^2
                    field_unit_factors[i,2] = 2.355 # sigma -> FWHM

                    aperature_correction_fields[i] = 'FLUX_AP_CORR'

        case 'LEGAC':
            line_prefixes = ['', '', '', '', '', 'OIII_5007', 'OIII_4959', 'H_BETA', 'OII_3727', '']
            for i in np.arange(num_lines):
                if line_prefixes[i] != '':
                    field_names[i,0] = f"{line_prefixes[i]}_FLUX"
                    field_names[i,1] = f"{line_prefixes[i]}_FLUX_ERR"

                    field_unit_factors[i,0] = 1e-2 # 10^-19 erg/s/cm^2/A
                    field_unit_factors[i,1] = 1e-2 # 10^-19 erg/s/cm^2/A

        case 'DESI':
            line_prefixes = ['SII6731', 'SII6716', 'NII6583', 'HALPHA', 'NII6548', 'OIII5007', 'OIII4959', 'HBETA', 'OII3729', 'OII3726']
            for i in np.arange(num_lines):
                if line_prefixes[i] != '':
                    field_names[i,0] = f"{line_prefixes[i]}_FLUX"
                    field_names[i,1] = f"{line_prefixes[i]}_FLUXERR"

                    field_unit_factors[i,0] = 1.0 # 10^-17 erg/s/cm^2
                    field_unit_factors[i,1] = 1.0 # 10^-17 erg/s/cm^2

        case 'UVISTA-PLUS' | '3D-HST-PLUS':
            line_suffixes = line_names
            for i in np.arange(num_lines):
                field_names[i,0] = f"f_{line_suffixes[i]}"
                field_names[i,1] = f"e_{line_suffixes[i]}"
                field_names[i,2] = f"fwhm_{line_suffixes[i]}"

    if (catalog_data.catalog == 'UVISTA-PLUS' or catalog_data.catalog == '3D-HST-PLUS') and suffix is None:
        lines_table = catalog_data.best
    else:
        lines_table = catalog_data.lines

    if suffix is not None:
        for i in np.arange(num_lines):
            for j in np.arange(num_line_properties):
                if field_names[i,j] == '':
                    continue
                field_names[i,j] = f"{field_names[i,j]}_{suffix}"

    for i in np.arange(num_lines):
        if field_names[i,0] == '':
            continue

        ##############################
        # Line Flux
        ##############################

        for j in np.arange(num_line_properties):
            if field_names[i,j] == '':
                continue

            if (aperature_correction_fields[i] != '') and (j < 2):
                aperature_correction = lines_table[aperature_correction_fields[i]][idx_or_filter]
            else:
                aperature_correction = np.ones((N))

            if N == 1:
                lines[i,j] = np.maximum(0.0, lines_table[field_names[i,j]][idx_or_filter] * aperature_correction * field_unit_factors[i,j])
            else:
                lines[:,i,j] = np.maximum(0.0, lines_table[field_names[i,j]][idx_or_filter] * aperature_correction * field_unit_factors[i,j])

        # Fix special cases
        if N == 1:
            if np.isinf(lines[i,1]):
                lines[i,1] = 0.0
        else:
            value_filter = np.isinf(lines[:,i,1])
            lines[value_filter,i,1] = 0.0

        ##############################
        # Quality Flag
        ##############################

        # line_flags =
        #    1: Reliable Emission Line
        #    2: Less reliable Emission Line
        #    8: Unreliable Emission Line
        #    9: Contaminated Emission Line
        #   99: Not provided

        if catalog_data.catalog != 'UVISTA-PLUS':
            if N == 1:
                if lines[i,0] > 0.0 and lines[i,1] > 0.0 and not np.isinf(lines[i,1]):
                    if lines[i,0] / lines[i,1] > catalog_data.params.min_line_snr:
                        line_flags[i] = 1
                    else:
                        line_flags[i] = 2
            else:
                values_filter = (lines[:,i,0] > 0.0) & (lines[:,i,1] > 0.0) & ~np.isinf(lines[:,i,1])
                high_snr_filter = np.zeros(values_filter.shape, dtype=np.bool)
                high_snr_filter[values_filter] = lines[values_filter,i,0] / lines[values_filter,i,1] > catalog_data.params.min_line_snr
                line_flags[high_snr_filter, i] = 1
                line_flags[values_filter & ~high_snr_filter, i] = 2

        match catalog_data.catalog:
            case 'MOSDEF':
                sky_field_name = f"{line_prefixes[i]}_slflag"
                if N == 1:
                    if lines_table[sky_field_name][idx_or_filter] >= 0.2:
                        line_flags[i] = 9
                else:
                    # _slflag: flag indicating degree to which emission line is affected by nearby skylines;
                    # ***BEWARE OF THE FLUX IF SLFLAG >= 0.2***
                    value_filter = lines_table[sky_field_name][idx_or_filter] >= 0.2
                    line_flags[value_filter, i] = 9

            case 'KMOS3D':
                if N == 1:
                    if lines_table['FLAG'][idx_or_filter] == 0:
                        line_flags[i] = 1
                    elif lines_table['FLAG'][idx_or_filter] == -1 or lines_table['FLAG'][idx_or_filter] == -2:
                        line_flags[i] = 8
                    elif lines_table['FLAG'][idx_or_filter] == 1:
                        line_flags[i] = 9
                else:
                    value_filter = lines_table['FLAG'][idx_or_filter] == 0
                    line_flags[value_filter, i] = 1
                    value_filter = (lines_table['FLAG'][idx_or_filter] == -1) | (lines_table['FLAG'][idx_or_filter] == -2)
                    line_flags[value_filter, i] = 8
                    value_filter = lines_table['FLAG'][idx_or_filter] == 1
                    line_flags[value_filter, i] = 9

            case 'UVISTA-PLUS':
                if N == 1:
                    line_flags[i] = lines_table[f"flag_{line_suffixes[i]}"][idx_or_filter]
                else:
                    line_flags[:,i] = lines_table[f"flag_{line_suffixes[i]}"][idx_or_filter]

    if np.ndim(lines) == 2:
        lines = np.array([lines])
        line_flags = np.array([line_flags])

    return lines, line_flags, line_names

def get_line_flux(catalog_data, line_name, idx_or_filter = None, suffix = None, nan_if_empty=False):
    match line_name:
        case 'SII':
            retvals  = get_line_flux(catalog_data, 'SIIa', idx_or_filter, suffix, nan_if_empty=False) \
                     + get_line_flux(catalog_data, 'SIIb', idx_or_filter, suffix, nan_if_empty=False)
        case 'NII':
            retvals  = get_line_flux(catalog_data, 'NIIa', idx_or_filter, suffix, nan_if_empty=False) \
                     + get_line_flux(catalog_data, 'NIIb', idx_or_filter, suffix, nan_if_empty=False)
        case 'OIII':
            retvals  = get_line_flux(catalog_data, 'OIIIa', idx_or_filter, suffix, nan_if_empty=False) \
                     + get_line_flux(catalog_data, 'OIIIb', idx_or_filter, suffix, nan_if_empty=False)
        case 'OII':
            retvals  = get_line_flux(catalog_data, 'OIIa', idx_or_filter, suffix, nan_if_empty=False) \
                     + get_line_flux(catalog_data, 'OIIb', idx_or_filter, suffix, nan_if_empty=False)
        case _:
            lines, _, line_names = get_lines(catalog_data, idx_or_filter, suffix)
            line_idx = np.where(line_names == line_name)[0][0]
            if np.size(line_idx) == 1:
                retvals = lines[:,line_idx,0]
            else:
                retvals = np.zeros((np.size(lines,0)))

    if nan_if_empty:
        retvals[retvals == 0.0] = np.nan

    return retvals

def get_snr(f, e):
    snr = np.zeros((len(e)))
    snr[e>0] = f[e>0]/e[e>0]
    return snr

def get_MOSDEF_line_prefixes():
    return ['SII6733', 'SII6718', 'NII6585', 'Ha6565', 'NII6550', 'OIII5008', 'OIII4960', 'Hb4863', 'OII3730', 'OII3727']

#endregion

#region Region

def get_catalog_region(catalog_data, added_distance = 0.0*u.degree):
    master_ra_field = get_ra_field(catalog_data)
    master_dec_field = get_dec_field(catalog_data)

    ra_max = max(catalog_data.sources[master_ra_field]) * u.degree
    ra_min = min(catalog_data.sources[master_ra_field]) * u.degree
    dec_max = max(catalog_data.sources[master_dec_field]) * u.degree
    dec_min = min(catalog_data.sources[master_dec_field]) * u.degree

    mid_ra = (ra_min + ra_max) / 2
    mid_dec = (dec_min + dec_max) / 2

    width = ra_max - ra_min
    height = dec_max - dec_min

    if added_distance > 0.0:
        width += 2 * added_distance / np.cos(mid_dec.value / 180.0 * np.pi)
        height += 2 * added_distance

    return RectangleSkyRegion(SkyCoord(ra=mid_ra, dec=mid_dec), width, height)

#endregion

#region Consolidate

def consolidate_best_data(catalog_data):
    if not table.has_field(catalog_data, 'best'):
        catalog_data.best = Table()
        catalog_data.best.add_column(catalog_data.sources[get_id_field(catalog_data)], name='id')
        catalog_data.best.add_column(catalog_data.sources[get_ra_field(catalog_data)], name='ra')
        catalog_data.best.add_column(catalog_data.sources[get_dec_field(catalog_data)], name='dec')

    N = catalog_data.count

    # Step 1: Consolidate Redshift, Flag, Source
    z_best, z_best_flag = get_redshift_any(catalog_data, skip_z_merged=True)
    z_best_source = np.empty((N), dtype='<U20')
    z_best_source[z_best_flag < 99] = catalog_data.source

    redshift_sources = np.array(catalog_data.all_sources)
    redshift_sources_date = np.array(catalog_data.all_sources_date)

    if len(redshift_sources) > 1:
        z = -99.0 * np.ones((N,len(redshift_sources)))
        z_flag = 99 * np.ones((N,len(redshift_sources)), dtype=np.int_)
        z_source = np.empty((N,len(redshift_sources)), dtype='<U20')

        # Pack redshifts into np.arrays
        i = 0
        z[:,i] = z_best
        z_flag[:,i] = z_best_flag
        z_source[:,i] = z_best_source

        for j in np.arange(len(redshift_sources)):
            if table.has_field(catalog_data.redshift, f"z_{redshift_sources[j]}"):
                i += 1
                z[:,i] = catalog_data.redshift[f"z_{redshift_sources[j]}"]
                z_flag[:,i] = catalog_data.redshift[f"z_{redshift_sources[j]}_flag"]
                z_source[:,i] = redshift_sources[i]

        # Select best redshifts
        for i in np.arange(N):
            valid_filter = z[i,:] > 0.0
            if np.sum(valid_filter) == 0:
                continue

            is_inconsistent = False

            # Case 1: Lower z_flag
            min_z_flag = min(z_flag[i,valid_filter])
            num_min_z_flag = np.sum(z_flag[i,:] == min_z_flag)

            if num_min_z_flag == 1:
                selected_index = np.argmax(z_flag[i,:] == min_z_flag)
            else:
                z_filter = z_flag[i,:] == min_z_flag
                index_map = np.arange(len(redshift_sources))[z_filter]
                # Case 2: Same z_flag but later date
                selected_index = index_map[np.argmax(redshift_sources_date[z_filter])]

                z_mean = np.mean(z[i,z_filter])
                z_diff = np.std(z[i,z_filter])
                if z_diff/(1+z_mean) > 0.01:
                    is_inconsistent = True

            z_best[i] = z[i, selected_index]
            z_best_flag[i] = z_flag[i, selected_index]
            z_best_source[i] = z_source[i, selected_index]

            if z_best_flag[i] < 8 and is_inconsistent:
                z_best_flag[i] = 8

    table.add_fields(catalog_data.best, 'z_best', z_best)
    table.add_fields(catalog_data.best, 'z_best_flag', z_best_flag)
    table.add_fields(catalog_data.best, 'z_best_source', z_best_source)

    # Step 2: Consolidate use_phot
    if get_has_use_phot(catalog_data):
        use_phot_best = catalog_data.sources[get_use_phot_field(catalog_data)]

        use_phot_sources = get_use_phot_sources()
        for i in np.arange(len(use_phot_sources)):
            if use_phot_sources[i] == catalog_data.source:
                continue
            if table.has_field(catalog_data.sources, f"use_phot_{use_phot_sources[i]}"):
                value_filter = catalog_data.sources[f"use_phot_{use_phot_sources[i]}"] == 1
                use_phot_best[value_filter] = 1

        table.add_fields(catalog_data.best, 'use_phot', use_phot_best)

    # Step 3: Consolidate star_flag
    if get_has_star_flag(catalog_data):
        star_flag_best = catalog_data.sources[get_star_flag_field(catalog_data)]

        star_flag_sources = get_star_flag_sources()
        for i in np.arange(len(star_flag_sources)):
            if star_flag_sources[i] == catalog_data.source:
                continue
            if table.has_field(catalog_data.sources, f"star_flag_{star_flag_sources[i]}"):
                value_filter = catalog_data.sources[f"star_flag_{star_flag_sources[i]}"] == 1
                star_flag_best[value_filter] = 1

        table.add_fields(catalog_data.best, 'star_flag', star_flag_best)

    # Step 4: Consolidate magnitude
    if get_has_flux(catalog_data):
        mag_best = np.zeros((len(catalog_data.sources)))
        value_filter = catalog_data.sources[get_flux_field(catalog_data)] > 0.0
        mag_best[value_filter] = -2.5*np.log10(catalog_data.sources[get_flux_field(catalog_data)][value_filter]) + get_flux_zero_point(catalog_data)
        mag_source_best = np.empty((N), dtype='<U20')
        mag_source_best[value_filter] = catalog_data.source

        flux_sources = get_flux_sources()
        for i in np.arange(len(flux_sources)):
            if flux_sources[i] == catalog_data.source:
                continue
            if table.has_field(catalog_data.sources, f"mag_{flux_sources[i]}"):
                value_filter = (mag_best == 0.0) & (catalog_data.sources[f"mag_{flux_sources[i]}"] > 0.0)
                mag_best[value_filter] = catalog_data.sources[f"mag_{flux_sources[i]}"][value_filter]
                mag_source_best[value_filter] = flux_sources[i]

        table.add_fields(catalog_data.best, 'mag', mag_best)
        table.add_fields(catalog_data.best, 'mag_source', mag_source_best)

    # Step 5: Consolidate flux_radius
    if get_has_flux_radius(catalog_data):
        flux_radius_best = catalog_data.sources[get_flux_radius_field(catalog_data)] * get_flux_radius_factor(catalog_data)
        flux_radius_source_best = np.empty((N), dtype='<U20')
        flux_radius_source_best[flux_radius_best > 0.0] = catalog_data.source

        flux_radius_sources = get_flux_radius_sources()
        for i in np.arange(len(flux_radius_sources)):
            if flux_radius_sources[i] == catalog_data.source:
                continue
            if table.has_field(catalog_data.sources, f"flux_radius_{flux_radius_sources[i]}"):
                value_filter = (flux_radius_best == 0.0) & (catalog_data.sources[f"flux_radius_{flux_radius_sources[i]}"] > 0.0)
                flux_radius_best[value_filter] = catalog_data.sources[f"flux_radius_{flux_radius_sources[i]}"][value_filter]
                flux_radius_source_best[value_filter] = flux_radius_sources[i]

        table.add_fields(catalog_data.best, 'flux_radius', flux_radius_best)
        table.add_fields(catalog_data.best, 'flux_radius_source', flux_radius_source_best)

    # Step 6: Consolidate SPP
    if table.has_field(catalog_data, 'spp'):
        spp_best, spp_flag_best, spp_names = get_spp(catalog_data)
        spp_source_best = np.empty((N), dtype='<U20')
        spp_source_best[spp_flag_best < 99] = catalog_data.source

        # List of sources sorted by catalog date ASC
        spp_sources, spp_sources_date = _sort_by_catalog_date(catalog_data, get_spp_sources())
        spp_date_best = catalog_data.date

        for i in np.arange(len(spp_sources)):
            if spp_sources[i] == catalog_data.source:
                continue

            tmp_spp, tmp_spp_flags, _ = get_spp(catalog_data, suffix=spp_sources[i])

            # Case 1: Lower spp_flag
            better_flag_filter = tmp_spp_flags < spp_flag_best
            spp_best[better_flag_filter,:] = tmp_spp[better_flag_filter,:]
            spp_flag_best[better_flag_filter] = tmp_spp_flags[better_flag_filter]
            spp_source_best[better_flag_filter] = spp_sources[i]

            # Case 2: Same spp_flag but later date
            if spp_sources_date[i] > spp_date_best:
                later_flag_filter = ~better_flag_filter & (tmp_spp_flags < 99) & (tmp_spp_flags == spp_flag_best)
                spp_best[later_flag_filter,:] = tmp_spp[later_flag_filter,:]
                spp_flag_best[later_flag_filter] = tmp_spp_flags[later_flag_filter]
                spp_source_best[later_flag_filter] = spp_sources[i]
                spp_date_best = spp_sources_date[i]

        # Update spp_flag if z_spp not consistent with z_best
        z_spp_idx = np.argmax(np.array(spp_names) == 'z_spp')
        value_filter = (spp_flag_best < 9) & (spp_best[:,z_spp_idx] < 0.0)
        spp_flag_best[value_filter] = 9

        value_filter = (spp_flag_best < 8) & (z_best_flag < 99) & (np.abs(z_best-spp_best[:,z_spp_idx])/(1+np.maximum(0.0, z_best)) > 0.1)
        spp_flag_best[value_filter] = 8

        table.add_fields(catalog_data.best, spp_names, spp_best)
        table.add_fields(catalog_data.best, 'spp_flag', spp_flag_best)
        table.add_fields(catalog_data.best, 'spp_source', spp_source_best)

    # Step 7: Consolidate Lines
    if table.has_field(catalog_data, 'lines'):
        lines, line_flags, line_names = get_lines(catalog_data)

        # List of sources sorted by catalog date ASC
        line_sources, line_sources_date = _sort_by_catalog_date(catalog_data, get_line_sources())

        for i in np.arange(len(line_names)):
            line_date_best = catalog_data.date

            lines_best = lines[:,i,:].transpose().copy()
            lines_flag_best = line_flags[:,i].copy()
            lines_source_best = np.empty((N), dtype="<U20")
            if get_has_lines(catalog_data):
                lines_source_best[lines_flag_best < 99] = catalog_data.source

            best_snr = np.round(get_snr(lines_best[0,:], lines_best[1,:]), 1)

            for j in np.arange(len(line_sources)):
                if line_sources[j] == catalog_data.source:
                    continue

                f_field_name   = f"f_{line_names[i]}_{line_sources[j]}"
                e_field_name   = f"e_{line_names[i]}_{line_sources[j]}"
                fwhm_field_name = f"fwhm_{line_names[i]}_{line_sources[j]}"
                flag_field_name = f"flag_{line_names[i]}_{line_sources[j]}"

                if not table.has_field(catalog_data.lines, f_field_name):
                    continue

                tmp_f = catalog_data.lines[f_field_name]
                tmp_e = catalog_data.lines[e_field_name]
                tmp_snr = np.round(get_snr(tmp_f, tmp_e), 1)
                tmp_fwhm = catalog_data.lines[fwhm_field_name]
                tmp_flags = catalog_data.lines[flag_field_name]

                # Case 1: Lower Flag
                better_flag_filter = tmp_flags < lines_flag_best
                lines_best[0, better_flag_filter] = tmp_f[better_flag_filter]
                lines_best[1, better_flag_filter] = tmp_e[better_flag_filter]
                lines_best[2, better_flag_filter] = tmp_fwhm[better_flag_filter]
                lines_flag_best[better_flag_filter] = tmp_flags[better_flag_filter]
                lines_source_best[better_flag_filter] = line_sources[j]

                # Case 2: Higher S/N
                better_snr_filter = ~better_flag_filter & (tmp_snr > best_snr)
                lines_best[0, better_snr_filter] = tmp_f[better_snr_filter]
                lines_best[1, better_snr_filter] = tmp_e[better_snr_filter]
                lines_best[2, better_snr_filter] = tmp_fwhm[better_snr_filter]
                lines_flag_best[better_snr_filter] = tmp_flags[better_snr_filter]
                lines_source_best[better_snr_filter] = line_sources[j]

                # Case 3: Same lines_flag and similar S/N but later date
                if line_sources_date[j] > line_date_best:
                    later_flag_filter = ~better_flag_filter & (tmp_flags < 99) & (tmp_flags == lines_flag_best) & (tmp_snr == best_snr)
                    lines_best[0, later_flag_filter] = tmp_f[later_flag_filter]
                    lines_best[1, later_flag_filter] = tmp_e[later_flag_filter]
                    lines_best[2, later_flag_filter] = tmp_fwhm[later_flag_filter]
                    lines_flag_best[later_flag_filter] = tmp_flags[later_flag_filter]
                    lines_source_best[later_flag_filter] = line_sources[j]
                    line_date_best = line_sources_date[j]

                # TODO: Instead of replacing values, consider doing a weighted average

            table.add_fields(catalog_data.best, [f"f_{line_names[i]}", f"e_{line_names[i]}", f"fwhm_{line_names[i]}"], lines_best)
            table.add_fields(catalog_data.best, f"flag_{line_names[i]}", lines_flag_best)
            table.add_fields(catalog_data.best, f"source_{line_names[i]}", lines_source_best)

def _sort_by_catalog_date(catalog_data, sources):
    all_sources = np.array(catalog_data.all_sources)
    all_sources_date = np.array(catalog_data.all_sources_date)

    _, selected_sources_indexes, _ = np.intersect1d(catalog_data.all_sources, sources, return_indices=True)

    index_map = np.arange(len(all_sources))[selected_sources_indexes]
    sorted_indexes = index_map[np.argsort(all_sources_date[selected_sources_indexes])]

    return all_sources[sorted_indexes], all_sources_date[sorted_indexes]

#endregion

#region Save

def save_matched_catalog(catalog_data, new_catalog_params, save_format='ascii', new_name=None):
    if not os.path.exists(new_catalog_params.catalog_path):
        os.makedirs(new_catalog_params.catalog_path)

    if table.has_field(catalog_data, 'field') and catalog_data.field != '':
        field_suffix = f"_{catalog_data.field}"
    else:
        field_suffix = ''

    match save_format:
        case 'ascii':
            if new_name is not None:
                catalog_name = new_name
            else:
                catalog_name = catalog_data.catalog

            catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_name}{field_suffix}_sources.csv"
            astropy.io.ascii.write(catalog_data.sources, catalog_file_path, format='csv', overwrite=True)

            if table.has_field(catalog_data, 'idmap'):
                catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_name}{field_suffix}_idmap.csv"
                astropy.io.ascii.write(catalog_data.idmap, catalog_file_path, format='csv', overwrite=True)

            if table.has_field(catalog_data, 'redshift'):
                catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_name}{field_suffix}_redshift.csv"
                astropy.io.ascii.write(catalog_data.redshift, catalog_file_path, format='csv', overwrite=True)

            if table.has_field(catalog_data, 'spp'):
                catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_name}{field_suffix}_spp.csv"
                astropy.io.ascii.write(catalog_data.spp, catalog_file_path, format='csv', overwrite=True)

            if table.has_field(catalog_data, 'lines'):
                catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_name}{field_suffix}_lines.csv"
                astropy.io.ascii.write(catalog_data.lines, catalog_file_path, format='csv', overwrite=True)

            if table.has_field(catalog_data, 'best'):
                catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_name}{field_suffix}_best.csv"
                astropy.io.ascii.write(catalog_data.best, catalog_file_path, format='csv', overwrite=True)

            if table.has_field(catalog_data, 'clumps'):
                catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_name}{field_suffix}_clumps.csv"
                astropy.io.ascii.write(catalog_data.clumps, catalog_file_path, format='csv', overwrite=True)

        case 'pickel':
            if new_name is not None:
                catalog_data = deepcopy(catalog_data)
                catalog_data.catalog = new_name

            catalog_file_path = f"{new_catalog_params.catalog_path}/{catalog_data.catalog}{field_suffix}.pkl.gz"
            with open(catalog_file_path, 'wb') as f:
                dump(catalog_data, f, pickler_kwargs={"protocol": pickle.HIGHEST_PROTOCOL})

        case 'fits':
            raise CatalogException('Not implemented')
            # TODO: Add support for saving to FITS files

#endregion

#region Flatten

def flatten_galaxy_data(catalog_data):
    if catalog_data.catalog != '3D-HST' and not table.has_field(catalog_data, 'best'):
        raise CatalogException('Catalog does not support flattening')

    if catalog_data.catalog == '3D-HST':
        # (star_flag=0=extended source, star_flag=1=point source, star_flag=2=very faint objects that aren't clearly a star)
        galaxy_filter = catalog_data.sources['star_flag'] != 1
    else:
        galaxy_filter = catalog_data.best['star_flag'] != 1

    galaxy_count = np.sum(galaxy_filter)

    # Compile Galaxy Data
    source_id = catalog_data.sources[get_id_field(catalog_data)][galaxy_filter]
    ra = catalog_data.sources[get_ra_field(catalog_data)][galaxy_filter]
    dec = catalog_data.sources[get_dec_field(catalog_data)][galaxy_filter]
    x = catalog_data.sources[get_x_field(catalog_data)][galaxy_filter]
    y = catalog_data.sources[get_y_field(catalog_data)][galaxy_filter]

    if catalog_data.catalog == '3D-HST':
        flux = catalog_data.sources[get_flux_field(catalog_data)][galaxy_filter]
        mag = np.zeros((galaxy_count))
        value_filter = flux > 0.0
        mag[value_filter] = -2.5*np.log10(flux[value_filter]) + 25

        flux_radius = catalog_data.sources['kron_radius'][galaxy_filter] * get_flux_radius_factor(catalog_data) # pixels -> arcsec
        clump_fraction = -1 * np.ones((galaxy_count))
    else:
        mag = catalog_data.best['mag'][galaxy_filter]
        flux_radius = catalog_data.best['flux_radius'][galaxy_filter] # arcsec
        if table.has_field(catalog_data, 'clumps'):
            clump_fraction = catalog_data.clumps['UV_frac_clump'][galaxy_filter]
        else:
            clump_fraction = -1 * np.ones((galaxy_count))

    # Stellar Population Parameters
    spp, spp_flags, spp_names = get_spp(catalog_data, galaxy_filter)
    if catalog_data.catalog == '3D-HST':
        spp_sources = np.repeat('3DHST', (galaxy_count))
    else:
        spp_sources = catalog_data.best['spp_source'][galaxy_filter]

    # Emission Line Fluxes
    tmp_lines, tmp_line_flags, line_names = get_lines(catalog_data, galaxy_filter)
    line_names_map = np.array([
        ['Ha', 'OIIIb', 'Hb', 'OIIb'],
        ['Ha', 'OIII', 'Hb', 'OII']
    ])
    line_other_names = [[], ['OIIIa'], [], ['OIIa']]

    if catalog_data.catalog == '3D-HST':
        is_spec_low_res = True
    else:
        is_spec_low_res = False

    num_lines = line_names_map.shape[1]
    line_flux    = np.zeros((num_lines, galaxy_count))
    line_snr     = np.zeros((num_lines, galaxy_count))
    line_fwhm    = np.zeros((num_lines, galaxy_count))
    line_sources = np.empty((num_lines, galaxy_count), dtype='<U20')
    line_flags   = 99 * np.ones((num_lines, galaxy_count), dtype=np.int_)

    line_idx = -1
    idx_flux = 0
    idx_err = 1
    idx_fwhm = 2

    for i in np.arange(len(line_names)):
        name_idx = np.argmax(line_names_map[0,:] == line_names[i])
        if line_names_map[0,name_idx] != line_names[i]:
            continue
        line_idx += 1

        value_filter = (tmp_lines[:,i,idx_flux] > 0) & (tmp_lines[:,i,idx_err] > 0)

        line_flux[line_idx, value_filter] = tmp_lines[:,i,idx_flux][value_filter]
        line_snr[line_idx, value_filter] = line_flux[line_idx, value_filter] / tmp_lines[:,i,idx_err][value_filter]
        line_fwhm[line_idx, value_filter] = tmp_lines[:,i,idx_fwhm][value_filter]
        line_flags[line_idx, value_filter] = tmp_line_flags[value_filter,i]

        if catalog_data.catalog == '3D-HST':
            line_sources[line_idx,value_filter] = np.repeat('3DHST', (np.sum(value_filter)))
        else:
            line_sources[line_idx,value_filter] = catalog_data.best[f"source_{line_names[i]}"][galaxy_filter][value_filter]

        # Sum emission from doublets
        if len(line_other_names[line_idx]) > 0:
            for k in np.arange(len(line_other_names[line_idx])):
                for j in np.arange(len(line_names)):
                    if line_names[j] != line_other_names[line_idx][k]:
                        continue

                    # Case 1: Has both line i and j
                    doublet_value_filter = (tmp_lines[:,i,idx_flux] > 0) & (tmp_lines[:,i,idx_err] > 0) & (tmp_line_flags[:,i] < 99) & \
                                           (tmp_lines[:,j,idx_flux] > 0) & (tmp_lines[:,j,idx_err] > 0) & (tmp_line_flags[:,j] < 99)
                    line_flux[line_idx,doublet_value_filter] = tmp_lines[:,i,idx_flux][doublet_value_filter] + tmp_lines[:,j,idx_flux][doublet_value_filter]
                    line_snr[line_idx,doublet_value_filter] = line_flux[line_idx,doublet_value_filter] / (tmp_lines[:,i,idx_err][doublet_value_filter] + tmp_lines[:,j,idx_err][doublet_value_filter])
                    line_flags[line_idx,doublet_value_filter] = np.maximum(tmp_line_flags[:,i][doublet_value_filter], tmp_line_flags[:,j][doublet_value_filter])
                    # line_sources[idx,doublet_value_filter] <- keep value from line i

                    # Case 2: Has line j but not i
                    doublet_value_filter = ((tmp_lines[:,i,idx_flux] == 0) | (tmp_lines[:,i,idx_err] == 0) | (tmp_line_flags[:,i] == 99)) & \
                                           (tmp_lines[:,j,idx_flux] > 0) & (tmp_lines[:,j,idx_err] > 0) & (tmp_line_flags[:,j] < 99)
                    line_flux[line_idx,doublet_value_filter] = tmp_lines[:,j,idx_flux][doublet_value_filter]
                    line_snr[line_idx,doublet_value_filter] = line_flux[line_idx,doublet_value_filter] / tmp_lines[:,j,idx_err][doublet_value_filter]
                    line_flags[line_idx,doublet_value_filter] = tmp_line_flags[:,j][doublet_value_filter]
                    if catalog_data.catalog == '3D-HST':
                        line_sources[line_idx,doublet_value_filter] = '3DHST'
                    else:
                        line_sources[line_idx,doublet_value_filter] = catalog_data.best[f"source_{line_names[j]}"][galaxy_filter][doublet_value_filter]

                    # Case 3: Has line i but not j
                    # Nothing to do

        # Account for NII contamination
        if is_spec_low_res and line_names[i] == 'Ha':
            flux_NII_Ha_ratio = 0.10
            line_flux[line_idx,value_filter] *= 1 - flux_NII_Ha_ratio

    # Redshift
    z_best = -99.0 * np.ones((galaxy_count))
    z_best_flag = 99 * np.ones((galaxy_count), dtype=np.int_)
    z_best_source = np.empty((galaxy_count), dtype='<U20')

    if catalog_data.catalog == '3D-HST':
        use_phot = catalog_data.redshift['use_phot'][galaxy_filter] == 1
        use_grism = catalog_data.redshift['use_zgrism'][galaxy_filter] == 1
    else:
        use_phot = catalog_data.best['use_phot'][galaxy_filter] == 1

    if catalog_data.catalog == '3D-HST':
        use_spec = (catalog_data.redshift['z_best_s'][galaxy_filter] == 1) | ((catalog_data.redshift['z_best_s'][galaxy_filter] == 2) & use_grism)
    else:
        use_spec = catalog_data.best['z_best_flag'][galaxy_filter] <= 2

    if table.has_field(catalog_data.redshift, 'z_phot'):
        z_phot = catalog_data.redshift['z_phot'][galaxy_filter]
    elif table.has_field(catalog_data.redshift, 'z_peak'):
        z_phot = catalog_data.redshift['z_peak'][galaxy_filter]
    elif table.has_field(catalog_data.redshift, 'z_peak_phot'):
        z_phot = catalog_data.redshift['z_peak_phot'][galaxy_filter]
    else:
        z_phot = -99.0 * np.ones((galaxy_count))

    z_spec, z_spec_flag = get_redshift_spec(catalog_data, galaxy_filter)

    if catalog_data.catalog == '3D-HST':
        value_filter = catalog_data.redshift['z_best'][galaxy_filter] > 0.0
        z_best[value_filter]       = catalog_data.redshift['z_best'][galaxy_filter][value_filter]
        z_best_flag[value_filter] = z_spec_flag[value_filter]
        z_best_source[value_filter] = np.repeat('3DHST', (np.sum(value_filter)))
    else:
        z_best = catalog_data.best['z_best'][galaxy_filter]
        z_best_flag = catalog_data.best['z_best_flag'][galaxy_filter]
        z_best_source = catalog_data.best['z_best_source'][galaxy_filter]

    use = z_best > 0.0

    if table.has_field(catalog_data, 'idmap') and table.has_field(catalog_data.idmap, '3DHST'):
        other_id = catalog_data.idmap['3DHST'][galaxy_filter]
    else:
        other_id = -1 * np.ones((galaxy_count), dtype=np.int_)

    galaxy_data = Table([
            np.arange(galaxy_count), np.repeat(catalog_data.field, galaxy_count), source_id, ra, dec, x, y, mag, flux_radius, use,
            line_flux[0,:], line_snr[0,:], line_fwhm[0,:], line_flags[0,:], line_sources[0,:],
            line_flux[1,:], line_snr[1,:], line_fwhm[1,:], line_flags[1,:], line_sources[1,:],
            line_flux[2,:], line_snr[2,:], line_fwhm[2,:], line_flags[2,:], line_sources[2,:],
            line_flux[3,:], line_snr[3,:], line_fwhm[3,:], line_flags[3,:], line_sources[3,:],
            z_best, z_best_flag, z_best_source, z_spec, z_spec_flag, use_spec, z_phot, use_phot,
            spp[:,0], spp[:,1], spp[:,2], spp[:,3], spp_flags, spp_sources, clump_fraction, other_id
        ], names=[
            'index', 'field', 'phot_id', 'ra', 'dec', 'x', 'y', 'mag', 'flux_radius', 'use',
            f"flux_{line_names_map[1,0]}", f"snr_{line_names_map[1,0]}", f"fwhm_{line_names_map[1,0]}", f"flag_{line_names_map[1,0]}", f"source_{line_names_map[1,0]}",
            f"flux_{line_names_map[1,1]}", f"snr_{line_names_map[1,1]}", f"fwhm_{line_names_map[1,1]}", f"flag_{line_names_map[1,1]}", f"source_{line_names_map[1,1]}",
            f"flux_{line_names_map[1,2]}", f"snr_{line_names_map[1,2]}", f"fwhm_{line_names_map[1,2]}", f"flag_{line_names_map[1,2]}", f"source_{line_names_map[1,2]}",
            f"flux_{line_names_map[1,3]}", f"snr_{line_names_map[1,3]}", f"fwhm_{line_names_map[1,3]}", f"flag_{line_names_map[1,3]}", f"source_{line_names_map[1,3]}",
            'z_best', 'z_best_flag', 'z_best_source', 'z_spec', 'z_spec_flag', 'use_spec', 'z_phot', 'use_phot',
            spp_names[0], spp_names[1], spp_names[2], spp_names[3], 'spp_flag', 'spp_source', 'clump_fraction', 'other_id'
        ]
    )

    return galaxy_data

#endregion

#region Source Info

def get_source_info(catalog_name, catalog_id, reload=False, skip_cache=False):
    if np.size(catalog_name) > 1 or np.size(catalog_id) > 1:
        raise CatalogException('Only one source currently supported')

    if not skip_cache:
        cache_file = f"{get_default_data_path()}/cache/targets/source-{catalog_name}-{catalog_id}.pkl"

        if not reload and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return load(f)

    catalog_params = get_params(catalog_name)
    with CatalogData(catalog_params) as catalog_data:
        ids = _get_catalog_ids(catalog_data, catalog_id)
        zs = _get_catalog_redshifts(catalog_data, catalog_id)
        spps = _get_catalog_spp(catalog_data, catalog_id)
        lines = _get_catalog_lines(catalog_data, catalog_id)

    retvals = (ids, zs, spps, lines)

    if not skip_cache:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            dump(retvals, f, pickler_kwargs={"protocol": pickle.HIGHEST_PROTOCOL})

    return retvals

def _get_catalog_ids(catalog_data, catalog_id):
    catalogs = [catalog_data.source]
    ids = [catalog_id]

    if table.has_field(catalog_data, 'idmap'):
        idmap = catalog_data.idmap[catalog_data.get_index(catalog_id)] # pylint: disable=no-member
        for colname in idmap.colnames:
            if colname in catalog_data.all_sources and idmap[colname] >= 0:
                catalogs.append(colname)
                ids.append(idmap[colname]) # pylint: disable=no-member

    return Table([catalogs, ids], names=['catalog', 'id'])

def _get_catalog_redshifts(catalog_data, catalog_id):
    catalogs = []
    zs = []
    # TODO: add support for z_unc

    if table.has_field(catalog_data, 'redshift'):
        redshift = catalog_data.redshift[catalog_data.get_index(catalog_id)] # pylint: disable=no-member
        for colname in redshift.colnames:
            if 'z_' in colname and redshift[colname] > 0:
                source = colname.replace('z_', '')
                if source in catalog_data.all_sources:
                    catalogs.append(source)
                    zs.append(redshift[colname]) # pylint: disable=no-member

    return Table([catalogs, zs], names=['catalog', 'z'])

def _get_catalog_spp(catalog_data, catalog_id):
    if not table.has_field(catalog_data, 'spp'):
        return None

    catalog_names = np.asarray(get_spp_sources())
    spp_names = get_spp_names()
    spp_values = np.full((len(catalog_names), len(spp_names)), np.nan)

    for spp_name in spp_names:
        spp = catalog_data.spp[catalog_data.get_index(catalog_id)] # pylint: disable=no-member
        for i, spp_name in enumerate(spp_names):
            for colname in spp.colnames:
                if colname == spp_name:
                    if spp[colname] != -99:
                        source_idx = np.where(catalog_names == catalog_data.source)[0][0]
                        spp_values[source_idx,i] = spp[colname]
                elif f"{spp_name}_" in colname:
                    if spp[colname] != -99:
                        source_idx = np.where(catalog_names == colname.replace(f"{spp_name}_", ''))[0][0]
                        spp_values[source_idx,i] = spp[colname]

    nan_rows = np.all(np.isnan(spp_values), axis=1)

    if np.all(nan_rows):
        return None

    catalog_names = catalog_names[~nan_rows]
    spp_values = spp_values[~nan_rows]

    return Table(
        [catalog_names] + [spp_values[:,i] for i in range(np.size(spp_values,1))],
        names=['catalog'] + spp_names.tolist()
    )

def _get_catalog_lines(catalog_data, catalog_id):
    if not table.has_field(catalog_data, 'lines'):
        return None

    catalog_names = np.asarray(get_line_sources())
    line_names = get_line_names()
    line_fluxes = np.full((len(catalog_names), len(line_names)), np.nan)

    for line_name in line_names:
        lines = catalog_data.lines[catalog_data.get_index(catalog_id)] # pylint: disable=no-member
        for i, line_name in enumerate(line_names):
            for colname in lines.colnames:
                if f"f_{line_name}" in colname:
                    if lines[colname] > 0.0:
                        source_idx = np.where(catalog_names == colname.replace(f"f_{line_name}_", ''))[0][0]
                        line_fluxes[source_idx,i] = lines[colname] * 1e-17 # erg/s/cm^2

    nan_rows = np.all(np.isnan(line_fluxes), axis=1)

    if np.all(nan_rows):
        return None

    catalog_names = catalog_names[~nan_rows]
    line_fluxes = line_fluxes[~nan_rows]

    return Table(
        [catalog_names] + [line_fluxes[:,i] for i in range(np.size(line_fluxes,1))],
        names=['catalog'] + line_names.tolist()
    )

#endregion

#region Spectra

def get_spectra_sources():
    return np.array(['ZCB', '3DHST', 'VUDS', 'DEIMOS', 'MOSDEF', 'FMOS', 'KMOS3D', 'C3R2', 'LEGAC', 'DESI'])

def get_is_space_based(catalog_source):
    return catalog_source in ['3DHST']

def get_spectra(catalog_name, field, source_id, ra, dec):
    spectra = None

    match catalog_name:
        case 'ZCB':
            spectra = get_ZCB_spectra(source_id)
        case '3DHST':
            spectra = get_3DHST_spectra(source_id, field)
        case 'VUDS':
            spectra = get_VUDS_spectra(source_id, field)
        case 'DEIMOS':
            spectra = get_DEIMOS_spectra(source_id)
        case 'MOSDEF':
            spectra = get_MOSDEF_spectra(source_id, field)
        case 'FMOS':
            spectra = get_FMOS_spectra(source_id)
        # case 'KMOS3D':
        #     spectra = get_KMOS3D_spectra(source_id, field)
        case 'C3R2':
            spectra = get_C3R2_spectra(source_id, field)
        case 'LEGAC':
            spectra = get_LEGAC_spectra(source_id)
        case 'DESI':
            spectra = get_DESI_spectra(source_id, ra, dec)

    if spectra is not None:
        if isinstance(spectra, StructType):
            spectra = [spectra]

    return spectra

def get_ZCB_spectra(zcb_id):
    catalog_params = get_params('ZCOSMOS-BRIGHT')
    with CatalogData(catalog_params) as catalog_data:
        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(zcb_id)
        if np.size(idx) == 0:
            raise CatalogException(f"Missing id {zcb_id} in catalog ZCOSMOS-BRIGHT")
        filename = catalog_data.sources['FileName'][idx]

    cache_filename = f"{catalog_params.catalog_path}/spectra/{filename}"
    if not os.path.isfile(cache_filename):
        url = f"https://irsa.ipac.caltech.edu/data/COSMOS/spectra/z-cosmos/DR3/{filename}"
        if not files.download(url, cache_filename):
            raise CatalogException(f"Missing spectra file for {zcb_id} in catalog ZCOSMOS-BRIGHT")

    with fits.open(cache_filename) as hdul:
        spectrum = hdul[1].data # pylint: disable=no-member

        spectra = StructType()
        spectra.catalog_source = catalog_source
        spectra.catalog_name = catalog_name
        spectra.source_id = zcb_id
        spectra.source_name = f"{zcb_id:.0f}"
        spectra.wavelength = spectrum['WAVE'].flatten()
        spectra.flux = spectrum['FLUX_REDUCED'].flatten()

    return spectra

def get_3DHST_spectra(source_id, field_name):
    catalog_params = get_params('3D-HST', field_name)

    with CatalogData(catalog_params) as catalog_data:
        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(source_id)
        if np.size(idx) == 0:
            raise CatalogException(f"Missing id {source_id} in catalog 3D-HST")

        wfc3_grism_id = catalog_data.lines['grism_id'][idx]

    all_spectra = []

    for instrument in ['ACS', 'WFC3']:
        file_field_name = field_name.upper().replace('-', '')

        match instrument:
            case 'ACS':
                filename_1d = None
                filename_2d = None
                with open(f"{get_default_data_path()}/indexes/3D-HST-{file_field_name}_ACS_V4.1.5.txt", 'r', encoding='utf-8') as f:
                    for line in f:
                        if f"_{source_id:05}.1D.fits" in line:
                            filename_1d = line.split()[0]
                            if filename_2d is not None:
                                break
                        if f"_{source_id:05}.2D.fits" in line:
                            filename_2d = line.split()[0]
                            if filename_1d is not None:
                                break
                if filename_1d is None:
                    continue

                filename_matches = re.search(r'([^\/]+)\/1D\/FITS\/([^\.]+)', filename_1d)
                pointing_id = filename_matches.group(1)
                grism_id = filename_matches.group(2)

            case 'WFC3':
                pointing_id_match = re.search(r'([^-]+-\d+)-G141_', wfc3_grism_id)
                if pointing_id_match is None:
                    continue
                pointing_id = pointing_id_match.group(1)
                grism_id = wfc3_grism_id
                filename_1d = f"{pointing_id}/1D/FITS/{grism_id}.1D.fits"
                filename_2d = f"{pointing_id}/2D/FITS/{grism_id}.2D.fits"

        cache_filename_1d = f"{catalog_params.catalog_path}/spectra/{instrument}/{filename_1d}"
       #cache_filename_2d = f"{catalog_params.catalog_path}/spectra/{instrument}/{filename_2d}"
        cache_tar_filename = f"{catalog_params.catalog_path}/spectra/{instrument}/{pointing_id}.tar.gz"

        if not os.path.isfile(cache_filename_1d):
            if not os.path.isfile(cache_tar_filename):
                url = f"https://archive.stsci.edu/missions/hlsp/3d-hst/RELEASE_V4.1.5/{file_field_name}/{file_field_name}_{instrument}_V4.1.5/{pointing_id}.tar.gz"
                if not files.download(url, cache_tar_filename):
                    raise CatalogException(f"Missing {instrument} spectra file for {source_id} in catalog 3D-HST")

            os.makedirs(os.path.dirname(cache_filename_1d), exist_ok=True)

            with tarfile.open(cache_tar_filename, 'r:gz') as tar:
                print(f"Extracting spectra from {os.path.basename(cache_tar_filename)}...", flush=True)
                members = [m for m in tar.getmembers() if m.name in [filename_1d, filename_2d]]
                tar.extractall(members=members, path=f"{catalog_params.catalog_path}/spectra/{instrument}", filter='data')

        with fits.open(cache_filename_1d) as hdul:
            spectrum = hdul[1].data # pylint: disable=no-member
            spectra = StructType()
            spectra.catalog_source = catalog_source
            spectra.catalog_name = catalog_name
            spectra.source_id = source_id
            spectra.source_name = f"{instrument} {grism_id}"
            spectra.wavelength = spectrum['wave'].flatten()
            spectra.flux_1d = spectrum['flux'].flatten()
            # TODO: add support for 2D spectrum
            spectra.zero_padded = True
            all_spectra.append(spectra)

    if len(all_spectra) == 0:
        return None

    return all_spectra

def get_VUDS_spectra(source_id, field_name):
    catalog_params = get_params('VUDS', field_name)
    with CatalogData(catalog_params) as catalog_data:
        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(source_id)
        if np.size(idx) == 0:
            raise CatalogException(f"Missing id {source_id} in catalog VUDS")

        vuds_id = catalog_data.sources['id'][idx]
        mask = catalog_data.sources['Mask'][idx]
        slit = catalog_data.sources['slit'][idx]
        obj = catalog_data.sources['obj'][idx]

    match field_name:
        case 'COSMOS':
            file_field_name = 'COSMOS'
        case 'GOODS-S':
            file_field_name = 'ECDFS'

    all_spectra = []

    spectra_path = f"{catalog_params.catalog_path}/spectra/VUDS-{file_field_name}-DR1/spec1d"
    if not os.path.isdir(spectra_path):
        raise CatalogException(f"Missing spectra file for {source_id} in catalog VUDS. Download from: https://data.lam.fr/vuds/download")

    for letter in ['A', 'B']:
        spectra_name = f"sc_{vuds_id}_{mask}_join_{letter}_{slit}_{obj}"
        cache_filename = f"{spectra_path}/{spectra_name}_atm_clean.fits"
        if not os.path.isfile(cache_filename):
            continue

        with fits.open(cache_filename) as hdul:
            header = hdul[0].header # pylint: disable=no-member
            spectra = StructType()
            spectra.catalog_source = catalog_source
            spectra.catalog_name = catalog_name
            spectra.source_id = source_id
            spectra.source_name = spectra_name
            spectra.wavelength = header["CRVAL1"] + header["CDELT1"] * np.arange(header["NAXIS1"])
            spectra.flux_1d = hdul[0].data.flatten() # pylint: disable=no-member
            # TODO: Add support for 2D spectrum
            all_spectra.append(spectra)

    return all_spectra

def get_DEIMOS_spectra(source_id):
    catalog_params = get_params('DEIMOS')
    with CatalogData(catalog_params) as catalog_data:
        index_filename = f"{catalog_params.catalog_path}/deimos_redshift_linksIRSA.tbl"

        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(source_id)
        file_id = catalog_data.sources['ID'][idx]

    if not os.path.isfile(index_filename):
        url = 'https://irsa.ipac.caltech.edu/data/COSMOS/spectra/deimos/deimos_redshift_linksIRSA.tbl'
        if not files.download(url, index_filename):
            raise CatalogException('Missing index file for catalog DEIMOS')

    file_index = astropy.io.ascii.read(index_filename)
    file_index = file_index[file_index['ID'] == file_id]

    if len(file_index) == 0 or file_index['fits1d'].filled('') == '':
        return None

    url = f"https://irsa.ipac.caltech.edu{re.search(r'href=\"([^\"]+)\"', file_index['fits1d'][0]).group(1)}"
    cache_filename = f"{catalog_params.catalog_path}/spectra/{os.path.basename(url)}"
    if not os.path.isfile(cache_filename):
        if not files.download(url, cache_filename):
            raise CatalogException(f"Missing spectra file for {source_id} in catalog DEIMOS")

    with fits.open(cache_filename) as hdul:
        spectrum = hdul[1].data # pylint: disable=no-member
        spectra = StructType()
        spectra.catalog_source = catalog_source
        spectra.catalog_name = catalog_name
        spectra.source_id = source_id
        spectra.source_name = file_id
        spectra.wavelength = spectrum['LAMBDA'].flatten()
        spectra.flux_1d = spectrum['FLUX'].flatten()
        # TODO: Add support for 2D spectrum

    return spectra

def get_MOSDEF_spectra(source_id, field_name):
    catalog_params = get_params('MOSDEF', field_name)
    with CatalogData(catalog_params) as catalog_data:
        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(source_id)
        if np.size(idx) == 0:
            raise CatalogException(f"Missing id {source_id} in catalog MOSDEF")

        mask_name = catalog_data.sources['MASKNAME'][idx]
        slit_object_name = catalog_data.sources['SLITOBJNAME'][idx]
        aperture_no = catalog_data.sources['APERTURE_NO'][idx]

    field_prefix = mask_name[0:2]
    spectra_path = f"{catalog_params.catalog_path}/spectra/{field_prefix}_1dspec"
    if not os.path.isdir(spectra_path):
        raise CatalogException(f"Missing spectra file for {source_id} in catalog MOSDEF. Download from: https://mosdef.astro.berkeley.edu/for-scientists/data-releases/")

    all_spectra = []

    for band in ['Y', 'J', 'H', 'K']:
        if aperture_no == 1:
            aperture_suffix = ''
        else:
            aperture_suffix = f'.{aperture_no}'

        spectra_name = f"{mask_name}.{band}.{slit_object_name}{aperture_suffix}"
        cache_filename = f"{spectra_path}/{spectra_name}.ell.1d.fits"
        if os.path.isfile(cache_filename):
            with fits.open(cache_filename) as hdul:
                header = hdul[1].header # pylint: disable=no-member
                spectra = StructType()
                spectra.catalog_source = catalog_source
                spectra.catalog_name = catalog_name
                spectra.source_id = source_id
                spectra.source_name = spectra_name
                spectra.wavelength = header["CRVAL1"] + header["CDELT1"] * np.arange(header["NAXIS1"])
                spectra.flux_1d = hdul[1].data.flatten() # pylint: disable=no-member
                # TODO: Add support for 2D spectrum
                spectra.zero_padded = True
                all_spectra.append(spectra)

    return all_spectra

def get_FMOS_spectra(source_id):
    catalog_params = get_params('FMOS')
    with CatalogData(catalog_params) as catalog_data:
        if not os.path.isdir(f"{catalog_params.catalog_path}/FitsFiles"):
            raise CatalogException('Missing FitsFiles directory in catalog FMOS, download from: https://member.ipmu.jp/fmos-cosmos/FC_spectra.html')

        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(source_id)
        if np.size(idx) == 0:
            raise CatalogException(f"Missing id {source_id} in catalog FMOS")

        name = catalog_data.sources['FMOS_ID'][idx]
        obs_date_hl = catalog_data.sources['OBS_DATE_HL'][idx]
        obs_date_hs = catalog_data.sources['OBS_DATE_HS'][idx]
        obs_date_jl = catalog_data.sources['OBS_DATE_JL'][idx]

    obs_date = max(obs_date_hl, obs_date_hs, obs_date_jl)
    if obs_date == -99:
        raise CatalogException(f"Missing OBS_DATE for {name} in catalog FMOS")

    try:
        filename_1d = glob.glob(f"{catalog_params.catalog_path}/FitsFiles/{obs_date}/{name}*1d.fits")[0]
    except IndexError as e:
        raise CatalogException(f"Missing 1D spectra file for {name} in catalog FMOS") from e

    with fits.open(filename_1d) as hdul:
        flux_1d = hdul[0].data # pylint: disable=no-member
        header = hdul[0].header # pylint: disable=no-member

    try:
        filename_2d = glob.glob(f"{catalog_params.catalog_path}/FitsFiles/{obs_date}/{name}*2d.fits")[0]
    except IndexError as e:
        raise CatalogException(f"Missing 2D spectra file for {name} in catalog FMOS") from e

    with fits.open(filename_2d) as hdul:
        flux_2d = hdul[0].data # pylint: disable=no-member

    wavelength = header["CRVAL1"] + header["CD1_1"] * np.arange(header["NAXIS1"])

    spectra = StructType()
    spectra.catalog_source = catalog_source
    spectra.catalog_name = catalog_name
    spectra.source_id = source_id
    spectra.source_name = name
    spectra.wavelength = wavelength
    spectra.flux_1d = flux_1d
    spectra.flux_2d = flux_2d

    return spectra

def get_KMOS3D_spectra(source_id, field_name):
    # https://www.mpe.mpg.de/ir/KMOS3D/data
    raise CatalogException('Not implemented')

def get_C3R2_spectra(source_id, field_name):
    catalog_params = get_params('C3R2', field_name)
    with CatalogData(catalog_params) as catalog_data:
        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(source_id)
        filename = catalog_data.sources['spec1dfile'][idx]

    if filename == '-1':
        return None

    cache_filename = f"{catalog_params.catalog_path}/spectra/{filename}"
    if not os.path.isfile(cache_filename):
        url = f"https://koa.ipac.caltech.edu/data/Contributed/C3R2/spec1d/{filename}"
        if not files.download(url, cache_filename):
            raise CatalogException(f"Missing spectra file for {source_id} in catalog C3R2")

    with fits.open(cache_filename) as hdul:
        spectrum = hdul[1].data # pylint: disable=no-member
        spectra = StructType()
        spectra.catalog_source = catalog_source
        spectra.catalog_name = catalog_name
        spectra.source_id = source_id
        spectra.source_name = str(source_id)
        spectra.wavelength = spectrum['LAMBDA'].flatten()
        spectra.flux_1d = spectrum['FLUX'].flatten()
        # TODO: Add support for 2D spectrum

    return spectra

def get_LEGAC_spectra(source_id):
    catalog_params = get_params('LEGAC')
    with CatalogData(catalog_params) as catalog_data:
        spectra_path = f"{catalog_params.catalog_path}/spectra"
        if not os.path.isdir(spectra_path):
            raise CatalogException('Missing spectra directory in catalog LEGAC, download files from: https://archive.eso.org/wdb/wdb/adp/phase3_spectral/query?collection_name=LEGA-C&max_rows_returned=5000 then rename files using "../scripts/rename-LEGAC-spectra.sh".')

        catalog_source = catalog_data.source
        catalog_name = catalog_data.name
        idx = catalog_data.get_index(source_id)
        if np.size(idx) == 0:
            raise CatalogException(f"Missing id {source_id} in catalog LEGAC")

        file_id = int(catalog_data.sources['ID'][idx])

    filenames = np.array([f for f in os.listdir(spectra_path) if f.startswith('legac')])
    file_ids = np.array([int(re.search(r'legac_[^_]+_(\d+)_', s).group(1)) for s in filenames])
    file_filter = file_ids == file_id
    filenames = filenames[file_filter]

    if np.size(filenames) == 0:
        return None

    spectra = []

    for i, filename in enumerate(filenames):
        with fits.open(f"{catalog_params.catalog_path}/spectra/{filename}") as hdul:
            data = hdul[1].data # pylint: disable=no-member
            spectrum = StructType()
            spectrum.catalog_source = catalog_source
            spectrum.catalog_name = catalog_name
            spectrum.source_id = source_id
            spectrum.source_name = filenames[i]
            spectrum.wavelength = data['WAVE'].flatten()
            spectrum.flux = data['FLUX'].flatten()
            spectra.append(spectrum)

    return spectra

DESI_HDUL_FIBREMAP = 1
DESI_BANDS = ['b', 'r', 'z']
DESI_HDUL_BAND_START = [3, 8, 13]

def get_DESI_spectra(desi_id, ra, dec):
    catalog_params = get_params('DESI')

    surveys = ['main', 'sv3', 'sv2', 'sv1']
    programs = ['dark', 'bright', 'backup']
    pixnum = healpix.get_healpix_from_skycoord(6, SkyCoord(ra=ra, dec=dec, unit=(u.degree,u.degree))) # 2**6 = nside=64
    pixgroup = pixnum // 100

    cache_path = f"{catalog_params.catalog_path}/spectra/{pixgroup}/{pixnum}"
    if not os.path.isdir(cache_path):
        tmp_cache_path = f"{cache_path}.part"
        os.makedirs(tmp_cache_path, exist_ok=True)
        for survey in surveys:
            for program in programs:
                url = f"https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/{survey}/{program}/{pixgroup}/{pixnum}/coadd-{survey}-{program}-{pixnum}.fits"
                cache_filename = f"{tmp_cache_path}/coadd-{survey}-{program}-{pixnum}.fits"
                if not os.path.isfile(cache_filename):
                    files.download(url, cache_filename)
        os.rename(tmp_cache_path, cache_path)

    spectra = []

    for survey in surveys:
        for program in programs:
            cache_filename = f"{cache_path}/coadd-{survey}-{program}-{pixnum}.fits"
            if os.path.isfile(cache_filename):
                with fits.open(cache_filename) as hdul:
                    fibremap = hdul[DESI_HDUL_FIBREMAP].data # pylint: disable=no-member
                    idx = np.where(fibremap['TARGETID'] == desi_id)[0]
                    if np.size(idx) == 1:
                        for i, band in enumerate(DESI_BANDS):
                            spectrum = StructType()
                            spectrum.catalog_source = 'DESI'
                            spectrum.catalog_name = 'DESI'
                            spectrum.source_id = desi_id
                            spectrum.source_name = f"{desi_id:.0f} {band}-band"
                            spectrum.wavelength = hdul[DESI_HDUL_BAND_START[i]].data
                            spectrum.flux = hdul[DESI_HDUL_BAND_START[i]+1].data[idx,:].flatten()
                            spectra.append(spectrum)
                    elif np.size(idx) > 1:
                        raise CatalogException(f"Multiple indices {idx} in {cache_filename}")

    if len(spectra) == 0:
        raise CatalogException(f"Missing spectra for {desi_id} in catalog DESI")

    return spectra

#endregion

#region Utilities

def compute_kron_radius(n, r=1e10, Re=1):
    # see https://arxiv.org/pdf/astro-ph/9911078.pdf Eq. 18
    bn = 2*n - 1/3 + 4/405/n + 46/25515/n**2 + 131/1148175/n**3 - 2194697/30690717750/n**4

    # see https://arxiv.org/pdf/astro-ph/0503176.pdf Eq. 32
    x = bn * np.power(r / Re, 1 / n)
    krad = Re / np.power(bn, n) * (gamma(3*n)*gammainc(3*n, x)) / (gamma(2*n)*gammainc(2*n, x))

    return krad

def compute_flux_Ha_from_SFR(sfr, Av, z, null_if_empty=False):
    # NOTE: sfr is in Msun/yr NOT log10(Msun/yr)

    return_value = np.ndim(sfr) == 0
    sfr = np.asarray(sfr).flatten()
    Av = np.asarray(Av).flatten()
    z = np.asarray(z).flatten()

    # From Kennicutt (1998) ApJ 498, 541
    # SFR = 7.9e-42 * L(Hα)                                 [Msun/yr]
    # L(Hα) = 4π dL^2 * Fcorr(Hα)                           [erg/s]
    # dL(z) = luminosity distance at  z                     [cm]
    # Fcorr(Hα) = Fobs(Hα) * 10^(0.4*A(Hα))                 [erg/s/cm^2]
    # A(Hα) = Av * R_Hα                                     [mag]
    # R(1/Hα) = extinction curve at Hα                      [factor]
    # Fobs(Ha) = SFR / 7.9e-42 / 4π dL^2 / 10^(0.4*A(Ha))   [erg/s/cm^2]

    extinction_model = F99(Rv=3.1)
    lambda_Ha_rest = sky.get_emission_line_rest_wavelengths()['Ha']
    R_Ha =  extinction_model(1/(lambda_Ha_rest/1e4*u.micron))
    A_Ha = Av * R_Ha

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dL = cosmo.luminosity_distance(z)

    flux_Ha = sfr / 7.9e-42 / (4*np.pi*np.power(dL.to(u.cm).value, 2)) / np.power(10, 0.4*A_Ha) # erg/s/cm^2

    if null_if_empty:
        flux_Ha[flux_Ha == 0] = np.nan

    if return_value:
        return flux_Ha[0]
    else:
        return flux_Ha

def calculate_flux_Ha_from_flux_Hb(flux_Hb, Av, null_if_empty=False):
    return_value = np.ndim(flux_Hb) == 0
    flux_Hb = np.asarray(flux_Hb).flatten()
    Av = np.asarray(Av).flatten()

    rest_lambdas = sky.get_emission_line_rest_wavelengths()
    extinction_model = F99(Rv=3.1)

    lambda_Ha_rest = rest_lambdas['Ha']
    R_Ha =  extinction_model(1/(lambda_Ha_rest/1e4*u.micron))
    A_Ha = Av * R_Ha

    lambda_Hb_rest = rest_lambdas['Hb']
    R_Hb =  extinction_model(1/(lambda_Hb_rest/1e4*u.micron))
    A_Hb = Av * R_Hb

    # Assume Balmer decrement of 2.86
    flux_Ha = 2.86 * flux_Hb * np.power(10, 0.4*(A_Hb-A_Ha))

    if null_if_empty:
        flux_Ha[flux_Ha == 0] = np.nan

    if return_value:
        return flux_Ha[0]
    else:
        return flux_Ha

def calculate_flux_Ha_from_flux_OIII(flux_OIII, Av, null_if_empty=False):
    return_value = np.ndim(flux_OIII) == 0
    flux_OIII = np.asarray(flux_OIII).flatten()
    Av = np.asarray(Av).flatten()

    rest_lambdas = sky.get_emission_line_rest_wavelengths()
    extinction_model = F99(Rv=3.1)

    lambda_Ha_rest = rest_lambdas['Ha']
    R_Ha =  extinction_model(1/(lambda_Ha_rest/1e4*u.micron))
    A_Ha = Av * R_Ha

    lambda_OII_rest = np.mean([rest_lambdas['OIIIa'], rest_lambdas['OIIIb']])
    R_OIII =  extinction_model(1/(lambda_OII_rest/1e4*u.micron))
    A_OIII = Av * R_OIII

    # Assume flux_OIII_corr / flux_OIII_corr ~ 4 and Balmer decrement of 2.86
    flux_Ha = 2.86 / 4.0 * flux_OIII * np.power(10, 0.4*(A_OIII-A_Ha))

    if null_if_empty:
        flux_Ha[flux_Ha == 0] = np.nan

    if return_value:
        return flux_Ha[0]
    else:
        return flux_Ha

def calculate_flux_Ha_from_flux_OII(flux_OII, Av, null_if_empty=False):
    return_value = np.ndim(flux_OII) == 0
    flux_OII = np.asarray(flux_OII).flatten()
    Av = np.asarray(Av).flatten()

    rest_lambdas = sky.get_emission_line_rest_wavelengths()
    extinction_model = F99(Rv=3.1)

    lambda_Ha_rest = rest_lambdas['Ha']
    R_Ha =  extinction_model(1/(lambda_Ha_rest/1e4*u.micron))
    A_Ha = Av * R_Ha

    lambda_OII_rest = np.mean([rest_lambdas['OIIa'], rest_lambdas['OIIb']])
    R_OII =  extinction_model(1/(lambda_OII_rest/1e4*u.micron))
    A_OII = Av * R_OII

    # Assume flux_Ha_corr ~ flux_OII_corr
    flux_Ha = 1.0 * flux_OII * np.power(10, 0.4*(A_OII-A_Ha))

    if null_if_empty:
        flux_Ha[flux_Ha == 0] = np.nan

    if return_value:
        return flux_Ha[0]
    else:
        return flux_Ha


#endregion
