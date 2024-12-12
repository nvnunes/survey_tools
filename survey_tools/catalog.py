#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long,broad-exception-raised
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from copy import deepcopy
from datetime import datetime
import os
import pathlib
import pickle
from compress_pickle import dump, load
import numpy as np
from astropy.coordinates import Angle, SkyCoord
import astropy.io
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u
from astropy.wcs import WCS
from regions import RectangleSkyRegion
from scipy.special import gamma, gammainc # pylint: disable=no-name-in-module
from survey_tools import sky
from survey_tools.utility import table

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

def _get_default_catalog_path():
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

    catalog_params.catalog_data_path  = _get_default_catalog_path()
    catalog_params.catalog_image_path = _get_default_image_path()

    match catalog_name:
        case 'UVISTA':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/UVISTA"
        case 'ZCOSMOS-BRIGHT':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/zCOSMOS"
        case 'ZCOSMOS-DEEP':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/zCOSMOS"
        case '3D-HST':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/3D-HST"
        case 'VUDS':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/VUDS"
        case 'Casey':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/Casey"
        case 'DEIMOS':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/DEIMOS"
        case 'MOSDEF':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/MOSDEF"
        case 'FMOS':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/FMOS"
        case 'KMOS3D':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/KMOS3D"
        case 'C3R2':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/C3R2"
        case 'LEGAC':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/LEGA-C"
        case 'HELP':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/HELP"
        case 'DESI':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/DESI"
        case 'UVISTA-PLUS':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/UVISTA-PLUS"
        case '3D-HST-PLUS':
            catalog_params.catalog_path = f"{catalog_params.catalog_data_path}/3D-HST-PLUS"

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
                    match filter_name:
                        case 'J':
                            filter_string = '152'
                        case 'H':
                            filter_string = '149'
                        case 'K':
                            filter_string = '155'
                case '3D-HST' | '3D-HST-PLUS':
                    filter_string = filter_name

            match catalog_name:
                case 'UVISTA' | 'UVISTA-PLUS':
                    catalog_params.catalog_image_folder = f"{catalog_params.catalog_image_path}/UVISTA"
                    catalog_params.catalog_image_file = f"ADP.2023-05-02T10_57_31.{filter_string}.fits"
                case '3D-HST' | '3D-HST-PLUS':
                    catalog_params.catalog_image_folder = f"{catalog_params.catalog_image_path}/3D-HST"
                    catalog_params.catalog_image_file = f"{catalog_params.field_file_prefix}_3dhst.v4.0.{filter_string}_orig_sci.fits"

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

#endregion

#region Read

# pylint: disable=no-member
def read(catalog_params, force_tables = False, region = None, region_wcs = None):
    catalog_data = StructType()
    catalog_data.params = catalog_params
    catalog_data.catalog = catalog_params.catalog
    catalog_data.field = catalog_params.field
    catalog_data.hduls = []

    catalog_data.redshift_is_copy = False
    catalog_data.spp_is_copy = False
    catalog_data.lines_is_copy = False
    catalog_data.clumps_is_copy = False

    match catalog_params.catalog:
        case 'UVISTA':
            catalog_data.source = 'UVISTA'
            catalog_data.name = 'COSMOS/UltraVISTA'
            catalog_data.date = datetime(2015, 8, 20)
            catalog_data.frame = 'fk5'

            # Columns: id ra dec xpix ypix Ks_tot eKs_tot Ks eKs H eH J eJ Y eY ch4 ech4 ch3 ech3 ch2 ech2 ch1 ech1 zp ezp ip eip rp erp V eV gp egp B eB u eu IA484 eIA484 IA527 eIA527 IA624 eIA624 IA679 eIA679 IA738 eIA738 IA767 eIA767 IB427 eIB427 IB464 eIB464 IB505 eIB505 IB574 eIB574 IB709 eIB709 IB827 eIB827 fuv efuv nuv enuv mips24 emips24 K_flag K_star K_Kron apcor z_spec z_spec_cc z_spec_id star contamination nan_contam orig_cat_id orig_cat_field USE
            file_path = f"{catalog_params.catalog_path}/UVISTA_final_v4.1.cat"
            include_names = ['id', 'ra', 'dec', 'xpix', 'ypix', 'Ks_tot', 'eKs_tot', 'Ks', 'eKs', 'H', 'eH', 'J', 'eJ', 'K_flag', 'K_star', 'K_Kron', 'z_spec', 'z_spec_cc', 'z_spec_id', 'star', 'contamination', 'USE']
            catalog_data.sources = astropy.io.ascii.read(file_path, include_names=include_names)
            catalog_data.count = len(catalog_data.sources) # needed in the code below so cannot wait until the end

            # Columns: id z_spec z_a z_m1 chi_a z_p chi_p z_m2 odds l68 u68 l95 u95 l99 u99 nfilt q_z z_peak peak_prob z_mc
            file_path = f"{catalog_params.catalog_path}/UVISTA_final_v4.1.zout"
            catalog_data.redshift = astropy.io.ascii.read(file_path)
            catalog_data.redshift['z_spec_cc'] = catalog_data.sources['z_spec_cc']

            # Columns: id z tau metal lage Av lmass lsfr lssfr la2t chi2
            file_path = f"{catalog_params.catalog_path}/UVISTA_final_BC03_v4.1.fout"
            catalog_data.spp = astropy.io.ascii.read(file_path, header_start=16)
            catalog_data.spp['lmass'][np.isnan(catalog_data.spp['lmass'])] = -99
            catalog_data.spp['lsfr'][np.isnan(catalog_data.spp['lsfr'])] = -99
            no_fit_filter = catalog_data.spp['chi2'] == -1
            catalog_data.spp['ltau'][no_fit_filter]  = -99
            catalog_data.spp['metal'][no_fit_filter] = -99
            catalog_data.spp['lage'][no_fit_filter]  = -99
            catalog_data.spp['Av'][no_fit_filter]    = -99
            catalog_data.spp['lmass'][no_fit_filter] = -99
            catalog_data.spp['lsfr'][no_fit_filter]  = -99
            catalog_data.spp['lssfr'][no_fit_filter] = -99
            catalog_data.spp['la2t'][no_fit_filter]  = -99

            #,id,ra,dec,z_spec,z_phot,log_mass,log_sfr,Av,is_mass_clumpy,is_UV_clumpy,UV_frac_clump
            file_path = f"{catalog_params.catalog_path}/Viz_COSMOS_deconv_clumpy_catalog.csv"
            clump_data = astropy.io.ascii.read(file_path)

            id = np.zeros((catalog_data.count), dtype=np.int_) # pylint: disable=redefined-builtin
            is_mass_clumpy = np.zeros((catalog_data.count), dtype=np.bool)
            is_UV_clumpy = np.zeros((catalog_data.count), dtype=np.bool)
            UV_frac_clump = -1 * np.ones((catalog_data.count))

            for i in np.arange(len(clump_data)):
                indexes = np.where(catalog_data.sources['id'] == clump_data['id'][i])[0]
                if len(indexes) == 0:
                    continue
                idx = indexes[0]

                if catalog_data.sources['z_spec'][idx] != clump_data['z_spec'][i]:
                    raise Exception(f"z_spec doesn't match: {clump_data['id'][i]}")

                id[idx] = clump_data['id'][i]
                is_mass_clumpy[idx] = clump_data['is_mass_clumpy'][i] == 'True'
                is_UV_clumpy[idx] = clump_data['is_UV_clumpy'][i] == 'True'
                UV_frac_clump[idx] = clump_data['UV_frac_clump'][i]

            catalog_data.clumps = Table([
                    id, is_mass_clumpy, is_UV_clumpy, UV_frac_clump
                ], names=[
                    'id', 'is_mass_clumpy', 'is_UV_clumpy', 'UV_frac_clump',
                ], dtype=[
                    np.int_, np.bool, np.bool, np.float64,
                ]
            )

        case 'ZCOSMOS-BRIGHT':
            catalog_data.source = 'ZCB'
            catalog_data.name = 'zCOSMOS Bright'
            catalog_data.date = datetime(2016, 1, 19)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/zCOSMOS_v2.csv" # zCOSMOS Bright DR3 with 20,689 galaxies
            include_names = ['OBJECT_ID','RAJ2000','DEJ2000','IMAG_AB','REDSHIFT','CC']
            catalog_data.sources = astropy.io.ascii.read(file_path, include_names=include_names)
            catalog_data.sources.rename_column('OBJECT_ID', 'id')
            catalog_data.sources.rename_column('RAJ2000'  , 'ra')
            catalog_data.sources.rename_column('DEJ2000'  , 'dec')
            catalog_data.sources.rename_column('IMAG_AB'  , 'mag')
            catalog_data.sources.rename_column('REDSHIFT' , 'z_spec')
            catalog_data.sources.rename_column('CC'       , 'z_spec_cc')

            source_filter = (catalog_data.sources['z_spec_cc'] > 0) & (catalog_data.sources['z_spec'] > 0.0)
            catalog_data.sources = catalog_data.sources[source_filter]

        case 'ZCOSMOS-DEEP':
            catalog_data.source = 'ZCD'
            catalog_data.name = 'zCOSMOS Deep'
            catalog_data.date = datetime(2016, 1, 19) # Not known
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/cosmos_zspec_zgt2.txt" # zCOSMOS Deep (not public, unreleased)
            include_names = ['ra','dec','z_spec','Q_f']
            catalog_data.sources = astropy.io.ascii.read(file_path, include_names=include_names)
            catalog_data.sources.add_column(np.arange(len(catalog_data.sources))+1, name='id', index=0)
            catalog_data.sources.rename_column('Q_f'  , 'z_spec_cc')

        case '3D-HST':
            catalog_data.source = '3DHST'
            catalog_data.name = '3D-HST'
            catalog_data.date = datetime(2014, 9, 3)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version}.cats/Catalog/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version}.cat.FITS"
            if force_tables:
                catalog_data.sources = Table.read(file_path)
            else:
                sources_hdul = fits.open(file_path)
                catalog_data.sources = sources_hdul[1].data
                catalog_data.hduls.append(sources_hdul)

            file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst_v4.1.5_catalogs/{catalog_params.field_file_prefix}_3dhst.v4.1.5.zfit.linematched.fits"
            if force_tables:
                catalog_data.redshift = Table.read(file_path)
            else:
                redshift_hdul = fits.open(file_path)
                catalog_data.redshift = redshift_hdul[1].data
                catalog_data.hduls.append(redshift_hdul)

            file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version:.1f}.cats/Fast/{catalog_params.field_file_prefix}_3dhst.v{catalog_params.field_version:.1f}.fout.FITS"
            if force_tables:
                catalog_data.spp = Table.read(file_path)
            else:
                spp_hdul = fits.open(file_path)
                catalog_data.spp = spp_hdul[1].data
                catalog_data.hduls.append(spp_hdul)

            file_path = f"{catalog_params.catalog_path}/{catalog_params.field_file_prefix}_3dhst_v4.1.5_catalogs/{catalog_params.field_file_prefix}_3dhst.v4.1.5.linefit.linematched.fits"
            if force_tables:
                catalog_data.lines = Table.read(file_path)
            else:
                line_hdul = fits.open(file_path)
                catalog_data.lines = line_hdul[1].data
                catalog_data.hduls.append(line_hdul)

        case 'VUDS':
            catalog_data.source = 'VUDS'
            catalog_data.name = 'VUDS'
            catalog_data.date = datetime(2015, 8, 12)
            catalog_data.frame = 'fk5'

            match catalog_data.field:
                case 'COSMOS':
                    file_path = f"{catalog_params.catalog_path}/cesam_vuds_spectra_dr1_cosmos_catalog.csv"
                case 'GOODS-S':
                    file_path = f"{catalog_params.catalog_path}/cesam_vuds_spectra_dr1_ecdfs_catalog.csv"

            catalog_data.sources = astropy.io.ascii.read(file_path)

            table.add_fields(catalog_data.sources, 'index', np.arange(len(catalog_data.sources))+1)

        case 'Casey':
            catalog_data.source = 'Casey'
            catalog_data.name = 'Casey DSFG'
            catalog_data.date = datetime(2017, 5, 10)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/apj449592t1_mrt.txt"
            catalog_data.sources = astropy.io.ascii.read(file_path)

            table.add_fields(catalog_data.sources, 'index', np.arange(len(catalog_data.sources))+1)

            catalog_data.sources['ra'] = Angle([f"{catalog_data.sources['Name'][i][-18:-16]}h{catalog_data.sources['Name'][i][-16:-14]}m{catalog_data.sources['Name'][i][-14:-9]}s" for i in np.arange(len(catalog_data.sources))]).degree
            catalog_data.sources['dec'] = Angle([f"{catalog_data.sources['Name'][i][-9:-6]}d{catalog_data.sources['Name'][i][-6:-4]}m{catalog_data.sources['Name'][i][-4:]}s" for i in np.arange(len(catalog_data.sources))]).degree

            # Filter to selected field
            match catalog_data.field:
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

            sources_filter = catalog_data.sources['f_zphot'] == field_flag
            catalog_data.sources = catalog_data.sources[sources_filter]

        case 'DEIMOS':
            catalog_data.source = 'DEIMOS'
            catalog_data.name = 'DEIMOS 10K'
            catalog_data.date = datetime(2017, 12, 27)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/deimos_redshifts.tbl"
            catalog_data.sources = astropy.io.ascii.read(file_path)

            table.add_fields(catalog_data.sources, 'index', np.arange(len(catalog_data.sources))+1)

            source_filter = (catalog_data.sources['Remarks'] != 'star') & ~np.isnan(catalog_data.sources['zspec'] > 0.0) & (catalog_data.sources['zspec'] > 0.0)
            catalog_data.sources = catalog_data.sources[source_filter]

        case 'MOSDEF':
            catalog_data.source = 'MOSDEF'
            catalog_data.name = 'MOSDEF'
            catalog_data.date = datetime(2018, 3, 11)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/mosdef_zcat.final_slitap.fits"
            sources_hdul = fits.open(file_path)
            catalog_data.sources = sources_hdul[1].data

            # Filter to selected field
            sources_filter = (catalog_data.sources['FIELD'] == catalog_data.field) & (catalog_data.sources['ID_V4'] >= 0) & (catalog_data.sources['Z_MOSFIRE'] >= 0.0)
            catalog_data.sources = catalog_data.sources[sources_filter]

            # Remove Duplicates
            dup_filter = np.zeros((len(catalog_data.sources)), dtype=np.bool)
            unique_ids, counts = np.unique(catalog_data.sources['ID_V4'], return_counts=True)
            dupped_ids = unique_ids[counts > 1]

            for i in np.arange(len(dupped_ids)):
                dup_id = dupped_ids[i]
                current_filter = catalog_data.sources['ID_V4'] == dup_id
                dup_filter[current_filter] = True
                keep_idx = np.argmax(current_filter)
                dup_filter[keep_idx] = False
                catalog_data.sources['z_MOSFIRE'][keep_idx] = np.mean(catalog_data.sources['z_MOSFIRE'][current_filter])

            catalog_data.sources = catalog_data.sources[~dup_filter]

            # Emission Lines (NOTE: these are not line matched, so do so manually)
            file_path = f"{catalog_params.catalog_path}/linemeas_nocor.fits"
            lines_hdul = fits.open(file_path)
            lines = lines_hdul[1].data

            line_prefixes = get_MOSDEF_line_prefixes()
            empty_idx = np.argmax(lines['ID'] == -9999) # HACK: use the first row we're going to skip anyway
            lines[empty_idx] = np.zeros((1), dtype=lines.columns.dtype)[0]
            line_indexes = empty_idx * np.ones((len(catalog_data.sources)), dtype=np.int_)
            indexes = np.arange(len(lines))
            for i in np.arange(len(catalog_data.sources)):
                row_filter = (lines['FIELD'] == catalog_data.sources['FIELD'][i]) & (lines['ID'] == catalog_data.sources['ID_V4'][i])
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

            catalog_data.lines = lines[line_indexes]

            catalog_data.hduls = [sources_hdul, lines_hdul]

        case 'FMOS':
            catalog_data.source = 'FMOS'
            catalog_data.name = 'FMOS'
            catalog_data.date = datetime(2019, 1, 25)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/fmos-cosmos_catalog_2019.fits"
            sources_hdul = fits.open(file_path)
            catalog_data.sources = sources_hdul[1].data
            catalog_data.lines = catalog_data.sources # copy of sources to access emission line data
            catalog_data.lines_is_copy = True
            catalog_data.hduls = [sources_hdul]

        case 'KMOS3D':
            catalog_data.source = 'KMOS3D'
            catalog_data.name = 'KMOS3D'
            catalog_data.date = datetime(2019, 7, 12)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/k3d_fnlsp_table_v3.fits"
            sources_hdul = fits.open(file_path)
            catalog_data.sources = sources_hdul[1].data

            file_path = f"{catalog_params.catalog_path}/k3d_fnlsp_table_hafits_v3.fits"
            lines_hdul = fits.open(file_path)
            catalog_data.lines = lines_hdul[1].data

            # Filter to selected field
            match catalog_data.field:
                case 'COSMOS':
                    field_code = 'COS'
                case 'GOODS-S':
                    field_code = 'GS'
                case 'UDS':
                    field_code = 'U'

            sources_filter = catalog_data.sources['FIELD'] == field_code
            catalog_data.sources = catalog_data.sources[sources_filter]

            sources_filter = catalog_data.lines['FIELD'] == field_code
            catalog_data.lines = catalog_data.lines[sources_filter]

            # Sources and Lines are line-matched but sources has more objects, drop them
            if len(catalog_data.sources) > len(catalog_data.lines):
                catalog_data.sources = catalog_data.sources[0:len(catalog_data.lines)]

            catalog_data.spp = catalog_data.sources # copy of sources to spp data
            catalog_data.spp_is_copy = True

            catalog_data.hduls = [sources_hdul, lines_hdul]

        case 'C3R2':
            catalog_data.source = 'C3R2'
            catalog_data.name = 'C3R2 DR1+DR2+DR3'
            catalog_data.date = datetime(2021, 6, 18)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/c3r2_DR1+DR2_2019april11.txt"
            catalog_data.sources12 = astropy.io.ascii.read(file_path)

            file_path = f"{catalog_params.catalog_path}/C3R2-DR3-18june2021.txt"
            catalog_data.sources3 = astropy.io.ascii.read(file_path)

            catalog_data.sources = vstack([catalog_data.sources12, catalog_data.sources3])

            source_filter = np.char.startswith(catalog_data.sources['ID'], catalog_data.field)
            catalog_data.sources = catalog_data.sources[source_filter]

            table.add_fields(catalog_data.sources, 'index', np.arange(len(catalog_data.sources))+1)

            catalog_data.sources['ra'] = Angle([f"{catalog_data.sources['RAh'][i]}h{catalog_data.sources['RAm'][i]}m{catalog_data.sources['RAs'][i]}s" for i in np.arange(len(catalog_data.sources))]).degree
            catalog_data.sources['dec'] = Angle([f"{(catalog_data.sources['DE-'][i] if catalog_data.sources['DE-'][i] != '' else '+')}{catalog_data.sources['DEd'][i]}d{catalog_data.sources['DEm'][i]}m{catalog_data.sources['DEs'][i]}s" for i in np.arange(len(catalog_data.sources))]).degree

        case 'LEGAC':
            catalog_data.source = 'LEGAC'
            catalog_data.name = 'LEGAC'
            catalog_data.date = datetime(2021, 8, 2)
            catalog_data.frame = 'fk5'

            file_path = f"{catalog_params.catalog_path}/legac_dr3_cat.fits"
            sources_hdul = fits.open(file_path)
            catalog_data.sources = sources_hdul[1].data
            catalog_data.lines = catalog_data.sources # copy to access emission line data
            catalog_data.lines_is_copy = True
            catalog_data.hduls = [sources_hdul]

        case 'HELP':
            catalog_data.source = 'HELP'
            catalog_data.name = 'HELP Public'
            catalog_data.date = datetime(2000, 1, 1) # So all other z-specs of the same file take precidence (since this is a cross-matched catalog)
            catalog_data.frame = 'fk5'

            match catalog_data.field:
                case 'AEGIS':
                    file_path = f"{catalog_params.catalog_path}/EGS-specz-v2.3.fits"
                case 'COSMOS':
                    file_path = f"{catalog_params.catalog_path}/COSMOS-specz-v2.8-public.fits"

            sources_hdul = fits.open(file_path)
            catalog_data.sources = sources_hdul[1].data
            catalog_data.hduls = [sources_hdul]

            catalog_data.sources = fits.FITS_rec.from_columns(catalog_data.sources.columns.add_col(fits.Column(name='index', array=np.arange(len(catalog_data.sources))+1, format='K')))

        case 'DESI':
            catalog_data.source = 'DESI'
            catalog_data.name = 'DESI EDR'
            catalog_data.date = datetime(2023, 6, 9)
            catalog_data.frame = 'icrs'

            file_path = f"{catalog_params.catalog_path}/edr_galaxy_stellarmass_lineinfo_v1.0.fits"
            sources_hdul = fits.open(file_path)
            catalog_data.sources = sources_hdul[1].data
            catalog_data.spp = catalog_data.sources # copy to access spp line data
            catalog_data.spp_is_copy = True
            catalog_data.lines = catalog_data.sources # copy to access emission line data
            catalog_data.lines_is_copy = True
            catalog_data.hduls = [sources_hdul]

            catalog_data.sources = fits.FITS_rec.from_columns(catalog_data.sources.columns.add_col(fits.Column(name='index', array=np.arange(len(catalog_data.sources))+1, format='K')))

            kron_radius = np.zeros((len(catalog_data.sources)))
            value_filter = (catalog_data.sources['SERSIC'] > 0.0) & (catalog_data.sources['SHAPE_R'] > 0.0)
            kron_radius[value_filter] = _compute_kron_radius(catalog_data.sources['SERSIC'][value_filter], re=catalog_data.sources['SHAPE_R'][value_filter])
            catalog_data.sources = fits.FITS_rec.from_columns(catalog_data.sources.columns.add_col(fits.Column(name='kron_radius', array=kron_radius, format='D')))

        case 'UVISTA-PLUS':
            catalog_file_path = f"{catalog_params.catalog_path}/UVISTA-PLUS_COSMOS.pkl.gz"
            with open(catalog_file_path, 'rb') as f:
                catalog_data = load(f)

            catalog_data.params = catalog_params
            catalog_data.source = 'UVISTAP'

        case '3D-HST-PLUS':
            catalog_file_path = f"{catalog_params.catalog_path}/3D-HST-PLUS_{catalog_params.field}.pkl.gz"
            with open(catalog_file_path, 'rb') as f:
                catalog_data = load(f)

            catalog_data.params = catalog_params
            catalog_data.source = '3DHSTP'

    catalog_data.count = len(catalog_data.sources)
    catalog_data.all_sources = [catalog_data.source]
    catalog_data.all_sources_date = [catalog_data.date]

    if table.has_field(catalog_data, 'spp') and isinstance(catalog_data.spp, Table):
        _, spp_flag, _ = get_spp(catalog_data)
        table.add_fields(catalog_data.spp, 'spp_flag', spp_flag)

    if region is not None:
        if region_wcs is None:
            if table.has_field(catalog_data.params, 'wcs'):
                region_wcs = catalog_data.params.wcs
            else:
                region_wcs = WCS(naxis=2)                                  # 2D WCS for an image
                region_wcs.wcs.crpix = [180, 180]*3600                     # Reference pixel (center of the image)
                region_wcs.wcs.cdelt = np.array([-0.00027778, 0.00027778]) # Pixel scale in degrees (i.e., 1 arcsec/pixel)
                region_wcs.wcs.crval = [180.0, 0.0]                        # Reference world coordinate (RA, Dec in degrees)
                region_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]            # Projection type (TAN for tangent-plane)

        sources_filter = region.contains(SkyCoord(ra=catalog_data.sources[get_ra_field(catalog_data)], dec=catalog_data.sources[get_dec_field(catalog_data)], unit=(u.degree, u.degree)), region_wcs)

        catalog_data.sources = catalog_data.sources[sources_filter]
        catalog_data.count = len(catalog_data.sources)

        if table.has_field(catalog_data, 'redshift'):
            if catalog_data.redshift_is_copy:
                catalog_data.redshift = catalog_data.sources
            else:
                catalog_data.redshift = catalog_data.redshift[sources_filter]

        if table.has_field(catalog_data, 'spp'):
            if catalog_data.spp_is_copy:
                catalog_data.spp = catalog_data.sources
            else:
                catalog_data.spp = catalog_data.spp[sources_filter]

        if table.has_field(catalog_data, 'lines'):
            if catalog_data.lines_is_copy:
                catalog_data.lines = catalog_data.sources
            else:
                catalog_data.lines = catalog_data.lines[sources_filter]

        if table.has_field(catalog_data, 'clumps'):
            if catalog_data.clumps_is_copy:
                catalog_data.clumps = catalog_data.sources
            else:
                catalog_data.clumps = catalog_data.clumps[sources_filter]

    return catalog_data

def open_image(catalog_params):
    image_hdul = fits.open(catalog_params.catalog_image_path)
    return image_hdul

def close(catalog_data):
    for i in np.arange(len(catalog_data.hduls)):
        catalog_data.hduls[i].close()

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
        case 'VUDS' | 'Casey' | 'DEIMOS' | 'C3R2' | 'HELP':
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
        case 'VUDS':
            return 'alpha'
        case 'DEIMOS':
            return 'Ra'
        case 'MOSDEF' | 'FMOS' | 'KMOS3D' | 'LEGAC' | 'HELP':
            return 'RA'
        case 'DESI':
            return 'TARGET_RA'
        case _:
            return 'ra'

def get_dec_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'VUDS':
            return 'delta'
        case 'DEIMOS':
            return 'Dec'
        case 'MOSDEF' | 'FMOS' | 'KMOS3D' | 'LEGAC' | 'HELP':
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
    return np.array(['UVISTA','3DHST','DESI'])

def get_flux_radius_field(catalog_data_or_params):
    match catalog_data_or_params.catalog:
        case 'UVISTA':
            return 'K_Kron'
        case '3D-HST':
            return 'kron_radius'
        case 'DESI':
            return 'kron_radius'
    return None

def get_has_flux_radius(catalog_data):
    return np.any(get_flux_radius_sources() == catalog_data.source)

def get_flux_radius_factor(catalog_data):
    match catalog_data.source:
        case 'UVISTA' | '3DHST':
            return catalog_data.params.pixel_scale[1].value * 3600.0 # pixels -> arcsec
        case 'DESI':
            return 1.0 # arcsec
    return 1.0

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

        case 'ZCOSMOS-BRIGHT' | 'ZCOSMOS-DEEP':
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
            z_spec = catalog_data.sources['z_spec'][idx_or_filter]

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

        case 'HELP':
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
                case 'VUDS':
                    z_field_name = 'z_spec'
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
            # zFlag>=1 is given for objects with approved detection of >=1 emission line(s).
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

        case 'HELP':
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
    return np.array(['UVISTA','3DHST','KMOS3D','DESI'])

def get_has_spp(catalog_data):
    return np.any(get_spp_sources() == catalog_data.source)

def get_spp(catalog_data, idx_or_filter = None, suffix = None):
    spp_names = ['lmass', 'lsfr', 'Av', 'chi2']
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

    spp_names.append('z_spp')
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

#endregion

#region Lines Info

def get_line_sources():
    return np.array(['3DHST', 'MOSDEF', 'FMOS', 'KMOS3D', 'LEGAC', 'DESI'])

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
    line_lambdas = sky.get_vacuum_to_air_wavelength([rest_lambda.SIIb, rest_lambda.SIIa, rest_lambda.NIIb, rest_lambda.Ha, rest_lambda.NIIa, rest_lambda.OIIIb, rest_lambda.OIIIa, rest_lambda.Hb, rest_lambda.OIIb, rest_lambda.OIIa])
    field_names = np.empty((num_lines, num_line_properties), dtype="<U30")
    field_unit_factors = np.ones((num_lines, num_line_properties))
    aperature_correction_fields = np.empty((num_lines), dtype="<U30")

    # SIIb  = 6732.67 # ansgrom
    # SIIa  = 6718.29 # angstrom
    # NIIb  = 6585.27  # angstrom
    # Ha    = 6564.610 # angstrom
    # NIIa  = 6549.86  # angstrom
    # OIIIb = 5008.240 # angstrom
    # OIIIa = 4960.295 # angstrom
    # Hb    = 4862.680 # angstrom
    # OIIb  = 3729.875 # angstrom
    # OIIa  = 3727.092 # angstrom

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
                line_flags[:,i] = lines_table[f"flag_{line_suffixes[i]}"][idx_or_filter]

    if np.ndim(lines) == 2:
        lines = np.array([lines])
        line_flags = np.array([line_flags])

    return lines, line_flags, line_names

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

#region Utilities

def _compute_kron_radius(n, r=1e10, re=1):
    # see https://arxiv.org/pdf/astro-ph/9911078.pdf Eq. 18
    bn = 2*n - 1/3 + 4/405/n + 46/25515/n**2 + 131/1148175/n**3 - 2194697/30690717750/n**4

    # see https://arxiv.org/pdf/astro-ph/0503176.pdf Eq. 32
    x = bn * np.power(r / re, 1 / n)
    krad = re / np.power(bn, n) * (gamma(3*n)*gammainc(3*n, x)) / (gamma(2*n)*gammainc(2*n, x))

    return krad

#endregion
