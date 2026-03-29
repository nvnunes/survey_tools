#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import argparse
import os
import time
import astropy.units as u
import numpy as np
import aomap
from astropy.table import Table, vstack
from survey_tools import asterism, healpix
from survey_tools._optional_ao_tools import get_training_module
from survey_tools.utility.table import has_field

parser = argparse.ArgumentParser(description="Export asterisms for AO map.")
parser.add_argument('--output-path', type=str, default='../output', help='Output directory path')
parser.add_argument('--mk', action='store_true', help='Export only declinations suited to Mauna Kea')
parser.add_argument('--survey', type=str, default=None, help='Survey name to export')
parser.add_argument('--rebuild', action='store_true', help='Rebuild output files')
parser.add_argument('--skip_ao', action='store_true', help="Skip adding AO predicted AO performance")
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for predicting AO performance')
parser.add_argument('--skip_backup', action='store_true', help='Skip backup of existing file before adding AO performance')
args = parser.parse_args()

output_path = args.output_path
limit_dec = args.mk
survey = args.survey
rebuild = args.rebuild
skip_ao = args.skip_ao
batch_size = args.batch_size
skip_backup = args.skip_backup

os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = aomap.read_config('config.yaml')

map_level = config.outer_level
suffix = ""
if survey is not None:
    suffix += f"-{survey}"
if limit_dec:
    dec_limit = [-20, 60]
    suffix += '-mk'
else:
    dec_limit = [-90, 90]

for ao_system in config.ao_systems:
    filename = f"{output_path}/asterisms-{ao_system['name']}{suffix}.fits"
    check_angles = ao_system['rot_range'] is not None and ao_system['rot_step'] is not None
    
    if rebuild or not os.path.isfile(filename):
        print(f"Generating asterism file for: {ao_system['name']}...")

        max_map_data = aomap.get_map_data(config, config.max_data_level, f"asterism-count-{ao_system['name']}", survey=survey, dec_limit=dec_limit)
        survey_pixs = max_map_data.pixs[max_map_data.values > 0]
        print(f"  Found {len(survey_pixs)} survey pixels.")

        all_asterisms = []
        outer_pixs = np.unique(healpix.get_parent_pixel(config.max_data_level, survey_pixs, config.outer_level))
        for i, pix in enumerate(outer_pixs):
            print(f"  Processing pixel {i} ({pix})...")
            asterisms = aomap.load_asterisms(config, pix, ao_system['name'], max_dust_extinction=config.max_dust_extinction)

            max_pixs = healpix.get_parent_pixel(config.inner_level, asterisms['pix'], config.max_data_level)
            asterisms = asterisms[np.isin(max_pixs, survey_pixs)]
            if len(asterisms) == 0:
                continue

            asterisms['id'] = asterisms['pix'] # Inner level pixel
            all_asterisms.append(asterisms)

        if len(all_asterisms) > 0:
            asterisms = vstack(all_asterisms)
            asterisms.remove_column('pix')
            asterisms.remove_column('radius')
            asterisms.remove_column('area')
            asterisms.remove_column('relarea')
            asterisms.remove_column('separation')
            asterisms.remove_column('relsep')

            asterisms.write(filename, format="fits", overwrite=True)
            print(f"  Wrote {len(asterisms)} asterisms to {filename}.")
        else:
            print('  No asterisms found.')

    if not skip_ao:
        training = get_training_module()

        if 'asterisms' not in locals():
            asterisms = Table.read(filename)
        
        if rebuild or not has_field(asterisms, 'SR_mean'):
            Y_min = np.zeros((len(asterisms), 3))
            Y_mean = np.zeros((len(asterisms), 3))
            Y_max = np.zeros((len(asterisms), 3))

            for num_stars in range(ao_system['min_wfs'], ao_system['max_wfs'] + 1):
                key = f"{num_stars}star"
                indexes = np.flatnonzero(asterisms['num_stars'] == num_stars)
                num_asterisms = len(indexes)
                if num_asterisms > 0 and key in ao_system['point_models']:
                    model_name = ao_system['point_models'][key]
                    print(f"Processing {num_stars} stars with {model_name}...")
                    model = training.load_model('../data/models', model_name)

                    data = {
                        'wavelength': aomap.get_prediction_wavelength(config),
                        'lgs': ao_system['lgs'],
                        'r': model['data_options']['r'],
                        'theta': model['data_options']['theta']
                    }

                    for lgs in data['lgs']:
                        lgs['x'] = lgs['zd'] * np.cos(np.deg2rad(lgs['az']))
                        lgs['y'] = lgs['zd'] * np.sin(np.deg2rad(lgs['az']))

                    data['x'] = data['r'] * np.cos(np.deg2rad(data['theta']))
                    data['y'] = data['r'] * np.sin(np.deg2rad(data['theta']))

                    num_batches = int(np.ceil(num_asterisms/batch_size))
                    for batch in range(num_batches):
                        current_time = time.strftime("%H:%M:%S", time.localtime())
                        print(f"\r  {current_time}: {batch + 1}/{num_batches}          ", end='', flush=True)

                        start_idx = batch * batch_size
                        end_idx = min((batch + 1) * batch_size, num_asterisms)
                        if start_idx >= end_idx:
                            continue
                        batch_indexes = indexes[start_idx:end_idx]
                        N = len(batch_indexes)
                        M = len(model['data_options']['r'])

                        data['ngs'] = asterism.get_ngs_from_asterisms(asterisms[batch_indexes])

                        X = training.get_model_X(data)
                        if check_angles:
                            # These exported field summaries use resolved point-model predictions,
                            # but the applied orientation is the catalog best_angle chosen by the
                            # mean-model ranking path. That makes this a mixed product: resolved
                            # model field performance at a mean-model-selected angle, not a
                            # self-consistent resolved-model best-over-rotation evaluation.
                            best_angle = asterisms['best_angle'][batch_indexes]
                            if np.ma.is_masked(best_angle):
                                best_angle = np.ma.filled(best_angle, 0)
                            best_angle = np.repeat(best_angle, M).reshape(-1, 1)
                            theta_idxs = training.get_ngs_theta_indexes(num_stars)
                            X[:, theta_idxs] += np.deg2rad(best_angle)
                            X[:, theta_idxs] = training.wrap_angle_rad(X[:, theta_idxs])

                        Y_pred = training.get_prediction(X, model)
                        Y_pred = Y_pred.reshape(N, M, Y_pred.shape[1])

                        Y_mean[batch_indexes, :] = np.mean(Y_pred, axis=1)
                        Y_min[batch_indexes, :] = np.min(Y_pred, axis=1)
                        Y_max[batch_indexes, :] = np.max(Y_pred, axis=1)

                        training.clear_cache(model)

                    print(f"\n  done\n")

            asterisms['SR_mean'] = Y_mean[:, training.get_sr_index()]
            asterisms['SR_min'] = Y_min[:, training.get_sr_index()]
            asterisms['SR_max'] = Y_max[:, training.get_sr_index()]

            ee_size = 100 # mas
            asterisms[f"EE{ee_size}_mean"] = Y_mean[:, training.get_ee_index()]
            asterisms[f"EE{ee_size}_min"] = Y_min[:, training.get_ee_index()]
            asterisms[f"EE{ee_size}_max"] = Y_max[:, training.get_ee_index()]

            asterisms['FWHM_mean'] = Y_mean[:, training.get_fwhm_index()]
            asterisms['FWHM_min'] = Y_min[:, training.get_fwhm_index()]
            asterisms['FWHM_max'] = Y_max[:, training.get_fwhm_index()]

            if not skip_backup:
                if os.path.exists(filename + ".bak"):
                    os.remove(filename + ".bak")
                os.rename(filename, filename + ".bak")

            asterisms.write(filename, format="fits", overwrite=True)
            print(f"  Appended AO performance to {len(asterisms)} asterisms in {filename}.")
