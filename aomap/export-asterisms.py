#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
import aomap
from astropy.table import vstack
from survey_tools import catalog, healpix

config = aomap.read_config('config.yaml')
ao_system_name = 'GNAO-Optimal'
dec_limit = [-20,60]
map_level = config.outer_level
survey = 'ews'
output_path = '../output'

max_map_data = aomap.get_map_data(config, config.max_data_level, f"asterism-count-{ao_system_name}", survey=survey, dec_limit=dec_limit)
survey_pixs = max_map_data.pixs[max_map_data.values > 0]
print(f"Found {len(survey_pixs)} survey pixels.")

all_asterisms = []
outer_pixs = np.unique(healpix.get_parent_pixel(config.max_data_level, survey_pixs, config.outer_level))
for i, pix in enumerate(outer_pixs):
    print(f"Processing pixel {i} ({pix})...")
    asterisms = aomap.load_asterisms(config, pix, ao_system_name, max_dust_extinction=config.max_dust_extinction)

    max_pixs = healpix.get_parent_pixel(config.inner_level, asterisms['pix'], config.max_data_level)
    asterisms = asterisms[np.isin(max_pixs, survey_pixs)]
    if len(asterisms) == 0:
        continue

    asterisms['id'] = asterisms['pix'] # Inner level pixel
    all_asterisms.append(asterisms)

if all_asterisms:
    combined_asterisms = vstack(all_asterisms)
    combined_asterisms.remove_column('pix')
    combined_asterisms.remove_column('radius')
    combined_asterisms.remove_column('area')
    combined_asterisms.remove_column('relarea')
    combined_asterisms.remove_column('separation')
    combined_asterisms.remove_column('relsep')

    filename = f"{output_path}/asterisms-{ao_system_name}.fits"
    combined_asterisms.write(filename, format="fits", overwrite=True)
    print(f"Wrote {len(combined_asterisms)} asterisms to {filename}.")
else:
    print('No asterisms found.')

# Generate sample target catalog from 3D-HST
catalog_name = '3D-HST'
field_names  = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S', 'UDS']
filter_name  = 'F160W'  # F125W, F140W, F160W
rows_per_field = 1000

targets = []
for i in range(len(field_names)):
    catalog_params = catalog.get_params(catalog_name, field_names[i], filter_name)
    catalog_data = catalog.CatalogData(catalog_params)
    galaxy_data = catalog.flatten_galaxy_data(catalog_data)

    random_indices = np.random.choice(len(galaxy_data), rows_per_field, replace=False)
    galaxy_data = galaxy_data[random_indices]

    galaxy_data['field'] = field_names[i]
    selected_columns = ['field', 'phot_id', 'ra', 'dec', 'z_best', 'lmass', 'lsfr', 'Av']
    galaxy_data = galaxy_data[selected_columns]
    galaxy_data.rename_column('phot_id', 'id')
    galaxy_data.rename_column('z_best', 'z')
    targets.append(galaxy_data)

targets = vstack(targets)
filename = f"{output_path}/sample-targets.fits"
targets.write(filename, format="fits", overwrite=True)
print(f"Wrote {len(targets)} sample targets to {filename}.")
