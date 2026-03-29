#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
from astropy.table import vstack
from survey_tools import catalog

output_path = '../output'

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
