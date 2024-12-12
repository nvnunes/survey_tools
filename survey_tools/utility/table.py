#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

def has_field(o, column):
    if isinstance(o, Table):
        return np.any(np.array(o.colnames) == column)

    if isinstance(o, fits.BinTableHDU) or isinstance(o, fits.FITS_rec):
        return np.any(np.array(o.columns.names) == column)

    return hasattr(o, column)

def add_fields(table, field_names, field_values):
    if np.size(field_values) > np.size(field_names):
        single_values = False
        if np.ndim(field_values) == 1:
            field_values = np.reshape(field_values,(-1,1))

        if np.size(field_names) > 1 and np.shape(field_values)[1] != len(field_names):
            field_values = np.transpose(field_values)
    else:
        single_values = True

    for i in np.arange(np.size(field_names)):
        if np.size(field_names) == 1:
            field_name = field_names
        else:
            field_name = field_names[i]

        if has_field(table, field_name):
            if single_values:
                table[field_name] = field_values[i]
            else:
                table[field_name] = np.reshape(field_values[:,i], (-1,))
        else:
            if single_values:
                if isinstance(field_values[i], int):
                    fits_format = 'K'
                else:
                    fits_format = 'D'

                if isinstance(table, fits.BinTableHDU) or isinstance(table, fits.FITS_rec):
                    table.columns.add_col(fits.Column(name=field_name, array=np.repeat(field_values[i], len(table)), format=fits_format))
                else:
                    table.add_column(field_values[i], name=field_name)
            else:
                if issubclass(field_values[i].dtype.type, np.integer):
                    fits_format = 'K'
                else:
                    fits_format = 'D'

                if isinstance(table, fits.BinTableHDU) or isinstance(table, fits.FITS_rec):
                    table.columns.add_col(fits.Column(name=field_name, array=field_values[:,i]), format=fits_format)
                else:
                    table.add_column(field_values[:,i], name=field_name)

def add_rows(catalog_data, table, field_names, field_values, default_value = None, default_value_func = None):
    if np.isscalar(field_values):
        field_values = np.array([[field_values]])
    elif np.size(field_values) == 1 and np.ndim(field_values) == 1:
        field_values = np.array([field_values])
    elif np.shape(field_values)[0] == np.size(field_names) and np.shape(field_values)[1] == 1:
        field_values = np.transpose(field_values)

    if np.size(field_values) > np.size(field_names):
        if np.ndim(field_values) == 1:
            field_values = np.reshape(field_values,(-1,1))

        if np.size(field_names) > 1 and np.shape(field_values)[1] != len(field_names):
            field_values = np.transpose(field_values)

    if isinstance(table, fits.BinTableHDU) or isinstance(table, fits.FITS_rec):
        dtypes = table.columns.dtype
        columns = table.columns.names
    else:
        dtypes = table.dtype
        columns = table.colnames

    row_vals = np.zeros((field_values.shape[0]), dtype=dtypes)

    if default_value is not None:
        row_vals[:] = default_value
    elif default_value_func is not None:
        for i in np.arange(len(columns)):
            column_default_value = default_value_func(columns[i], catalog_data)
            if column_default_value is not None:
                row_vals[columns[i]] = column_default_value

    for i in np.arange(field_values.shape[0]):
        for j in np.arange(field_values.shape[1]):
            if np.size(field_names) == 1:
                field_name = field_names
            else:
                field_name = field_names[j]

            row_vals[field_name][i] = field_values[i,j]

    return vstack([table, Table(row_vals)])
