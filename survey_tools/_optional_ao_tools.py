#!/usr/bin/env python3
# pylint: disable=missing-module-docstring

from importlib import import_module


def get_training_module():
    try:
        return import_module("ao_tools.legacy.training")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "AO model prediction requires 'ao_tools.legacy.training', which is provided by girmos-aosims. "
            "Install girmos-aosims to use AO prediction paths."
        ) from exc
