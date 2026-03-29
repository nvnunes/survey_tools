"""Interim wrapper exposing the top-level ``aomap`` package via ``survey_tools``."""

from importlib import import_module

_AOMAP_MODULE = import_module("aomap")


def __getattr__(name):
    return getattr(_AOMAP_MODULE, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_AOMAP_MODULE)))
