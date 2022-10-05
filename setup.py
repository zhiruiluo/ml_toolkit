#!/usr/bin/env python

import os
from importlib.util import module_from_spec, spec_from_file_location
from setuptools import find_packages, setup
# from distutils.core import setup

_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")

def _load_py_module(fname, pkg="ml_toolkit"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py

about = _load_py_module("__about__.py")
setup_tools = _load_py_module("setup_tools.py")


setup(
    name='ml_toolkit',
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    license=about.__license__,
    include_package_data=True,
    keywords=['deep learning','ML','pytorch','AI'],
    python_requires=">=3.7",
    install_requires=setup_tools._load_requirements(_PATH_ROOT),
)