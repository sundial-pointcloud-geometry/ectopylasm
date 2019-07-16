#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit ectopylasm/__version__.py
version = {}
with open(os.path.join(here, 'ectopylasm', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='ectopylasm',
    version=version['__version__'],
    description="Tools for visualizing and fitting pointcloud data.",
    long_description=readme + '\n\n',
    author="E. G. Patrick Bos",
    author_email='egpbos@gmail.com',
    url='https://github.com/sundial-pointcloud-geometry/ectopylasm',
    packages=[
        'ectopylasm',
    ],
    package_dir={'ectopylasm':
                 'ectopylasm'},
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='ectopylasm',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'numpy>=1.16',
        'plyfile',
        'vaex-core',
        'pandas',
        'ipyvolume',
        'pptk',
        'symfit',
        'tqdm'
    ],
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'dev': [
            'prospector[with_pyroma]==1.1.6.2',  # newer versions have a problem with pylint, see e.g. https://github.com/PyCQA/pylint-plugin-utils/issues/12
        ],
    }
)
