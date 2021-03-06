################################################################################
ectopylasm
################################################################################

Tools for visualizing and fitting pointcloud data.

.. image:: https://travis-ci.org/sundial-pointcloud-geometry/ectopylasm.svg?branch=master
    :target: https://travis-ci.org/sundial-pointcloud-geometry/ectopylasm

.. image:: https://codecov.io/gh/sundial-pointcloud-geometry/ectopylasm/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/sundial-pointcloud-geometry/ectopylasm

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/sundial-pointcloud-geometry/ectopylasm/master?filepath=notebooks%2FSundial%20surface.ipynb

.. image:: https://readthedocs.org/projects/ectopylasm/badge/?version=latest
 :target: https://ectopylasm.readthedocs.io/en/latest/?badge=latest
 :alt: Documentation Status

Installation
************

The recommended way to install ectopylasm is by using a virtual environment. Using Miniconda, one can get a working environment on Linux with the following commands:

.. code-block:: console

  # on Linux
  conda create -n ectopylasm python=3.7
  conda activate ectopylasm
  pip install git+https://github.com/sundial-pointcloud-geometry/ectopylasm.git

On macOS, the pptk dependency is only available for Python 3.6, so there one should use:

.. code-block:: console

  # on macOS
  conda create -n ectopylasm python=3.6
  conda activate ectopylasm
  pip install git+https://github.com/sundial-pointcloud-geometry/ectopylasm.git


To install ectopylasm from a cloned git repo, do:

.. code-block:: console

  git clone https://github.com/sundial-pointcloud-geometry/ectopylasm.git
  cd ectopylasm
  pip install .


Run tests (including coverage) with:

.. code-block:: console

  python setup.py test


.. Documentation
.. *************

.. .. _README:

.. Include a link to your project's full documentation here.

Contributing
************

If you want to contribute to the development of ectopylasm,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
*******

Copyright (c) 2019, Humboldt-Universität zu Berlin

Licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Credits
*******

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
