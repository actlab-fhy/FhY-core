Getting Started
===============

Install
-------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   pip install .

For contributors:

.. code-block:: bash

   pip install ".[dev]"

Build docs locally
------------------

.. code-block:: bash

   # one-shot build
   sphinx-build -b html docs docs/_build/html

   # or with tox
   tox -e docs

Generated pages are written to ``docs/_build/html``.
