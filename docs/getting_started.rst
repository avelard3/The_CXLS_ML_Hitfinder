Getting Started with Documentation
==================================

The documentation source lives in ``docs/``. The generated HTML output is
written to ``docs/_build/html/`` and is intentionally ignored by git.

Install Sphinx
--------------

Use the project environment. If Sphinx is not already available in that
environment, install it there:

.. code-block:: bash

   conda activate hitfinder_sol_env
   python -m pip install sphinx

Build the Docs
--------------

From the repository root, run with the activated environment:

.. code-block:: bash

   cd /scratch/avelard3/The_CXLS_ML_Hitfinder
   python -m sphinx -b html docs docs/_build/html

Or run through the environment without activating it first:

.. code-block:: bash

   cd /scratch/avelard3/The_CXLS_ML_Hitfinder
   mamba run -n hitfinder_sol_env python -m sphinx -b html docs docs/_build/html

Then open:

.. code-block:: text

   docs/_build/html/index.html

Writing Good Docstrings
-----------------------

Sphinx is configured to understand Google-style docstrings like this:

.. code-block:: python

   def normalize_image(image, scale):
       """Normalize an image by a scale factor.

       Args:
           image (np.ndarray): Image data to normalize.
           scale (float): Scale factor to divide by.

       Returns:
           np.ndarray: Normalized image data.
       """

For classes, put the high-level explanation in the class docstring. Put
parameter details either in the class docstring or in ``__init__``.
