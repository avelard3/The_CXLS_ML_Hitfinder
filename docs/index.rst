The CXLS ML Hitfinder Documentation
===================================

The CXLS ML Hitfinder is a Python workflow for preparing CXLS HDF5 image
data, training hit-finding neural networks, evaluating trained models, and
running classification on new files.

This home page describes the project at a high level and shows how to create
and activate the required environment. For usage-specific commands and flags,
see the ``Running the hitfinder`` page. For internal architecture and workflow
details, see ``How the hitfinder works``.

What the hitfinder does
~~~~~~~~~~~~~~~~~~~~~~~

The hitfinder ingest pipeline takes one or more HDF5 datasets, assembles a
virtual dataset for training or inference, and feeds image data through a
PyTorch-based neural network. The main use cases are:

- Prepare source HDF5 files and master metadata for training and validation.
- Train a hit-finding model and save the resulting state dictionary.
- Evaluate trained models with reports, confusion matrices, and ROC curves.
- Run inference over new HDF5 files and produce lists of predicted hits.

Environment setup
~~~~~~~~~~~~~~~~~

The project requires a mamba environment, as defined in ``environment.yaml``, that must be activated before running training or inference.

Example commands:

.. code-block:: bash

   module load mamba/latest
   mamba env create -f environment.yaml -n hitfinder_env
   # or to update an existing env
   mamba env update -f environment.yaml -n hitfinder_env

Activate the environment:

.. code-block:: bash

   source activate hitfinder_env

.. toctree::
   :maxdepth: 2
   :caption: Contents

   running
   developer_notes
   api

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
