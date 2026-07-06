Running the hitfinder
======================

This page describes how to run the hitfinder: running inference with a
pretrained model, training models, and performing optional hyperparameter
tuning. It focuses on the commands and inputs required to run the code and
does not explain the internal implementation details or rationale for the
design — those belong in the developer section.


For End-Users
-----------------------------------------------

This quick guide shows how an end-user (someone who already has a trained
model file) can run the hitfinder to produce lists of HDF5 files that
contain peaks.

Prerequisites
~~~~~~~~~~~~~

- Have a trained model state dictionary (``.pt``), e.g. ``hitfinder_model_7.pt``.
- A newline-separated ``.lst`` file that lists the HDF5 files to classify.
- A working environment with dependencies (see the repository's environment
   YAML files). Activate the environment first:

.. code-block:: bash

    module load mamba/latest
    source activate hitfinder_env_try8

Run the hitfinder
-----------------

Use the included runner script ``src/run_hitfinder_model.py``. Example:

.. code-block:: bash

    cd /scratch/avelard3/The_CXLS_ML_Hitfinder
    python src/run_hitfinder_model.py \
          -l path/to/your_list.lst \
          -m ModelClassName \
          -d /path/to/your_model.pt \
          -o /path/to/save/output/lists \
          -b 64

- ``-l/--list``: path to the ``.lst`` file with HDF5 file paths (one per line).
- ``-m/--model``: the model class name as defined in ``src/lib/models.py``.
- ``-d/--dict``: path to the saved PyTorch state dict (``.pt``).
- ``-o/--output``: directory where the generated ``found_peaks-*.lst`` and ``no_peaks-*.lst`` files will be written.
- ``-b/--batch``: batch size to use when running inference.

The script will print progress and create two ``.lst`` files in the output
directory: one for predicted peak files and one for predicted empty files.

For Developers
--------------------------------------------------------------------------------

This section explains how developers can optimize hyperparameters, train
models, evaluate them, and run the hitfinder. It links the main helper
scripts in the repository and shows example commands.

Prepare data and .lst files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a newline-separated ``.lst`` containing the HDF5 files to use for
  training/validation/testing.

- HDF5 files must be listed with the absolute path and file name
  (example: ``/home/path/to/file/file_name.h5``).

- If a dataset has a master file, the master file should be listed before
  the corresponding data files.

- If you have mixed datasets where some experiments include master files
  and others do not, list the non-master datasets first.

.. code-block:: bash

   /home/path/to/file/file_nameA.h5
   /home/path/to/file/file_nameB.h5
   /home/path/to/dataC/exp1_master.h5
   /home/path/to/dataC/exp1_ds001.h5
   /home/path/to/dataC/exp1_ds002.h5
   /home/path/to/file/dataD/exp2_master.h5
   /home/path/to/file/dataD/exp2_ds001.h5

Train a model
~~~~~~~~~~~~~

The training driver is ``src/train_and_evaluate_hitfinder.py``. Use it when
training a new model and evaluating test performance in one run.
Example template scripts are available in ``sbatch_script_templates/`` and
actual cluster job examples are available in ``sbatch_scripts/``.

.. code-block:: bash

    # Example: run a single training job (replace args as needed)
    python src/train_and_evaluate_hitfinder.py \
          -l path/to/train_list.lst \
          -m ModelClassName \
          -d /path/to/save/model.pt \
          -o /path/to/save/analysis \
          -e 20 \
          -b 64 \
          -op Adam -s ReduceLROnPlateau -c BCEWithLogitsLoss \
          -lr 0.001

Evaluate a trained model
~~~~~~~~~~~~~~~~~~~~~~~~~

The training driver ``src/train_and_evaluate_hitfinder.py`` also performs
model evaluation after training by generating a confusion matrix and ROC
curve. If you need standalone inference only, use
``src/run_hitfinder_model.py`` instead.

.. code-block:: bash

    python src/train_and_evaluate_hitfinder.py \
          -l path/to/test_list.lst \
          -m ModelClassName \
          -d /path/to/trained_model.pt \
          -o /path/to/save/plots \
          -e 1 \
          -b 64 \
          -op Adam -s ReduceLROnPlateau -c BCEWithLogitsLoss \
          -lr 0.001

Run the hitfinder (production inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After you have a trained model, use ``src/run_hitfinder_model.py`` (see the
End-user section above). For developers you may want to run with different
batch sizes, different model classes, or to wrap the script in a job scheduler
for large datasets.

Hyperparameter tuning (Optuna)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the tuning driver which wraps Optuna and the training loop. Example:

.. code-block:: bash

    cd /scratch/avelard3/The_CXLS_ML_Hitfinder
    python src/hyperparameter_tuning.py \
          -sn my-study-name \
          -nt 50 \
          -l path/to/train_list.lst \
          -m ModelClassName \
          -b 64 \
          -op Adam -s ReduceLROnPlateau -c BCEWithLogitsLoss

- ``-sn/--study_name``: Optuna study name (used for the sqlite storage).
- ``-nt/--num_trials``: number of Optuna trials to run.
- Other flags control ranges for explored hyperparameters; see the script's
   ``--help`` output for all options.

After running, the script prints ``Best Hyperparameters`` and saves visual
plots of the tuning history (requires plotly/graphviz extras for image output).

Notes and tips
~~~~~~~~~~~~~~

- Model class names are defined in ``src/lib/models.py``; use the exact class
   name when passing ``-m``.
- The model loader helper is ``src/lib/utils.py::LoadModel`` which calls
   ``torch.load`` and ``load_state_dict``; saved state dicts must be compatible
   with the chosen model class.
- If you need GPU execution, ensure CUDA is available in the environment; the
   scripts automatically select GPU if available.


Using sbatch job scripts (cluster runs)
---------------------------------------

The repository includes example sbatch job templates under
``sbatch_script_templates/`` and actual job examples under ``sbatch_scripts/``.
Use the templates to copy and customize for your own cluster or HPC account.

Examples:

- Hyperparameter tuning template:
  ``sbatch_script_templates/hyperparameter_tune_hitfinder_example.sh``
- Training template:
  ``sbatch_script_templates/train_and_evaluate_hitfinder_example.sh``
- Inference template:
  ``sbatch_script_templates/run_hitfinder_example.sh``



NOT DONE YET
~~~~~~~~~~~~


Notes for maintainers / FIXMEs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The sbatch scripts set `HDF5_PLUGIN_PATH` to a location inside the activated
   conda environment. Ensure the plugin path matches your environment layout.
   #FIXME: verify and document which conda env name(s) we officially support.

- Several scripts pass a geometry file path via ``-g`` or set it to ``None``.
   This controls multipanel detector handling. If your HDF5 files use different
   attribute names for image/camera_length/photon_energy, update the sbatch
   script and the Python driver accordingly.  #FIXME: add exact attribute
   key names used by our dataset loader here once confirmed.

- The training and tuning drivers expect many command-line flags that are
   commonly provided by the sbatch wrappers; when running manually, inspect
   the sbatch script to see the chosen values.  #FIXME: add a short table of
   the most important sbatch-set flags (epochs, lr, batch size, model class)
   after you confirm the canonical defaults.

- The sbatch scripts conditionally set a `transfer_learning` variable and
   pass it into training; ensure that the path and file name conventions for
   transfer-learning state dicts are what you expect.  #FIXME: confirm the
   preferred storage location for trained `.pt` files.

Labeling real data (master files)
---------------------------------

Overview
~~~~~~~~

Some experiments include "master" files (metadata about runs or detectors)
that must be used when labeling real HDF5 data for supervised training or
validation. The instructions below outline what information is needed and how
to provide it to the hitfinder drivers. Several fields require your input;
they are left intentionally blank for you to fill in and are marked with
``#FIXME``.

Master file process
--------------------------

- Describe the master files and their role here.  #FIXME

Creating label files
~~~~~~~~~~~~~~~~~~~~

- Describe the expected label format (per-event, per-file, CSV, JSON, list,
   etc.). Leave concrete examples blank for now.  #FIXME

Format
''''''

- ``<label-file-format>``:  #FIXME - specify format, fields, and required keys.

How to attach labels to HDF5 files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Explain whether labels are embedded into HDF5, provided as a separate
   ``.lst`` with label columns, or passed via a mapping file.  #FIXME

Providing labels to the hitfinder when running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- If labels are passed at runtime, show the CLI flag(s) to use here (example
   placeholder): ``--labels path/to/labels.file``  #FIXME: confirm exact flag
   or configuration mechanism.

- If labels are expected in a certain path or naming convention, document
   that convention here.  #FIXME

Integration notes
~~~~~~~~~~~~~~~~~

- If the training/tuning drivers require special preprocessing for master
   files (geometry transforms, normalization, indexing), document the steps
   here.  #FIXME

Please fill in the ``#FIXME`` sections above with the exact formats and
examples you want included in the public docs; I will update the file after
you provide those details.
