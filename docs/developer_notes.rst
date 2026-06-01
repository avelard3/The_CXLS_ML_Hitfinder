Developer Notes
===============

This page is for people maintaining the code. The public API pages focus on
the methods that a user is expected to call directly. These notes describe the
internal flow, including private helper methods whose names start with
``_``.

Training and Evaluation Flow
----------------------------

The main training entry point is ``src/train_and_evaluate_hitfinder.py``.
The high-level sequence is:

1. Parse command-line arguments with ``arguments()``.
2. Build a configuration dictionary for model training.
3. Create a ``Paths`` object in ``training`` mode.
4. Call ``Paths.run_paths()`` to inspect input HDF5 files and build a virtual
   dataset.
5. Create a ``Data`` object, which opens the virtual dataset created by
   ``Paths``.
6. Use ``CreateDataLoader.split_training_data()`` to create train and test
   loaders.
7. Create a ``TrainModel`` object, initialize training objects, optionally load
   transfer-learning weights, then run the epoch loop.
8. Save the trained model state dictionary.
9. Create an ``EvaluateModel`` object and generate evaluation outputs such as
   the classification report, confusion matrix, and ROC curve.
10. Remove the temporary ``training_vds_delete_me.h5`` file.

The important thing to remember is that the data path is staged through a
temporary HDF5 virtual dataset before PyTorch sees it.

Path and VDS Preparation
------------------------

The ``Paths`` class in ``lib.load_paths`` is responsible for turning a list of
input HDF5 files into a consistent virtual dataset.

The public method is ``Paths.run_paths()``. Internally it calls:

``_prepare_file_info()``
   Opens the input ``.lst`` file, skips master files, checks each source HDF5
   file, finds the image dataset, records whether each file contains a single
   event or multiple events, and calculates the total virtual dataset shape.

``_map_dataset_to_vds()``
   Creates HDF5 virtual layouts for image data, camera length, photon energy,
   and, in training mode, hit labels. It then loops through the source files
   again and maps each source dataset into the virtual layout.

``_find_path_in_h5()``
   Looks through a configured list of possible HDF5 dataset paths and returns
   the first one that exists in the current file. The candidate paths live in
   ``lib.conf``.

``_define_photon_energy()``
   Converts wavelength values to photon energy when the source file stores
   wavelength instead of energy, then returns an HDF5 virtual source.

``_crop_image()``
   Crops image data to ``conf.required_image_size`` when an input image is
   larger than the expected square size.

``_multipanel_to_single()``
   Uses ``read_scattering_matrix.ScatteringMatrix`` to convert multipanel
   detector data into a single image-like array before it is mapped into the
   virtual dataset.

``_add_file_to_list()``
   Records the source file and event number for later output. Multi-event HDF5
   files produce one recorded name per event.

Training mode also maps hit labels into the VDS. Running mode maps only the
image and metadata needed for inference.

Data Loading
------------

The ``Data`` class in ``lib.load_data`` expects the VDS file created by
``Paths`` to already exist. Its constructor opens:

* ``vsource_image``
* ``vsource_camera_length``
* ``vsource_photon_energy``
* ``vsource_hit_parameter`` in training mode

``Data.__getitem__()`` returns image data, metadata, the hit label, and the
source file/event name. ``CreateDataLoader`` then wraps this dataset in PyTorch
``DataLoader`` objects.

Model Training
--------------

``TrainModel`` owns the training loop. The usual call order is:

1. ``make_training_instances()``
2. ``load_model_state_dict()``
3. ``assign_new_data(train_loader, test_loader)``
4. ``epoch_loop()``
5. ``plot_loss_accuracy()``
6. ``save_model()``
7. ``get_model()``

Model architecture names are resolved dynamically, so the command-line
``--model`` argument must match a class name in ``lib.models``.

Evaluation
----------

``EvaluateModel`` runs the trained model over the test loader, stores
predictions, and produces reporting plots. The training script calls:

1. ``run_testing_set()``
2. ``make_classification_report()``
3. ``plot_confusion_matrix()``
4. ``plot_roc_curve()``

Configuration
-------------

Shared constants and default hyperparameters live in ``lib.conf``. This module
also contains the lists of possible HDF5 paths that ``Paths._find_path_in_h5()``
checks when it searches for images and metadata.

Files Created During Runs
-------------------------

Several workflows create temporary or output files:

* ``training_vds_delete_me.h5`` or ``running_vds_delete_me.h5`` is the temporary
  virtual dataset file.
* Some metadata conversion paths create helper files such as
  ``camera_length*.h5``, ``photon_energy*.h5``, or
  ``scaled_photon_energy.h5``.
* Training saves model state dictionaries and evaluation figures to paths
  provided by command-line arguments.

These generated files are runtime artifacts, not source files.
