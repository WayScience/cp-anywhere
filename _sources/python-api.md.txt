# Python API

```{eval-rst}
cp_anywhere.api
-------------------
.. automodule:: src.cp_anywhere.api
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
cp_anywhere.io
-------------------
The simplest entry point for files on disk: :func:`cp_anywhere.io.measure_files`
accepts TIFF/OME-TIFF, PNG, or saved NumPy array paths (or already-loaded
arrays) directly and returns a tidy ``pandas.DataFrame``, instead of
requiring images to be pre-converted to NumPy arrays first.

.. automodule:: src.cp_anywhere.io
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
cp_anywhere.segment
-------------------
Object segmentation includes :func:`cp_anywhere.identify_primary_objects`,
:func:`cp_anywhere.identify_secondary_objects`, and
:func:`cp_anywhere.identify_tertiary_objects`. The native implementations
use the corresponding CellProfiler modules as
the compatibility oracle -- see
`segmentation coverage <project-docs.html#segmentation-coverage-dashboard>`_
for the declared scope and verification status.

.. automodule:: src.cp_anywhere.segment
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
cp_anywhere.cppipe
-------------------
Importing real CellProfiler ``.cppipe`` pipeline files
(:func:`cp_anywhere.load_pipeline`) -- see
`CellProfiler pipeline import <project-docs.html#cellprofiler-pipeline-cppipe-import>`_
for the declared scope and every explicit gap.

.. automodule:: src.cp_anywhere.cppipe
   :members:
   :undoc-members:
   :show-inheritance:
```
