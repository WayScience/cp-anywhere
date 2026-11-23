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
Segmentation preview
--------------------
Use :func:`cp_anywhere.preview_segmentation` to inspect object boundaries over
a grayscale image or view labels as filled colors. This is a 2D display helper,
not an implementation of the CellProfiler desktop window or a pipeline module.

.. automodule:: src.cp_anywhere.preview
   :members:
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

```{eval-rst}
Batch processing
----------------
Apply one translated pipeline to complete, independent image sets. Batch
processing does not replace single-field or individual module calls.

.. automodule:: src.cp_anywhere.batch
   :members:
```

```{eval-rst}
Image quality gate
------------------
Stage 0 of AutoSegment: every image is compared against a ``QcProfile``
built from a real, pinned CellProfiler pipeline's
``MeasureImageQuality``/``FlagImage`` pair, or from explicit
caller-calibrated criteria. Decisions carry reason codes and override
provenance; nothing here is an invented universal image-quality
standard. See
`AutoSegment <project-docs.html#autosegment-image-quality-pipeline-selection-and-reporting>`_
for the full stage descriptions.

.. automodule:: src.cp_anywhere.quality
   :members:
```

```{eval-rst}
AutoSegment
-----------
One call runs the quality gate, splits accepted fields into tuning and
held-out sets (never sharing an acquisition when ``acquisition_groups``
is given), picks a candidate pipeline, tunes its parameters with a
seeded search, re-checks the winner on held-out fields, applies the
selected configuration to new images, and writes offline reports and
review provenance. ``"classical"`` is always available; ``"stardist"``
needs the opt-in ``stardist-model`` native build and raises rather than
falling back silently.

.. automodule:: src.cp_anywhere.autosegment
   :members:
```
