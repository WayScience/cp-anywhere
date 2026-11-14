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
