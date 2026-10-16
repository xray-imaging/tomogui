Installation
============

TomoGUI is a Qt front end that drives external reconstruction / logging
backends. Those backends must be reachable from the same Python environment
that runs the GUI; the recommended path is a single conda environment that
holds everything (GUI + CUDA + tomocupy + tomolog).

Requirements
------------

Python Version
~~~~~~~~~~~~~~

Python ≥ 3.8, tested up to 3.12. The provided ``environment.yml`` pins
3.11, which is the version used on the production beamline nodes.

Core Dependencies (bundled in environment.yml)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GUI stack:

- **PyQt5** ≥ 5.15
- **pyqtgraph** ≥ 0.13.3 — SSH / software fallback renderer
- **VisPy** ≥ 0.14 — GPU renderer (default when OpenGL is available)
- **PyOpenGL** ≥ 3.1

Scientific / IO stack:

- **numpy** ≥ 2.0
- **scipy** ≥ 1.13
- **h5py** ≥ 3.11 (+ ``hdf5`` runtime)
- **matplotlib** ≥ 3.7
- **Pillow** ≥ 10.0
- **tifffile** ≥ 2023.1
- **numexpr** ≥ 2.8
- **psutil** ≥ 5.9

AI center-of-rotation search:

- **PyTorch** ≥ 2.2 (with CUDA 12 matching the ``cuda-toolkit`` pin)
- **einops** ≥ 0.6
- **timm** ≥ 0.9 (DINOv2 backbone)
- **CuPy** 13.x on CUDA 12.4

Optional:

- **anthropic** ≥ 0.40 — enables the in-GUI assistant tab.

CUDA
~~~~

The default environment installs:

- ``cuda-toolkit=12.4`` + ``cuda-version=12.4``
- ``pytorch-cuda=12.4``

CUDA ≥ 12.4 is the minimum that provides ``__nv_fp8_e8m0`` support in
CuPy and the NVRTC FP8 compilation path tomocupy uses on driver 12.8+.
``nvidia-smi`` must list every GPU you plan to use for Batch AI.

External Backends
-----------------

These are **separate projects** that TomoGUI drives as subprocesses. They
must be importable (tomolog) or on ``$PATH`` (tomocupy) in the same
environment the GUI runs in.

TomoCuPy (required — reconstruction engine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drives the Try / Full / Batch / AI reconstructions. Needs the full CUDA
stack (``cupy``, ``cuda-toolkit``, PyTorch with CUDA, FFTW), which is why
the ``environment.yml`` pulls those conda packages first.

- Project: https://tomocupy.readthedocs.io
- Source:  https://github.com/nikitinvv/tomocupy

The default ``environment.yml`` installs it from PyPI via pip::

    - tomocupy

If PyPI is unavailable or you need an unreleased patch, swap for a
source install::

    pip install git+https://github.com/nikitinvv/tomocupy.git

Verify::

    tomocupy --help

TomoLog (optional — logbook / PDF report)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pushes a reconstruction summary + preview to the beamline logbook, and
can emit a PDF. Only required if you use the Tomolog panel in the main
window or the ``tomolog`` phase of ``tomogui-cli``.

- Project: https://tomologcli.readthedocs.io
- Source:  https://github.com/xray-imaging/tomolog

No PyPI wheel yet — install from source into the same environment::

    pip install git+https://github.com/xray-imaging/tomolog.git

Verify::

    python -c "import tomolog; print(tomolog.__version__)"

AI model checkpoints
~~~~~~~~~~~~~~~~~~~~

The AI center-of-rotation search uses a bundled copy of ``tomocor_infer``
in ``src/tomogui/_tomocor_infer/`` so you do **not** need to install
``tomocor`` separately. You *do* need the model weights:

- **FINE** (``--infer-model-path``): default
  ``<env>/site-packages/tomogui/AImodels/fine_v2.pt``
- **FULL** (``--bin-infer-model-path``): default
  ``/home/beams/TANGS/conda/tomocor_models/.../checkpoints/epoch_26.pt``

Both can be overridden per-run via the Browse button in the AI tab. See
``src/tomogui/AImodels/README.txt`` for current defaults and architecture
notes.

Installation Methods
--------------------

Conda Environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full environment — GUI, CUDA, tomocupy, PyTorch — is reproducible from
the provided ``environment.yml``::

    git clone <repository-url> tomogui
    cd tomogui
    conda env create -f environment.yml
    conda activate tomogui

The last line of ``environment.yml`` (``- -e .``) performs an editable
install of TomoGUI, so git-pulled updates to ``src/`` take effect
immediately — no second ``pip install`` needed.

Install tomolog on top (optional)::

    pip install git+https://github.com/xray-imaging/tomolog.git

From Source (pip only — no CUDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only need the GUI shell for inspection / HDF5 viewing (not
reconstruction), a plain pip install into any venv works::

    git clone <repository-url> tomogui
    cd tomogui
    pip install -e . --config-settings editable_mode=compat

Reconstruction features will fail at launch time with an "import
tomocupy" error — that is expected without the CUDA stack.

Verifying Installation
----------------------

.. code-block:: bash

   conda activate tomogui
   tomogui --version
   tomocupy --help            # external backend reachable
   python -c "import tomolog" # external backend reachable (optional)
   tomogui                     # launches the GUI

If the main window opens and the *Browse Data Folder* button responds, the
installation is complete.

.. figure:: /_static/screenshots/main_window.png
   :alt: TomoGUI main window
   :align: center

   Main window on successful launch.

Switching from a non-editable install
-------------------------------------

If a previous session installed TomoGUI as a non-editable copy into
``site-packages``, edits in ``src/`` will **not** take effect until you
clear the old copy::

   pip uninstall tomogui -y
   rm -rf $(python -c "import site; print(site.getsitepackages()[0])")/tomogui*
   pip install -e . --config-settings editable_mode=compat

Check the install mode with::

   cat $(python -c "import tomogui, os; print(os.path.join(os.path.dirname(tomogui.__path__[0]), 'tomogui-*.dist-info', 'direct_url.json'))") 2>/dev/null

Editable installs show ``"editable": true`` inside ``dir_info``.

Troubleshooting
---------------

Qt Platform Plugin Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

Common on remote nodes without an X11 server::

   export QT_QPA_PLATFORM_PLUGIN_PATH=""
   # or, for a pure offscreen render:
   export QT_QPA_PLATFORM=offscreen

PyQt5 Import Errors
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip uninstall PyQt5
   pip install PyQt5==5.15.9

CUDA / NVRTC FP8 errors
~~~~~~~~~~~~~~~~~~~~~~~

If CuPy / tomocupy crashes with ``__nv_fp8_e8m0 is not defined`` or
similar NVRTC messages, your CUDA toolkit is older than the driver.
Pin ``cuda-toolkit=12.4`` (or newer) in the env — this is what the
provided ``environment.yml`` already does.

Verify::

   nvidia-smi            # should list every GPU
   nvcc --version        # should be ≥ 12.4

AI Reco model not found
~~~~~~~~~~~~~~~~~~~~~~~

In the AI tab, set *--infer-model-path* / *--bin-infer-model-path* to an
absolute path reachable **from the host that will run the recon** (SSH
targets, batch workers). The GUI defaults assume the beamline's canonical
checkpoint tree.
