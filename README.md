# TomoGUI

Qt front end for GPU tomographic reconstruction. Drives
[TomoCuPy](https://tomocupy.readthedocs.io) for Try / Full / Batch / AI
reconstruction and (optionally) [TomoLog](https://tomologcli.readthedocs.io)
for logbook / PDF reports.

## Install

The recommended path is a single conda environment that holds the GUI
stack, CUDA, PyTorch and the external reconstruction backends.

```bash
git clone <repository-url> tomogui
cd tomogui
conda env create -f environment.yml
conda activate tomogui
```

`environment.yml` already does an editable `pip install -e .` of TomoGUI
itself, so edits under `src/` take effect without reinstalling.

### Optional: TomoLog

TomoLog has no PyPI wheel yet. Install from source into the same env:

```bash
pip install git+https://github.com/xray-imaging/tomolog.git
```

See [docs/user_guide/installation.rst](docs/user_guide/installation.rst)
for the full dependency list, CUDA requirements, non-editable-install
cleanup and troubleshooting.

## Run

```bash
conda activate tomogui
tomogui
```

CLI entry point (headless phases: try / ai / full / tomolog):

```bash
tomogui-cli --help
```

## Dependencies at a glance

GUI: PyQt5, pyqtgraph, VisPy, PyOpenGL
IO / science: numpy ≥ 2.0, scipy, h5py, matplotlib, Pillow, tifffile, numexpr, psutil
AI stack: PyTorch (CUDA 12), einops, timm, CuPy 13.x, CUDA toolkit ≥ 12.4
Optional: anthropic (chatbot tab)
External backends: **TomoCuPy** (required), **TomoLog** (optional)
