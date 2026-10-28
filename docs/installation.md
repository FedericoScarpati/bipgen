# Installation

bipgen needs Python 3.10 or later. Its dependencies are NumPy, SciPy, pandas,
Matplotlib, PyYAML and threadpoolctl.

```bash
pip install "git+https://github.com/FedericoScarpati/bipgen@v0.1.0"
```

This installs version 0.1.0 from GitHub and needs git. Without git, download the wheel
attached to the [release](https://github.com/FedericoScarpati/bipgen/releases) and install it
with `pip install bipgen-0.1.0-py3-none-any.whl`. Version 0.1.0 was tested with Python 3.13,
NumPy 2.5, SciPy 1.18, pandas 3.1 and Matplotlib 3.11, and with Python 3.10 and the newest
versions of the dependencies available for it.

From a clone of the repository, in editable mode with the development tools:

```bash
git clone https://github.com/FedericoScarpati/bipgen
cd bipgen
pip install -e ".[dev]"
pytest
```

## Hardware requirements

- **CPU**: one realization runs on one core, in about half a minute for
  27 000 x 19 000 nodes. Independent realizations can run in parallel processes.
- **Memory**: fitting the kernel model holds one dense `n_bcn x n_tgt` matrix of 32-bit
  floats, about 2 GB for 27 000 x 19 000 nodes. Sampling works on row blocks.
- **Disk**: a saved graph takes about 22 bytes per edge.
