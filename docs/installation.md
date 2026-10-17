# Installation

bipgen needs Python 3.10 or later. Its dependencies are NumPy, SciPy, pandas,
Matplotlib, PyYAML and threadpoolctl.

```bash
pip install bipgen
```

From a clone of the repository, in editable mode with the development tools:

```bash
git clone https://github.com/FedericoScarpati/bipgen
cd bipgen
pip install -e ".[dev]"
pytest
```

The documentation is built with `pip install -e ".[docs]"` and `mkdocs serve`.

## Memory and time

Fitting the kernel model holds one dense `n_bcn x n_tgt` matrix of 32-bit floats:
about 2 GB for 27 000 x 19 000 nodes. Sampling works on row blocks and needs little
memory. One realization of that size takes about half a minute on one core.
