# bipgen

Spatial bipartite proximity graphs from degree-matched null models.

bipgen places two populations of nodes, BCN (beacons) and TGT (targets), in the unit
disk (2D) or the unit ball (3D) and links them with a probability that decays with distance. Node
weights are fitted by Sinkhorn scaling so that expected degrees match a target
degree sequence, which can be taken from an observed graph. Background edges,
unrelated to distance, can be added in a chosen proportion. Positions can be uniform,
follow a spatially correlated density, or be given.

Every graph is described by one YAML specification; a seed selects one realization.

## Installation

```bash
pip install "git+https://github.com/FedericoScarpati/bipgen@v0.1.0"
```

Python 3.10 or later, and git. The wheel is also attached to each
[release](https://github.com/FedericoScarpati/bipgen/releases). Dependencies: NumPy, SciPy,
pandas, Matplotlib, PyYAML, threadpoolctl.

## Example

```python
import bipgen

spec = bipgen.spec_from_dict({
    "sizes": {"n_bcn": 2000, "n_tgt": 3000},
    "degrees": {"distribution": "lognormal", "bcn": {"median": 15, "sigma": 0.5},
                "tgt": {"sigma": 0.6}},  # shape only: scale set by the BCN total
    "edges": {"model": "kernel", "kernel": "gaussian", "scale": 0.12},
    "background": {"model": "uniform", "ratio": 0.05},
})
graph = bipgen.generate(spec)            # seed 42 unless set otherwise
ensemble = [bipgen.generate(spec, seed=s) for s in range(10)]
bipgen.plot_graph(graph)
```

Presets built from published Slide-recon and Slide-tags data reproduce the sizes, degree
distributions and spatial density of observed graphs:

```python
spec = bipgen.presets.load("c58_4")
table = bipgen.compare(bipgen.presets.observed("c58_4"), [bipgen.generate(spec)])
```

## Documentation

The [documentation](https://federicoscarpati.github.io/bipgen/) covers the specification
format, the model with its assumptions and known limitations, and two tutorials. Build it
locally with `pip install -e ".[docs]"` and `mkdocs serve`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Authors

[Federico Scarpati](https://orcid.org/0009-0001-3248-8528) and
[Martin Weigt](https://orcid.org/0000-0002-0492-3684), Sorbonne Université, CNRS, Institut de
Biologie Paris-Seine, Laboratoire de Biologie Computationnelle, Quantitative et Synthétique (CQSB),
Paris, France.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Sorbonne Université.

## Funding

<p>
  <a href="https://www.voluminex-project.eu/"><img src="https://raw.githubusercontent.com/FedericoScarpati/bipgen/v0.1.0/docs/assets/funding/voluminex.png" alt="VOLUMINEX" height="56"></a>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/FedericoScarpati/bipgen/v0.1.0/docs/assets/funding/eu_funded.png" alt="Funded by the European Union" height="64">
</p>

bipgen is developed within [VOLUMINEX](https://www.voluminex-project.eu/), *Instrument-free
3D molecular imaging with the VOLumetric UMI-Network EXplorer*. This project has received
funding from the European Innovation Council (EIC) established under the EU Horizon Europe
programme, [grant 101184647](https://cordis.europa.eu/project/id/101184647). Views and
opinions expressed are however those of the authors only and do not necessarily reflect
those of the European Union and European Innovation Council (EIC). Neither the European
Union nor the granting authority can be held responsible for them.
