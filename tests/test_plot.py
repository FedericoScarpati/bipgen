import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import bipgen  # noqa: E402


@pytest.mark.parametrize("dim,model", [(2, {"model": "kernel", "scale": 0.3}),
                                       (3, {"model": "knn", "k": 3})])
def test_plots_run(dim, model):
    spec = {"sizes": {"n_bcn": 80, "n_tgt": 90}, "dim": dim, "edges": model,
            "background": {"model": "uniform", "ratio": 0.1}}
    if model["model"] == "kernel":
        spec["degrees"] = {"distribution": "constant", "bcn": {"value": 4}, "tgt": {}}
    g = bipgen.generate(bipgen.spec_from_dict(spec))
    bipgen.plot_graph(g)
    bipgen.plot_degrees(g)
    bipgen.plot_edge_lengths(g)
    plt.close("all")
