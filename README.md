# Reproducible materials
This repository contains scripts to reproduce the numerical results analysis described in "[skscope: Fast Sparsity-Constrained Optimization in Python](https://www.jmlr.org/papers/volume25/23-1574/23-1574.pdf)". 
A step-by-step instruction for reproducing is provided here.

## Instruction

We compare algorithms in ``skscope`` and other well-known methods under several models. Here are the steps for reproducing Table-A3, similar to others.

* First, manually install two python libraries: ``skscope_experiment`` (provided the simulation data production method required for the experiment), ``parallel_experiment_util`` (convenient for running experiments with multiple processes). Note that these libraries are not available in Pypi, users should use `pip install ./skscope_experiment` and `pip install ./parallel_experiment_util`.

* Next, run experiments like `python ./figure_A3/A3_skscope.py`, `python ./figure_A3/A3_gurobi.py`. Note that, gurobi need a license to run.

* Finally, statistic the result by `./figure_A3/plot.ipynb`.

## Citations

Please cite the following publications if you make use of the material here.

- Zezhi Wang, Junxian Zhu, Xueqin Wang, Jin Zhu, Huiyang Pen, Peng Chen, Anran Wang, Xiaoke Zhang (2024). skscope: Fast Sparsity-Constrained Optimization in Python. Journal of Machine Learning Research, 25(290), 1−9.

The corresponding BibteX entries:

```
@article{JMLR:v25:23-1574,
  author  = {Zezhi Wang and Junxian Zhu and Xueqin Wang and Jin Zhu and Huiyang Pen and Peng Chen and Anran Wang and Xiaoke Zhang},
  title   = {skscope: Fast Sparsity-Constrained Optimization in Python},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {290},
  pages   = {1--9},
  url     = {http://jmlr.org/papers/v25/23-1574.html}
}
```

## Contact

Please direct questions and comments to the [issues page](https://github.com/abess-team/skscope-Fast-Sparsity-Constrained-Optimization-in-Python/issues).
