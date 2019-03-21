# adaptive-knn
This repository contains the Cython implementation of the adaptive *k*-nearest-neighbor algorithm [1] and the experiments from the paper.

In order to be able to run the adaptive *k*-NN algorithm, you will need to 
first compile the Cython code with the command

```
$ python setup.py build_ext --inplace
```

Afterwards, run the experiments in [`Adaptive k-NN Subspaces Tiny ImageNet.ipynb`](https://github.com/dlej/adaptive-knn/blob/master/Adaptive%20k-NN%20Subspaces%20Tiny%20ImageNet.ipynb)
and generate the plots with [`Adaptive k-NN Figures.ipynb`](https://github.com/dlej/adaptive-knn/blob/master/Adaptive%20k-NN%20Figures.ipynb).

[1] D. LeJeune, R. Heckel, R. G. Baraniuk, "Adaptive Estimation for Approximate k-Nearest-Neighbor Computations," 2019. AISTATS 2019.
