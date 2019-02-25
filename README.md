# adaptive-knn
This repository contains the Cython implementation of the adaptive *k*-nearest-neighbor algorithm [1] and the experiments from the paper.

In order to be able to run the adaptive *k*-NN algorithm, you will need to 
first compile the Cython code with the command

```
$ python setup.py build_ext --inplace
```

Afterwards, run the experiments in `ActiveKNN Subspaces Tiny ImageNet.ipynb`
and generate the plots with `Active KNN Figures.ipynb`.

[1]: D. LeJeune, R. G. Baraniuk, R. Heckel, "Adaptive Estimation for Approximate k-Nearest-Neighbor Computations," 2019. AISTATS 2019.
