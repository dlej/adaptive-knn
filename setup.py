from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension('heap', ['heap.pyx'], language='c++', include_dirs=[np.get_include()]),
    Extension('adaptive_knn', ['adaptive_knn.pyx'], language='c++', include_dirs=[np.get_include()])
]

setup(ext_modules=cythonize(extensions))
