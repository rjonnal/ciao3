try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "centroid",
        ["centroid.pyx"],
        extra_compile_args=['-fopenmp','-march=native'],
        extra_link_args=['-fopenmp','-march=native'],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
    )
]

setup(
    name='centroid-parallel',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
