try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "ceiling",
        ["ceiling.pyx"],
    )
]

setup(
    name='ceiling',
    ext_modules=cythonize(ext_modules),
    include_dirs=[]
)
