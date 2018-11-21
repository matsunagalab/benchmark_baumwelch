from distutils.core import setup
from Cython.Build import cythonize

setup(name="msmbaumwelch",
      ext_modules=cythonize("msmbaumwelch.pyx"))
