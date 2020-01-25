from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

'''
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True
'''

setup(
  name = 'HyperDoc2vec_inner',
  ext_modules=cythonize([
    Extension("HyperDoc2vec_inner",
              ["HyperDoc2vec_inner.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include(),
                            "/home/ashwath/miniconda3/lib/python3.6/site-packages/gensim/models"],
              #define_macros = [('CYTHON_TRACE_NOGIL','1')]
      )
    ],
  ),
)