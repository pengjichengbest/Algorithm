# coding:utf-8
"""
    Author: apple
    Date: 2/5/2024
    File: setup.py
    ProjectName: Algorithm
    Time: 11:59
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


compile_flags = []
linker_flags = []

module = Extension('difference',
                   ['difference.pyx'],
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=compile_flags,
                   extra_link_args=linker_flags)
setup(name='difference',
      ext_modules=cythonize(module),
      script_args=['build_ext', '--inplace'])
