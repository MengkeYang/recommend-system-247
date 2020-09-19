from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os, platform

pyx_directories = ["evaluator/cpp/"]

if platform.system() == 'Darwin':
    extra_compile_args = ['-stdlib=libc++', '-fPIC']
else:
    extra_compile_args = ['-std=c++11']

extensions = [
    Extension(
        '*',
        ["*.pyx"],
        extra_compile_args=extra_compile_args)
]

pwd = os.getcwd()
for dir in pyx_directories:
    target_dir = os.path.join(pwd, dir)
    os.chdir(target_dir)
    setup(
        ext_modules=cythonize(extensions,
                              language="c++"),
        include_dirs=[np.get_include()]
    )
    os.chdir(pwd)
