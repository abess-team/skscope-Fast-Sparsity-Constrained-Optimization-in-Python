from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    name="skscope_experiment",
    version="0.0.1",
    author="",
    author_email="",
    url="",
    description="",
    long_description="",
    ext_modules=[
        Pybind11Extension("_skscope_experiment",
            ["src/main.cpp"],
            # Example: passing in the version to the compiled code
            include_dirs=["include"],
            extra_compile_args=["-O3", "-Werror", "-DNDEBUG"],
            cxx_std=17,
            ),
    ],
    python_requires=">=3.7",
)
