# this is a setup file for python package 'parallel'
from setuptools import setup, find_packages

setup(
    name='parallel_experiment_util',
    author='Ang',
    packages=find_packages(),
    description="parallel_experiment_util: use for parallel experiment",
    install_requires=[
        "numpy",
        "pandas"
    ]
)
