#pylint: disable=C
import os
from setuptools import setup, find_packages

this_dir = os.path.dirname(__file__)

setup(
    name='hessian_pytorch',
    packages=find_packages(exclude=["build"])
)
