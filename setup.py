"""
Installation script
"""
from setuptools import setup, find_packages

setup(
    name='hessian',
    packages=find_packages(exclude=["build"])
)
