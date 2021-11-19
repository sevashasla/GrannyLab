from setuptools import setup, find_packages
import os


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name="GrannyLab",
    version="0.0.3",
    author="sevashasla",
    author_email="skorokhodov.vs@phystech.edu",
    url="https://github.com/sevashasla/GrannyLab",
    # long_description=read("README.md"),
    description="library for easy errors cointing",
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.17.4",
    ],

)
