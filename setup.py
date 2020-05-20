from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()
from ai4neb.version import __version__

setup(
    name="ai4neb",
    version=__version__,
    author="Christophe Morisset",
    author_email="Chris.Morisset@gmail.com",
    description="A package to developp AI tools for Nebular studies",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/morisset/AI4neb",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
