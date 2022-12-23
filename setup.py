# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-11T16:12:13+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-12-23T14:49:36+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

docs_extras = [
    "Sphinx >= 3.0.0",  # Force RTD to use >= 3.0.0
    "docutils",
    "pylons-sphinx-themes >= 1.0.8",  # Ethical Ads
    "pylons_sphinx_latesturl",
    "repoze.sphinx.autointerface",
    "sphinxcontrib-autoprogram",
    "sphinx-copybutton",
    "sphinx-tabs",
    "sphinx-panels",
    "sphinx-rtd-theme",
    "pillow>=6.2.0",
]

setuptools.setup(
    name="zellij",
    version="1.0.1",
    author="Thomas Firmin",
    author_email="thomas.firmin@univ-lille.fr",
    description="A software framework for HyperParameters Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CeCILL-C Free Software License Agreement (CECILL-C)",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "fractal",
        "continuous optimization",
        "global optimization",
        "black-box functions",
        "decision space partitioning",
        "exploration",
        "exploitation",
        "metaheuristics",
        "tree search",
    ],
    url="https://github.com/ThomasFirmin/zellij",
    project_urls={
        "Bug Tracker": "https://github.com/ThomasFirmin/zellij/issues",
    },
    package_dir={"": "lib"},
    packages=setuptools.find_packages("lib"),
    install_requires=[
        "numpy>=1.21.4",
        "scipy>=1.7.3",
        "DEAP>=1.3.1",
        "botorch>=0.6.3.1",
        "gpytorch>=1.6.0",
        "pandas>=1.3.4",
        "enlighten>=1.10.2",
    ],
    extras_require={
        "mpi": ["mpi4py>=3.1.2"],
        "docs": docs_extras,
    },
    python_requires=">=3.6",
)
