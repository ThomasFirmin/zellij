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
    version="0.0.1",
    author="Thomas Firmin",
    author_email="thomas.firmin.etu@univ-lille.fr",
    description="A generic framework for fractal-based optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
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
    package_dir={"": "lib"},
    packages=setuptools.find_packages("lib"),
    install_requires=[
        "numpy",
        "torch",
        "deap",
        "matplotlib",
        "pandas",
        "seaborn",
        "enlighten",
        "botorch",
    ],
    extras_require={"mpi": ["mpi4py"], "docs": docs_extras},
    python_requires=">=3.6",
)
