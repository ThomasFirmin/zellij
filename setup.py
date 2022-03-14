import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    packages=['zellij', 'zellij.strategies', 'zellij.strategies.utils', 'zellij.utils'],
    keywords=["fractal", "continuous optimization", "global optimization", "black-box functions", "decision space partitioning","exploration", "exploitation", "metaheuristics", "tree search"  ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['numpy', 'deap', 'matplotlib', 'pandas', 'seaborn', 'enlighten', 'GPy'],
    python_requires=">=3.6"
)
