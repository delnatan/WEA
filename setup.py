from setuptools import setup

setup(
    name="WEA",
    version="0.1.0",
    author="Daniel Elnatan",
    author_email="delnatan@ucdavis.edu",
    description="Wound edge analysis in Python",
    install_requires=[
        "numpy",
        "colorcet",
        "scikit-image",
        "cellpose",
        "scipy",
        "matplotlib",
        "mrc",
        "pandas",
        "nd2reader",
    ],
)
