# WEA
Single-cell wound-edge analysis

For cell segmentation, this relies on the Cellpose project (https://www.cellpose.org).

# Installation

Some things will need to be installed in order using `conda` first, then we can install the rest with `pip`. At the moment (when this README document was written), PyQt5 (5.15) can be installed across all platforms including the newer M1 Macs (arm64).

The miniconda distribution is highly recommended: https://docs.conda.io/en/latest/miniconda.html

Install miniconda and open the "Anaconda (Command Prompt)" on Windows or the "Terminal" on MacOS.

Create a new conda environment for python 3.8. You don't need to name it `wea-env`, you can use whatever makes sense to you. 
But since I've only tested this on Python 3.8, we'll create the new Python environment with this version:
`conda create -n wea-env python=3.8`

Then activate the newly created environment:
`conda activate wea-env`

Set the conda channels:
`conda config --add channels conda-forge`
`conda config --set channel_priority strict`

Install `pyqt`:
`conda install pyqt pyqt5-sip`

Install `pytorch`:
`conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`

Install `napari`:
`python -m pip install napari[all]`

Install `cellpose`:
`python -m pip install cellpose[gui]`

Note: if you're having trouble installing cellpose with the `[gui]` option, you can just install with `python -m pip install cellpose` and install `pyqtgraph` separately by running:
`python -m pip install pyqtgraph`

# Download this github repository
If using `git` in the command line (recommended to easily get updated codes in the future). You can install `git` via conda by running:
`conda install git`

Then you can run these commands to copy my code repository from github:
`git clone https://github.com/delnatan/WEA.git`
`git clone https://github.com/delnatan/napari-WEA.git`

The
To install the `WEA` (Wound Edge Analysis) package, go into the `WEA` directory (using `cd WEA` in the command line) and run:
`python -m pip install -e .`

Repeat the same steps with `napari-WEA` (https://github.com/delnatan/napari-WEA) package (go into its directory and run the same line as above to install the package locally)

After installation, you can run Python and try importing the module, or run `napari` and go to the `Plugins > napari-WEA` from the menu on top. A side panel widget should open.
