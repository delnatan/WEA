"""
Image io for WAC

"""

from mrc import DVFile
from nd2reader import ND2Reader
from tifffile import imread, TiffFile
from scipy.ndimage import sobel
import numpy as np
from pathlib import Path


class CanonizedImage:
    """class to conveniently handle various image structures and metadata

    The canonized image should have its axes arranges by "zyxc", where
    the last axis corresponds to channel. Time axis is not supported.


    """

    def __init__(self, fn):
        self.filename = Path(fn)
        self.metadata = None
        self._read()

    def _read(self):
        ext = self.filename.suffix
        if ext == ".mrc" or ext == ".dv":
            with DVFile(self.filename) as img:
                self.dxy = img.hdr.dx
                self.data = img.asarray()
                self.metadata = img.hdr
            self.data = np.moveaxis(self.data, 0, -1)
            wavelengths = [
                self.metadata.__getattribute__(f"wave{i:d}") for i in range(1, 5)
            ]
            ch_names = [f"{i:d} nm" for i in wavelengths if i != 0]
            self.channels = ch_names

        elif ext == ".nd2":
            # ND2Reader takes in path as a string
            with ND2Reader(str(self.filename)) as img:
                # define axis bundle here to "lump" together data as a single variable
                img.bundle_axes = "zyxc"
                # the first 'image' then has all of the data
                self.data = img[0]
                self.dxy = img.metadata["pixel_microns"]
                self.channels = img.metadata["channels"]

        elif ext == ".tif" or ext == ".tiff":
            data = imread(self.filename)

            with TiffFile(self.filename) as tif:
                xres = tif.pages[0].tags["XResolution"].value
                dxy = np.round(xres[1] / xres[0], 5)

            # try to infer channel axis
            ch_axis = np.argmin(data.shape)
            # and move it as the last axis
            data = np.moveaxis(data, ch_axis, -1)
            self.data = data
            self.dxy = dxy
            self.channels = [f"{i:d}" for i in range(1, data.shape[0] + 1)]

        else:
            raise NotImplementedError

    def get_focused_plane(self, channel):
        """return best focused z-slice in all channel"""
        zid = find_focus(self.data[..., channel])
        return self.data[zid, ...]

    def max_project(self):
        return self.data.max(axis=0)

    def __str__(self):
        str_repr = f"filename: {self.filename}\n"
        str_repr += f"shape: {self.data.shape}\n"
        str_repr += f"dxy: {self.dxy:0.4f} um"
        return str_repr

    def __repr__(self):
        return self.__str__()


def find_focus(img3d, edge=True):
    """returns index of in-focus plane

    Given a 3D image with (z, y, x) axis ordering

    Returns:
        integer index at plane of best focus
    """
    if edge:
        wrk = sobel_intensity_np(img3d)
    else:
        wrk = img3d

    zvars = np.var(wrk, axis=(1, 2))
    zmeans = np.mean(wrk, axis=(1, 2))
    normvars = zvars / zmeans

    return np.argmax(normvars)


def sobel_intensity(img):
    csobel = sobel(img.astype(np.float32), axis=-1)
    rsobel = sobel(img.astype(np.float32), axis=-2)
    return np.sqrt(rsobel * rsobel + csobel * csobel)


def sobel_intensity_np(img):
    """do sobel filtering on 3D image, "zyx" axis ordering"""
    fimg = img.astype(np.float32)
    wrk = fimg[:, :, 2:] - fimg[:, :, :-2]
    h = wrk[:, :-2, :] + wrk[:, 2:, :] + 2 * wrk[:, 1:-1:, :]
    wrk = fimg[:, 2:, :] - fimg[:, :-2, :]
    v = wrk[:, :, :-2] + wrk[:, :, 2:] + 2 * wrk[:, :, 1:-1]
    return np.sqrt(h * h + v * v)
