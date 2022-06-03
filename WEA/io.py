"""
Image io for WAC

"""

from mrc import DVFile
from nd2reader import ND2Reader
from tifffile import imread
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

        elif ext == ".nd2":
            # ND2Reader takes in path as a string
            with ND2Reader(str(self.filename)) as img:
                # define axis bundle here to "lump" together data as a single variable
                img.bundle_axes = "zyxc"
                # the first 'image' then has all of the data
                self.data = img[0]
                self.dxy = img.metadata["pixel_microns"]
                # dz = np.abs(
                #     np.round(
                #         np.median(np.diff(img.metadata["z_coordinates"])), 3
                #     )
                # )
        elif ext == ".tif" or ext == ".tiff":
            data = imread(self.filename)
            # try to infer channel axis
            ch_axis = np.argmin(data.shape)
            # and move it as the last axis
            data = np.moveaxis(data, ch_axis, -1)
            self.data = data

        else:
            raise NotImplementedError

    def get_focused_plane(self, channel):
        """return best focused z-slice in all channel"""
        zid = find_focus(self.data[..., channel])
        return self.data[zid, ...]

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
