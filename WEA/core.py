"""
Wound edge analysis modules and functions

"""
import logging
import re

from itertools import chain
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from cellpose import models
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.signal import convolve2d, fftconvolve
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import (
    convex_hull_image,
    disk,
    remove_small_objects,
    skeletonize_3d,
)
from skimage.segmentation import clear_border, find_boundaries

from . import __file__
from .vis import makeRGBComposite

# check whether GPU is available
if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False


# get current module path by using the __file__ attribute
__module_dir = Path(__file__).parent
__model_dir = __module_dir / "models"
__log_dir = Path.home() / "WEA_log"

if not __log_dir.exists():
    __log_dir.mkdir(exist_ok=True)


logger = logging.getLogger("WEA_logger")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f"{__log_dir / 'WEA_dev.log'}")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


# custom models are stored in $HOME/.cellpose/models
# cytoengine = models.CellposeModel(
#     gpu=use_gpu,
#     pretrained_model=str(__model_dir / "CP_bcat-nuc_v02_blur"),
# )

# nucengine = models.CellposeModel(
#     gpu=use_gpu, pretrained_model=str(__model_dir / "CP_dapi_v01")
# )

DEFAULT_CYTO_PATH = str(__model_dir / "CP_bcat-nuc_v3c")
DEFAULT_NUC_PATH = str(__model_dir / "CP_dapi_v1b")
DEFAULT_TUBASCYTO_PATH = str(__model_dir / "CP_tub-nuc_v3d")

logger.info(f"Using models from {str(__model_dir)}")


endpt_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float64)


def collect_cells(pathstr):
    """get a collection of cells"""
    ptn = re.compile("cell_(?P<cellnum>[0-9]+).tif")
    flist = [f for f in Path(pathstr).glob("cell*.tif")]
    cflist = [f for f in flist if ptn.match(f.name)]
    nums = map(int, [ptn.match(f.name)["cellnum"] for f in cflist])
    return {n: Cell(pathstr, n) for n in sorted(nums)}


def norm99(x):
    y = x.copy()
    p01 = np.percentile(x, 1)
    p99 = np.percentile(x, 99)
    return (y - p01) / (p99 - p01)


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class ImageField:
    def __init__(
        self, data, pixel_size, nucleus_ch=0, cyto_channel=1, tubulin_ch=2
    ):
        self.data = data
        self.dxy = pixel_size
        self.nuc_ch = nucleus_ch
        self.cyt_ch = cyto_channel
        self.tub_ch = tubulin_ch
        self.Ny, self.Nx, self.Nch = self.data.shape
        self.cp_labcells = None
        self.cp_labnucs = None
        self.downscale_factor = 1.0
        self._load_cellpose_model()

    def _load_cellpose_model(
        self, cyto_model_path=DEFAULT_CYTO_PATH, nuc_model_path=DEFAULT_NUC_PATH
    ):

        self.cytoengine = models.CellposeModel(
            gpu=use_gpu, pretrained_model=cyto_model_path
        )
        self.nucengine = models.CellposeModel(
            gpu=use_gpu, pretrained_model=nuc_model_path
        )

    def _create_cellpose_input(
        self, input_cell_diam, target_cell_diam=100, downsize=True
    ):

        if downsize:
            downscale_factor = target_cell_diam / input_cell_diam
            self.downscale_factor = downscale_factor
            ty, tx = [round(downscale_factor * s) for s in self.data.shape[:2]]
            img = cv2.resize(
                self.data, (tx, ty), interpolation=cv2.INTER_LINEAR
            )
            scaled_dxy = self.dxy / downscale_factor
        else:
            img = self.data.astype(np.float32)
            scaled_dxy = self.dxy

        # create RGB input
        Ny, Nx = img.shape[0:2]
        cp_input = np.zeros(img.shape[0:2] + (3,), dtype=np.float32)
        cp_input[:, :, 0] = img[:, :, self.tub_ch]
        cp_input[:, :, 1] = img[:, :, self.cyt_ch]
        cp_input[:, :, 2] = img[:, :, self.nuc_ch]

        return cp_input, scaled_dxy

    def segment_cells(
        self,
        cytochs=[2, 3],
        nucchs=[3, 0],
        celldiam=70.0,
        nucdiam=15.0,
        downsize=True,
        **kwargs,
    ):
        """resizes images and run cellpose

        These defaults have been acquired from working with 3T3 cells. You
        should change them to suit your images

        Args:
            celldiam(float, optional): average diameter of the cell in micron.
            nucdiam(float, optional): average diameter of the nucleus in micron.


        """

        unscaled_cell_diam = celldiam / self.dxy

        img, scaled_dxy = self._create_cellpose_input(
            unscaled_cell_diam, downsize=downsize
        )

        self._celldiam = celldiam / scaled_dxy
        self._nucdiam = nucdiam / scaled_dxy

        if "cyto_flow_threshold" in kwargs:
            cyto_flow_threshold = kwargs["cyto_flow_threshold"]
        else:
            cyto_flow_threshold = 0.5

        if "nuc_flow_threshold" in kwargs:
            nuc_flow_threshold = kwargs["nuc_flow_threshold"]
        else:
            nuc_flow_threshold = 0.5

        cmask, cflow, cstyle = self.cytoengine.eval(
            img,
            diameter=self._celldiam,
            resample=True,
            channels=cytochs,
            flow_threshold=cyto_flow_threshold,
            cellprob_threshold=-2.0,
        )

        nmask, nflow, nstyle = self.nucengine.eval(
            img,
            diameter=self._nucdiam,
            resample=True,
            channels=nucchs,
            flow_threshold=nuc_flow_threshold,
            cellprob_threshold=-2.0,
        )

        logger.info(f"Original image size is : {self.data.shape[0:2]}")
        logger.info(f"Cellpose input is resized to : {img.shape[0:2]}")
        logger.info("Segmentation run with parameters:")
        logger.info(
            f"cell diameter={celldiam/scaled_dxy:.2f} px, nucleus diameter={nucdiam/scaled_dxy:.2f} px."
        )

        self.cp_input = img
        self.cp_labcells = cmask
        # internal variables used to resample cell mask
        self._cyt_dP = cflow[1]
        self._cyt_cellprob = cflow[2]
        self.cp_labnucs = nmask

    def run_detection(
        self,
        cell_diam=70.0,
        nuc_diam=15.0,
        cytochs=[2, 3],
        nucchs=[3, 0],
        downsize=True,
    ):

        if self.cp_labcells is None:
            self.segment_cells(
                celldiam=cell_diam,
                nucdiam=nuc_diam,
                downsize=downsize,
                cytochs=cytochs,
                nucchs=nucchs,
            )

        # identify wound edge
        cellarea = binary_fill_holes(self.cp_labcells > 0)
        woundarea = remove_small_objects(
            ~cellarea, min_size=self._celldiam**2
        )
        # we need this thick so that it overlaps with cell masks
        self.woundedge = find_boundaries(woundarea, mode="thick")

        # remove cells touching the border
        self.labcells = clear_border(self.cp_labcells)
        self.labnucs = self.labcells * (self.cp_labnucs > 0)

    def _segmentation_result(self, **kwargs):

        cell_boundaries = find_boundaries(self.cp_labcells, mode="thin")
        nuc_boundaries = find_boundaries(self.cp_labnucs, mode="thin")

        rgb_img = makeRGBComposite(self.cp_input, ch_axis=-1, **kwargs)

        rgb_img[cell_boundaries, 0] = 1.0
        rgb_img[cell_boundaries, 1] = 1.0
        rgb_img[nuc_boundaries, 1] = 1.0
        rgb_img[nuc_boundaries, 2] = 1.0
        # rgb_img[self.woundedge, 0] = 1.0

        return rgb_img

    def _detection_result(self):
        """returns pre-processed detection result for visual diagnostics

        Note: the images will be rescaled to the raw image dimensions
        via nearest-neighbor interpolation

        """
        targetsize = self.data.shape[:2][::-1]
        woundedge = cv2.resize(
            self.woundedge.astype(np.uint8),
            targetsize,
            interpolation=cv2.INTER_NEAREST,
        )
        cp_output = cv2.resize(
            self.cp_labcells, targetsize, interpolation=cv2.INTER_NEAREST
        )
        cp_nucleus = cv2.resize(
            self.cp_labnucs, targetsize, interpolation=cv2.INTER_NEAREST
        )
        return cp_output, cp_nucleus, woundedge

    def edge_cells(self, nuc_diam=15.0):
        """iterator function that yields edge cells

        Args:
            nuc_diam (float): diameter of a nucleus, used for computing morphological
            opening operation footprint. This number is divided by 4 and scaled
            by pixel size to simulate a disk approximately 1/4 the area.

        Each iteration yields ROI-cropped data:
        raw data, cytoplasm mask, wound edge, nucleus mask

        """
        nucleus_opening_radius = int(np.round(nuc_diam / 4) / self.dxy)

        # get label ids for cells at the edge
        woundcells = self.labcells * self.woundedge
        edge_id = np.delete(np.unique(woundcells), 0)

        # rescale image labels
        labeled_cells = cv2.resize(
            self.labcells, (self.Nx, self.Ny), interpolation=cv2.INTER_NEAREST
        )
        wound_edge = cv2.resize(
            self.woundedge.astype(np.float32),
            (self.Nx, self.Ny),
            interpolation=cv2.INTER_LINEAR,
        ).astype(bool)

        nuc_mask = cv2.resize(
            self.labnucs, (self.Nx, self.Ny), interpolation=cv2.INTER_NEAREST
        )
        # do the binary opening via fft
        cell_nucs = fft_binary_opening(
            nuc_mask > 0, disk(nucleus_opening_radius)
        )

        for i in edge_id:
            # clean up cell_mask
            cell_mask = binary_closing(labeled_cells == i)
            cell_wound = skeletonize_3d(cell_mask * wound_edge).astype(bool)
            proper_edge = label(cell_wound).max() == 1

            if proper_edge:
                rmin, rmax, cmin, cmax = bbox2(cell_mask)
                ri = rmin - 2
                rf = rmax + 3
                ci = cmin - 2
                cf = cmax + 3
                # bounding box center for reference
                centroid = (ri, ci)
                # pad edges
                mask_ = cell_mask[ri:rf, ci:cf]
                data_ = self.data[ri:rf, ci:cf] * cell_mask[ri:rf, ci:cf, None]
                wound_ = cell_wound[ri:rf, ci:cf]
                # if multiple nuclei are present, we want to label each one
                nuc_ = label(mask_ * cell_nucs[ri:rf, ci:cf])
                yield i, data_, mask_, nuc_, wound_, centroid
            else:
                pass

    def run_analysis(self, img_tag):
        """do single-cell analysis for this ImageField"""
        # for mtoc stats
        datacol = []
        # for cell stats
        cellcol = []

        for i, d, m, n, w, yxoffset in self.edge_cells():

            try:
                ec = EdgeCell(i, d, m, n, w)
            except UnboundLocalError:
                logger.info(
                    f"Cell #{i} in {img_tag} does not have a proper wound edge. Skipping"
                )
                continue

            celly, cellx = ec.cellprops[0].centroid

            if ec.nuclei_num == 1:
                oy, ox = ec.nucleus_centroid
            elif ec.nuclei_num >= 2:
                logger.info(
                    f"Cell {i:d} in {img_tag} has {ec.nuclei_num} nuclei. Skipping this cell"
                )
                continue
            elif ec.nuclei_num == 0:
                logger.info(
                    f"Cell {i:d} in {img_tag} has no detected nuclei. Skipping this cell"
                )
                continue

            # compute orientation

            # box size for integrating tubulin intensities, default 1.5 um box
            tub_box_size = int(1.5 // self.dxy)
            # enforce odd-sized box (add one if size is even)
            odd_offset = 1 if (tub_box_size % 2 == 0) else 0
            tub_box_size += odd_offset

            p, tub_ints, oris = ec.get_mtoc_orientation(
                channel=self.cyt_ch,
                tubulin_channel=self.tub_ch,
                tub_box=tub_box_size,
            )

            # compute nucleus orientation
            nori = ec.nucprops[0].orientation
            nucmajlength = ec.nucprops[0].axis_major_length
            nucminlength = ec.nucprops[0].axis_minor_length
            nucy_major = np.cos(nori) * nucmajlength / 2.0
            nucx_major = np.sin(nori) * nucmajlength / 2.0
            nucy_minor = np.sin(nori) * nucminlength / 2.0
            nucx_minor = np.cos(nori) * nucminlength / 2.0

            # nori_ma = relative_angle(ec.ma, (nucy_major, nucx_major))
            # use cosine convention to measure nucleus orientation (due to
            # 2 2-fold symmetry axes, we only need 0 to pi/2.
            norm_ma = ec.ma / np.linalg.norm(ec.ma)
            major_nucvec = np.array([nucy_major, nucx_major])
            norm_nuc = major_nucvec / np.linalg.norm(major_nucvec)
            nori_ma = np.abs(np.arccos(np.dot(norm_ma, norm_nuc))) % np.pi / 2

            # after orientation is computed, we can access '.ma' attribute
            # which is for 'migration axis'

            cellentry = {
                "Cell #": i,
                "cell_x": celly + yxoffset[1],
                "cell_y": cellx + yxoffset[0],
                "migration_x": ec.ma[1] + ox + yxoffset[1],
                "migration_y": ec.ma[0] + oy + yxoffset[0],
                "wound_length": ec.single_edge.sum() * self.dxy,
                "cell_perimeter": ec.cellprops[0].perimeter * self.dxy,
                "equivalent_diameter": ec.cellprops[0].equivalent_diameter_area
                * self.dxy,
                "nucleus_diameter": ec.nucprops[0].equivalent_diameter_area
                * self.dxy,
                "nucleus_orientation": np.rad2deg(nori_ma),
                "nucleus_x": ox + yxoffset[1],
                "nucleus_y": oy + yxoffset[0],
                "nucleus_major_axis_x": ox - nucx_major + yxoffset[1],
                "nucleus_major_axis_y": oy - nucy_major + yxoffset[0],
                "nucleus_minor_axis_x": ox + nucx_minor + yxoffset[1],
                "nucleus_minor_axis_y": oy - nucy_minor + yxoffset[0],
            }
            cellcol.append(cellentry)

            # form convex hull of the cell "front"
            _img = ec.single_edge
            _img[int(oy), int(ox)] = True
            cone = convex_hull_image(_img)

            for mtoc_loc, tub_intensity, ori in zip(p, tub_ints, oris):
                on_nucleus = n[mtoc_loc[0], mtoc_loc[1]]
                entry = {
                    "Cell #": i,
                    "tubulin_intensity": tub_intensity,
                    "orientation": ori,
                    "on_nucleus": on_nucleus,
                    "classic_alignment": cone[mtoc_loc[0], mtoc_loc[1]],
                    "x": mtoc_loc[1] + yxoffset[1],
                    "y": mtoc_loc[0] + yxoffset[0],
                }
                datacol.append(entry)

        # convert mtoc data into DataFrame
        df = pd.DataFrame(datacol)

        mother_ids = df.loc[:, "tubulin_intensity"] == df.groupby("Cell #")[
            "tubulin_intensity"
        ].transform("max")
        df.loc[:, "mtoc_identity"] = "daughter"
        df.loc[mother_ids, "mtoc_identity"] = "mother"
        df.insert(0, "filename", img_tag)

        df2 = pd.DataFrame(cellcol)
        df2.insert(0, "filename", img_tag)

        return df, df2


class Cell:
    def __init__(self, cell_id, data, cytomask, nucmask):
        self.id = cell_id
        self.data = data
        self.cytomask = cytomask
        self.nucmask = nucmask
        self.compute_basic_properties()

    def compute_basic_properties(self):
        self.cellprops = regionprops(self.cytomask.astype(int))
        self.nucprops = regionprops(self.nucmask)

    def RGB_composite(self):
        return makeRGBComposite(self.data, ch_axis=-1)

    @property
    def nucleus_centroid(self):
        if self.nuclei_num == 1:
            nucy, nucx = self.nucprops[0].centroid
            return nucy, nucx
        else:
            return None

    @property
    def nuclei_num(self):
        return self.nucmask.max()

    def get_mtoc_locs(self, channel=1):
        p = peak_local_max(
            self.data[:, :, channel],
            num_peaks=8,
            min_distance=3,
            threshold_rel=0.6,
        )
        return p


class EdgeCell(Cell):
    def __init__(self, cell_id, data, cytomask, nucmask, woundedge):
        super().__init__(cell_id, data, cytomask, nucmask)
        self.woundedge = woundedge
        self.single_edge, self.edge_endpts = trim_skeleton_to_endpoints(
            woundedge
        )
        self.endpoints_computed = False

    def compute_migration_axis(self):
        oy, ox = self.nucleus_centroid
        sortededge = sort_edge_coords(self.single_edge, self.edge_endpts[0])
        _dy = sortededge[:, 0] - oy
        _dx = sortededge[:, 1] - ox
        self.distweights = np.sqrt(_dy * _dy + _dx * _dx)
        normweights = self.distweights / self.distweights.sum()
        maxis_index = int(
            np.sum(np.arange(self.distweights.size) * normweights)
        )
        my, mx = sortededge[maxis_index, :]
        return np.array([my - oy, mx - ox])

    def get_mtoc_orientation(self, channel=1, tubulin_channel=2, tub_box=5):
        p = self.get_mtoc_locs(channel=channel)
        _data = self.data[:, :, tubulin_channel]
        w = tub_box // 2
        tub_intensities = np.array(
            [
                _data[r - w : r + (w + 1), c - w : c + (w + 1)].sum()
                for r, c in p
            ]
        )

        oy, ox = self.nucleus_centroid
        # compute angle w.r.t migration axes
        self.ma = self.compute_migration_axis()
        orientation = np.rad2deg(
            np.array([relative_angle((y - oy, x - ox), self.ma) for y, x in p])
        )

        return p, tub_intensities, orientation


def remove_small_labels(labeled_img, area_threshold=400):
    labels = np.unique(labeled_img)
    wrk = np.copy(labeled_img)
    for n in labels:
        roi = labeled_img == n
        objectmask = remove_small_objects(roi, area_threshold).astype(bool)
        wrk[roi] *= objectmask[roi]
    return wrk


def sort_edge_coords(skeletonized_edge, endpoint):
    """routine to sort y,x coordinates of skeletonized edge

    Args:
        skeletonized_edge (2-d bool array): skeletonized edge image
        endpoint (2-tuple of y,x): endpoint coordinate

    Returns:
        2-d array (N x 2), rc coordinate

    """

    ydir = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    xdir = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    pos = np.array(endpoint)

    numel = skeletonized_edge.sum()

    wrkimg = skeletonized_edge.copy()

    # preallocate output array
    sorted_edge = np.zeros((numel, 2), dtype=int)

    curpos = pos.copy()
    sorted_edge[0, :] = curpos
    # define pixel counter and start loop
    i = 0

    while True:
        i += 1
        wrkimg[curpos[0], curpos[1]] = 0
        sbox = wrkimg[
            curpos[0] - 1 : curpos[0] + 2, curpos[1] - 1 : curpos[1] + 2
        ]
        if sbox.sum() == 0:
            break
        # move current position
        curpos[0] += ydir[sbox][0]
        curpos[1] += xdir[sbox][0]
        sorted_edge[i, :] = curpos

    return sorted_edge


def relative_angle(v, ref):
    """compute relative angle between vectors

    positive : counter-clockwise
    negative : clockwise
    https://wumbo.net/formula/angle-between-two-vectors-2d/

    assuming that v = (y, x)
    """
    return np.arctan2(
        v[0] * ref[1] - v[1] * ref[0], v[1] * ref[1] + v[0] * ref[0]
    )


def get_indexer(img, ch_axis, ch_slice):
    """returns slice objects for indexing"""
    # slicer object to get all elements
    alldims = slice(None, None, None)

    # accommodate numpy notation
    if ch_axis < 0:
        ch_idx = img.ndim + ch_axis
    else:
        ch_idx = ch_axis

    # for getting the correct slices
    slice_id = []

    for d in range(img.ndim):
        if d != ch_idx:
            slice_id.append(alldims)
        else:
            slice_id.append(ch_slice)

    return tuple(slice_id)


def normalize(img):
    """normalize input array to its dynamic range"""
    dynrange = img.max() - img.min()
    return (img - img.min()) / dynrange


def norm_to_8bit(img):
    return np.uint8(normalize(img) * 255)


# internal function used in trimming skeletonized image
def __get_last_coordinates(dict_endpoints):
    N = len(dict_endpoints.keys())
    return np.array([dict_endpoints[i][-1] for i in range(N)])


# kernel for detecting endpoints in 2D skeletonized image
endpoint_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)


def __find_endpoints(img):
    endpt_response = convolve2d(
        img.astype(np.uint8), endpoint_kernel, mode="same"
    )
    endpts = np.where(endpt_response == 11)
    return endpts


def trim_skeleton_to_endpoints(skelimg, n_ends=2):

    epts = __find_endpoints(skelimg)
    dict_eps = {i: [pt] for i, pt in enumerate(list(zip(*epts)))}
    wrk = skelimg.copy()

    if len(epts[0]) == n_ends:
        epts = tuple(zip(*epts))
        return skelimg, epts

    else:
        while len(epts[0]) > n_ends:
            wrk[epts] = 0
            a1 = __get_last_coordinates(dict_eps)
            epts = __find_endpoints(wrk)
            a2 = np.array(epts).T
            pwdist = cdist(a1, a2)
            eid_ = pwdist.argmin(axis=0)
            for id_, yx in zip(eid_, a2):
                dict_eps[id_].append((yx[0], yx[1]))

        # flatten the list of coordinates
        survived_ends = list(chain(*[dict_eps[i] for i in eid_]))
        survived_ends_id = tuple(i for i in np.array(survived_ends).T)

        # re-fill erased skeleton pixels
        wrk[survived_ends_id] = 1

        survived_epts = tuple([dict_eps[i][0] for i in eid_])
        return wrk, survived_epts


def find_back_pos(cellmask, front_pos, centroid):
    """find the back of the cell

    Args:
        cellmask (boolean 2d np.array): cell mask
        front_pos (2-tuple or 2-element np.array): (y,x) coordinate of migration front
        centroid (2-tuple or 2-element np.array): (y,x) coordinate of centroid. For example,
        can be the nucleus or the cell centroid. Distance map will be computed
        fromt this position.

    Return:
        angle_map, distance_map, back_position

    """
    # get mask indices (coordinates)
    ymask, xmask = np.where(cellmask)

    # dy, dx w.r.t from edge to centroid
    dy = front_pos[0] - centroid[0]
    dx = front_pos[1] - centroid[1]

    # calculate the migration edge w.r.t input centroid
    theta = np.arctan2(dy, dx)

    # compute angles wrt center and edge
    angle_vals = np.arctan2(ymask - centroid[0], xmask - centroid[1])
    angle_map = np.zeros(cellmask.shape)
    # adjust angle by theta
    adj_angle_vals = angle_vals - theta
    # wrap angles to -pi,pi (the opposite of np.unwrap)
    angle_map[ymask, xmask] = (adj_angle_vals + np.pi) % (2 * np.pi) - np.pi

    # compute distance transform
    boolarr_origin = np.zeros(cellmask.shape, dtype=np.uint8)
    boolarr_origin[int(centroid[0]), int(centroid[1])] = True
    dist_map = distance_transform_edt(np.logical_not(boolarr_origin))

    # find pixels where angle is near 180˚ or π
    angle_devs = np.abs(angle_map - np.pi)
    crit1 = angle_devs < 1e-2
    crit2 = dist_map * crit1
    back_pos = np.unravel_index(np.argmax(crit2), angle_devs.shape)

    return angle_map, dist_map, back_pos


def fft_binary_erosion(img, strel):
    res = fftconvolve(img, strel, mode="same")
    return res > (strel.sum() - 0.1)


def fft_binary_dilation(img, strel):
    res = fftconvolve(img, strel, mode="same")
    return res > 0.1


def fft_binary_opening(img, strel):
    wrk = fft_binary_erosion(img, strel)
    return fft_binary_dilation(wrk, strel)
