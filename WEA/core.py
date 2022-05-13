"""
Wound edge analysis modules and functions

"""
import logging
import torch
import re
import numpy as np
import cv2

from pathlib import Path
from cellpose import models
from scipy.ndimage import (
    convolve,
    binary_fill_holes,
    binary_closing,
)
from scipy.spatial.distance import cdist
from skimage.segmentation import clear_border, find_boundaries
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.transform import rescale
from skimage.feature import peak_local_max
from skimage.morphology import (
    convex_hull_image,
    remove_small_objects,
    binary_erosion,
    disk,
    skeletonize,
)
from itertools import chain
import pandas as pd

from .vis import makeRGBComposite
from . import __file__

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

# setup basic logging (may change directory later)
logging.basicConfig(
    filename=f"{__log_dir / 'WEA_dev.log'}",
    filemode="w",
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


# custom models are stored in $HOME/.cellpose/models
cytoengine = models.CellposeModel(
    gpu=use_gpu,
    pretrained_model=str(__model_dir / "CP_bcat-nuc_v01"),
)

nucengine = models.CellposeModel(
    gpu=use_gpu, pretrained_model=str(__model_dir / "CP_dapi_v01")
)


logging.info(f"Using models from {str(__model_dir)}")


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


def erode_labels(img, disk_radius=2):
    return sum(
        [
            binary_erosion(img == i, footprint=disk(disk_radius)) * i
            for i in range(1, img.max() + 1)
        ]
    )


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

    def _create_cellpose_input(
        self, input_cell_diam, target_cell_diam=100, downsize=True
    ):

        if downsize:

            downscale_factor = target_cell_diam / input_cell_diam
            self.downscale_factor = downscale_factor

            img = rescale(
                self.data,
                downscale_factor,
                channel_axis=-1,
                preserve_range=True,
                anti_aliasing=True,
            )
            scaled_dxy = self.dxy / downscale_factor
        else:
            img = self.data.astype(np.float32)
            scaled_dxy = self.dxy

        # create RGB input
        Ny, Nx = img.shape[0:2]
        cp_input = np.zeros(img.shape[0:2] + (3,), dtype=np.float32)
        cp_input[:, :, 1] = img[:, :, self.cyt_ch]
        cp_input[:, :, 2] = img[:, :, self.nuc_ch]

        return cp_input, scaled_dxy

    def segment_cells(self, celldiam=68.0, nucdiam=14.0, downsize=True):
        """resizes images to about 512x512 and run cellpose

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

        cmask, cflow, cstyle = cytoengine.eval(
            img,
            diameter=self._celldiam,
            resample=True,
            channels=[2, 3],
        )

        nmask, nflow, nstyle = nucengine.eval(
            img,
            diameter=self._nucdiam,
            resample=True,
            channels=[3, 0],
        )

        logging.info(f"Original image size is : {self.data.shape[0:2]}")
        logging.info(f"Cellpose input is resized to : {img.shape[0:2]}")
        logging.info("Segmentation run with parameters:")
        logging.info(
            f"cell diameter={celldiam/scaled_dxy:.2f} px, nucleus diameter={nucdiam/scaled_dxy:.2f} px."
        )

        self.cp_input = img
        self.cp_labcells = cmask
        # internal variables used to resample cell mask
        self._cyt_dP = cflow[1]
        self._cyt_cellprob = cflow[2]
        self.cp_labnucs = nmask

    def run_detection(self, cell_diam=68.0, nuc_diam=15.0, downsize=True):

        if self.cp_labcells is None:
            self.segment_cells(
                celldiam=cell_diam, nucdiam=nuc_diam, downsize=downsize
            )

        # identify wound edge
        cellarea = binary_fill_holes(self.cp_labcells > 0)
        woundarea = remove_small_objects(
            ~cellarea, min_size=self._celldiam**2
        )
        # we need this thick so that it overlaps with cell masks
        self.woundedge = find_boundaries(woundarea, mode="thick")

        # remove cells touching the border
        self.labcells = label(clear_border(self.cp_labcells))
        self.labnucs = self.labcells * (self.cp_labnucs > 0)

    def _segmentation_result(self):

        cell_boundaries = find_boundaries(self.labcells, mode="thin")
        nuc_boundaries = find_boundaries(self.labnucs, mode="thin")

        rgb_img = makeRGBComposite(self.cp_input, ch_axis=-1)

        rgb_img[cell_boundaries, 0] = 1.0
        rgb_img[cell_boundaries, 1] = 1.0
        rgb_img[nuc_boundaries, 2] = 1.0
        rgb_img[self.woundedge, 0] = 1.0

        return rgb_img

    def edge_cells(self, nucleus_erosion_radius=4):
        """iterator function that yields edge cells

        Each iteration yields ROI-cropped data:
        raw data, cytoplasm mask, wound edge, nucleus mask

        """

        # resize labels and wound edge
        Ncells = int(self.labcells.max())

        for i in range(1, Ncells + 1):
            # clean up cell_mask
            cell_mask = binary_closing(self.labcells == i)
            cell_wound = skeletonize(cell_mask * self.woundedge)
            on_edge = np.sum(cell_wound) > 0

            # rescale boxes images
            cell_mask = _rescale_mask(cell_mask, 1 / self.downscale_factor)
            cell_wound = _rescale_mask(cell_wound, 1 / self.downscale_factor)
            cell_nucs = erode_labels(
                _rescale_mask(self.labnucs, 1 / self.downscale_factor),
                disk_radius=nucleus_erosion_radius,
            )

            # cell_nucs = remove_small_objects(cell_nucs, 64)

            if on_edge:
                rmin, rmax, cmin, cmax = bbox2(cell_mask)
                ri = rmin - 2
                rf = rmax + 3
                ci = cmin - 2
                cf = cmax + 3
                # pad edges
                mask_ = cell_mask[ri:rf, ci:cf]
                data_ = self.data[ri:rf, ci:cf] * cell_mask[ri:rf, ci:cf, None]
                wound_ = cell_wound[ri:rf, ci:cf]
                nuc_ = label(cell_nucs[ri:rf, ci:cf] * cell_mask[ri:rf, ci:cf])
                yield i, data_, mask_, nuc_, wound_
            else:
                pass

    def run_analysis(self, img_tag):
        datacol = []

        for i, d, m, n, w in self.edge_cells(nucleus_erosion_radius=5):
            ec = EdgeCell(i, d, m, n, w)
            p, tub_ints, oris = ec.get_mtoc_orientation()
            oy, ox = ec.nucleus_centroid

            # form convex hull of the cell "front"
            _img = ec.single_edge
            _img[int(oy), int(ox)] = True
            cone = convex_hull_image(_img)

            for mtoc_loc, tub_intensity, ori in zip(p, tub_ints, oris):
                on_nucleus = n[mtoc_loc[0], mtoc_loc[1]]
                entry = {
                    "Cell #": i,
                    "mtoc_x": mtoc_loc[1],
                    "mtoc_y": mtoc_loc[0],
                    "tubulin_intensity": tub_intensity,
                    "orientation": ori,
                    "on_nucleus": on_nucleus,
                    "classic_alignment": cone[mtoc_loc[0], mtoc_loc[1]],
                }
                datacol.append(entry)

        df = pd.DataFrame(datacol)

        mother_ids = df.loc[:, "tubulin_intensity"] == df.groupby("Cell #")[
            "tubulin_intensity"
        ].transform("max")
        df.loc[:, "mtoc_identity"] = "daughter"
        df.loc[mother_ids, "mtoc_identity"] = "mother"

        df.insert(0, "filename", img_tag)

        return df


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
            skeletonize(woundedge)
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
            [_data[r - 2 : r + 3, c - 2 : c + 3].sum() for r, c in p]
        )

        oy, ox = self.nucleus_centroid
        # compute angle w.r.t migration axes
        ma = self.compute_migration_axis()
        orientation = np.rad2deg(
            np.array([relative_angle((y - oy, x - ox), ma) for y, x in p])
        )

        return p, tub_intensities, orientation


class _Cell__old:
    """an abstract cell representation

    Each cell contains its:
        - raw data (cropped by cytoplasmic mask)
        - cytoplasm, nucleus, and wound edge masks (if available)

    """

    def __init__(
        self,
        folder,
        num,
    ):
        self.root = Path(folder)
        cell_ptn = "cell_{:d}.tif"
        cytmask_ptn = "cell_mask_{:d}.tif"
        edgemask_ptn = "edge_mask_{:d}.tif"
        nucmask_ptn = "nucleus_mask_{:d}.tif"

        self.id = num
        self.data = imread(self.root / cell_ptn.format(num))
        self.cytoplasm_mask = imread(self.root / cytmask_ptn.format(num))
        self.nucleus_mask = imread(self.root / nucmask_ptn.format(num))

        # shrink the nucleus a bit to compensate for 'cyto' model mask
        self.nucleus_mask = np.uint8(
            binary_erosion(self.nucleus_mask > 0, disk(10))
        )

        edge = self.root.stem == "edge"
        self.endpoints_computed = False
        self.edge_endpts = None
        self.woundedge = edge

        if edge:
            # binarize the edge mask
            self.edge_mask = imread(self.root / edgemask_ptn.format(num)) > 0
            # trace the boundary of the cell before producing a single-pixel edge
            thin_cell_boundary = trace_object_boundary(self.cytoplasm_mask)
            # convert single_edge to integer for finding endpoints later
            self.single_edge = np.uint8(
                skeletonize(self.edge_mask * thin_cell_boundary)
            )

        self.compute_basic_properties()

    def compute_basic_properties(self):
        self.wound_length = self.single_edge.sum()
        self.cellprops = regionprops(self.cytoplasm_mask)
        self.nucprops = regionprops(self.nucleus_mask)

    def composite(self, z_axis=1, **kwargs):
        if self.data.shape[z_axis] > 1:
            # if there are more than one slices, do a max-projection
            wrk = self.data.max(axis=z_axis)
        elif self.data.ndim == 4:
            # if the z dimension is a singleton
            if self.data.shape[z_axis] == 1:
                wrk = self.data[0, ...]
            else:
                # otherwise just pass the array
                wrk = self.data

        return makeRGBComposite(wrk, **kwargs)

    def composite_with_edge(self, z_axis=1, **kwargs):
        if self.woundedge:
            wrk = self.composite(z_axis=z_axis, **kwargs)
            wrk[self.edge_mask, 0] = 1.0
            wrk[self.edge_mask, 1] = 1.0
        else:
            wrk = self.composite(z_axis=z_axis, **kwargs)
        return wrk

    @property
    def edge_length(self):
        if self.woundedge:
            return self.single_edge.sum()
        else:
            return 0

    @property
    def edge_endpoints(self):
        if self.woundedge:
            if not self.endpoints_computed:
                endpt_response = convolve(self.single_edge, endpt_kernel)
                yends, xends = np.where(endpt_response == 11)
                Nends = len(yends)
                errmsg = "There can only be 2 endpoints! Found {Nends:d}. "
                errmsg += "Cell {id:d} in {folder:s}."
                assert Nends == 2, errmsg.format(
                    Nends=Nends, id=self.id, folder=str(self.root)
                )
                endpt1 = (yends[0], xends[0])
                endpt2 = (yends[1], xends[1])

                self.edge_endpts = (endpt1, endpt2)
                self.endpoints_computed = True

                return endpt1, endpt2
            else:
                return self.edge_endpts
        else:
            return None

    @property
    def mtoc_locs(self, mtoc_channel=1, mt_channel=2):
        # do 3d max-projection
        if self.data.squeeze().ndim > 4:
            wrkimg = self.data[mtoc_channel, ...].max(axis=0)
        elif self.data.squeeze().ndim == 3:
            idx = get_indexer()
            wrkimg = self.data.squeeze()
        mlocs = find_2d_spots(wrkimg)
        if mlocs.ndim == 1:
            return mlocs[None, :]
        else:
            return mlocs

    @property
    def nucleus_centroid(self):
        nucy, nucx = regionprops(self.nucleus_mask)[0].centroid
        return nucy, nucx

    @property
    def core_vecs(self):
        pt1, pt2 = self.edge_endpoints
        nucy, nucx = self.nucleus_centroid
        e1 = np.array([pt1[0] - nucy, pt1[1] - nucx])
        e2 = np.array([pt2[0] - nucy, pt2[1] - nucx])
        midvec = (e1 + e2) / 2.0
        return e1, midvec, e2

    @property
    def _hw(self):
        # head width for drawing arrow (temporary)
        Nr, Nc = self.data.shape[-2:]
        _l = (Nr + Nc) / 2.0
        return _l / 20

    @property
    def _hl(self):
        # head length for drawing arrow (temporary)
        Nr, Nc = self.data.shape[-2:]
        _l = (Nr + Nc) / 2.0
        return _l / 10

    def compute_migration_axis(self):
        """returns position of migration axis on the wound edge

        computed as an 'area'-weighted direction from the nucleus centroid
        such that larger distances along the wound edge carries more weight
        in determining the migration axis.

        Returns:
            y,x position of along the wound edge
        """
        if self.woundedge:
            end1, end2 = self.edge_endpoints
            oriy, orix = self.nucleus_centroid
            # sorted wound edge from an endpoint
            swedge = sort_edge_coords(self.single_edge, end1)
            sdists = np.linalg.norm(swedge - np.array([oriy, orix]), axis=1)
            # compute distance from nucleus centroid as 'weights'
            edge_weights = sdists / sdists.sum()
            axis_id = int(np.sum(edge_weights * np.arange(sdists.size)))
            return swedge[axis_id] - self.nucleus_centroid
        else:
            return None

    def get_mtoc(self, tub_channel=2, aperture_radius=5):
        """returns centrosome locations and whether its within the nucleus

        'mother' vs 'daughter' is distinguished by having higher vs lower
        microtubule intensity with a 5-pixel radius

        'nuc' vs 'cyto' indicates whether the centriole is on the nucleus or
        in the cytoplasm

        Args:
            tub_channel (int): channel for tubulin intensities
            radius (int): radius of aperture used for integrating tubulin ch.

        Returns:
            Dictionary containing key pairs ('mother', 'nuc'/'cyto')
            with its (y,x) coordinate

        """

        mtoc_locs = self.mtoc_locs
        mt_intensities = []
        aperture = disk(aperture_radius)
        drow, dcol = [n // 2 for n in aperture.shape]
        on_nucleus = []

        for i, c in enumerate(mtoc_locs):
            yc, xc = c
            # check whether centriole is on nucleus
            on_nucleus.append(self.nucleus_mask[yc, xc] > 0)
            _boxed = self.data[tub_channel, ...].max(axis=0)[
                yc - drow : yc + drow + 1, xc - dcol : xc + dcol + 1
            ]
            tub_intensity = (_boxed * aperture).sum()
            mt_intensities.append(tub_intensity)

        # higher intensity --> mother centriole
        if len(mt_intensities) == 2:
            mother_id = mt_intensities.index(max(mt_intensities))
            daughter_id = mt_intensities.index(min(mt_intensities))
            mid = np.array([mother_id, daughter_id])
            return list(
                zip(
                    [mtoc_locs[i] for i in mid],
                    [on_nucleus[i] for i in mid],
                    ["mother", "daughter"],
                    [mt_intensities[i] for i in mid],
                )
            )
        else:
            mother_id = 0
            return [
                (mtoc_locs[0], on_nucleus[0], "only", mt_intensities[0]),
            ]

    def get_mtoc_orientation(self):
        migaxis = self.compute_migration_axis()
        mtocs = self.get_mtoc()
        origin = np.array(self.nucleus_centroid)

        # normalize coordinates wrt nucleus centroid
        migration_vector = migaxis - origin
        mtocs_out = []

        # each mtoc entry is
        # mtoc_coordinates, on_nucleus, mother_mtoc, tubulin_intensity)
        # (np.array, bool, str, int)
        for mtoc in mtocs:
            # for each mtoc, compute vector wrt nucleus centroid also
            mtoc_vector = mtoc[0] - origin
            relative_theta = relative_angle(mtoc_vector, migration_vector)
            # re-append the rest of the metadata about mtoc
            mtocs_out.append(
                (np.rad2deg(relative_theta), mtoc[1], mtoc[2], mtoc[3])
            )

        return mtocs_out

    def get_endpoints_orientation(self):
        origin = np.array(self.nucleus_centroid)
        migration_vector = self.compute_migration_axis() - origin
        endpts = self.edge_endpoints
        endangles = [
            relative_angle(endpt - origin, migration_vector) for endpt in endpts
        ]
        return [np.rad2deg(x) for x in endangles]


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


def __get_last_coordinates(dict_endpoints):
    N = len(dict_endpoints.keys())
    return np.array([dict_endpoints[i][-1] for i in range(N)])


endpoint_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)


def __find_endpoints(img):
    endpt_response = convolve(img.astype(np.uint8), endpoint_kernel)
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


def _rescale_mask(img, scale):
    Ly = int(scale * img.shape[0])
    Lx = int(scale * img.shape[1])
    if img.dtype == "bool":
        out = cv2.resize(
            img.astype(np.float32), (Lx, Ly), interpolation=cv2.INTER_LINEAR
        )
        return (out > 0).astype(img.dtype)
    elif img.dtype == "int":
        # used for resizing labeled masks
        out = cv2.resize(img, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        return out
