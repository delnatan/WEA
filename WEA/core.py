"""
Wound edge analysis modules and functions

"""
import logging
import torch
import re
import numpy as np
import cv2
from cellpose import models
from scipy.ndimage import (
    maximum_filter,
    convolve,
    binary_fill_holes,
    binary_closing,
)
from scipy.spatial.distance import cdist
from skimage.segmentation import clear_border, find_boundaries
from skimage.measure import label, regionprops
from skimage.io import imsave, imread
from skimage.transform import rescale
from skimage.morphology import (
    remove_small_objects,
    binary_dilation,
    binary_erosion,
    disk,
    skeletonize,
)
from itertools import chain
from pathlib import Path
import matplotlib.pyplot as plt

from .vis import makeCellComposite, makeRGBComposite, drawSegmentationBorder

logging.basicConfig(filename="WEA_dev.log", filemode="w", level=logging.DEBUG)


if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

# custom models are stored in $HOME/.cellpose/models
cytoengine = models.CellposeModel(
    gpu=use_gpu,
    pretrained_model="/Users/delnatan/.cellpose/models/CP_bcat-nuc_v01",
)

nucengine = models.CellposeModel(
    gpu=use_gpu, pretrained_model="/Users/delnatan/.cellpose/models/CP_dapi_v01"
)


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
            scaled_dxy = 1.0

        # create RGB input
        Ny, Nx = img.shape[0:2]
        cp_input = np.zeros(img.shape[0:2] + (3,), dtype=np.float32)
        cp_input[:, :, 1] = img[:, :, self.cyt_ch]
        cp_input[:, :, 2] = img[:, :, self.nuc_ch]

        return cp_input, scaled_dxy

    def segment_cells(self, celldiam=80.0, nucdiam=14.0):
        """resizes images to about 512x512 and run cellpose

        These defaults have been acquired from working with 3T3 cells. You
        should change them to suit your images

        Args:
            celldiam(float, optional): average diameter of the cell in micron.
            nucdiam(float, optional): average diameter of the nucleus in micron.


        """

        unscaled_cell_diam = celldiam / self.dxy

        img, scaled_dxy = self._create_cellpose_input(unscaled_cell_diam)

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
            compute_masks=True,
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

    def run_detection(self, cell_diam=68.0, nuc_diam=15.0):

        if self.cp_labcells is None:
            self.segmentCell(celldiam=cell_diam, nucdiam=nuc_diam)

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

    def edge_cells(self):
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
            cell_nucs = _rescale_mask(self.labnucs, 1 / self.downscale_factor)

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
                nuc_ = cell_nucs[ri:rf, ci:cf]
                yield i, data_, mask_, nuc_, wound_
            else:
                pass

    def _masks(self):
        """iterator which returns sampled-down masks"""
        Ncells = int(self.labcells.max())
        for i in range(1, Ncells + 1):
            cell_mask = self.labcells == i
            cell_wound = cell_mask * self.woundedge
            on_edge = np.sum(cell_wound) > 0
            if on_edge:
                rmin, rmax, cmin, cmax = bbox2(cell_mask)
                ri = rmin - 2
                rf = rmax + 3
                ci = cmin - 2
                cf = cmax + 3
                mask_ = cell_mask[ri:rf, ci:cf]
                nuc_ = (self.labnucs[ri:rf, ci:cf] > 0) * cell_mask[
                    ri:rf, ci:cf
                ]
                cell_boundary = trace_object_boundary(cell_mask[ri:rf, ci:cf])
                wound_ = skeletonize(
                    cell_boundary & self.woundedge[ri:rf, ci:cf]
                )

                yield i, mask_, nuc_, wound_
            else:
                pass


class Cell:
    def __init__(self, cell_id, data, cytomask, nucmask):
        self.id = cell_id
        self.data = data
        self.cytomask = cytomask
        self.nucmask = nucmask

    def compute_basic_properties(self):
        self.cellprops = regionprops(self.cytoplasm_mask)
        self.nucprops = regionprops(self.nucleus_mask)

    @property
    def nucleus_centroid(self):
        nucy, nucx = regionprops(self.nucleus_mask)[0].centroid
        return nucy, nucx


class EdgeCell(Cell):
    def __init__(self, cell_id, data, cytomask, nucmask, woundedge):
        super().__init__(cell_id, data, cytomask, nucmask)
        self.woundedge = woundedge
        self.single_edge_uint8 = np.uint8(skeletonize(self.woundedge))
        self.edge_endpts = None
        self.endpoints_computed = False

    @property
    def edge_endpoints(self):
        if not self.endpoints_computed:
            endpt_response = convolve(self.single_edge_uint8, endpt_kernel)
            yends, xends = np.where(endpt_response == 11)
            Nends = len(yends)
            errmsg = "There can only be 2 endpoints! Found {Nends:d}. "
            errmsg += "Cell {id:d} in {folder:s}."
            assert Nends == 2, errmsg.format(Nends=Nends, id=self.id)
            endpt1 = (yends[0], xends[0])
            endpt2 = (yends[1], xends[1])

            self.edge_endpts = (endpt1, endpt2)
            self.endpoints_computed = True

            return endpt1, endpt2
        else:
            return self.edge_endpts


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


def separateEdgeCells(
    rawimg,
    metadata,
    celldiam=65.0,
    nucdiam=14.0,
    shrink_by=0.25,
    empty_area_threshold=5000,
    output_prefix="wrk",
    output_path="WEA_analysis",
):
    """
    all units are expressed in micron (micron^2 for area).
    Metadata should contain a dictionary with at least these keys:
    "dxy", for pixel size information

    The output is saved at original resolution, but processing is done on
    resized images.

    """

    # do max-intensity projection along z-axis and resize if requested
    # channel, Nz, Ny, Nx
    img_is_3D = rawimg.ndim == 4

    if shrink_by != 1:
        metadata["dxy"] /= shrink_by

    if img_is_3D:
        # rescale image for faster processing
        if shrink_by != 1:
            img = rescale(rawimg, shrink_by, channel_axis=0).max(axis=1)
        else:
            img = rawimg.max(axis=1)
    else:
        if shrink_by != 1:
            img = rescale(rawimg, shrink_by, channel_axis=0)
        else:
            img = rawimg

    cpinput = makeCellComposite(img)

    # cell segmentation uses 2 channels for now
    cellmask = segmentCell(cpinput, metadata["dxy"], celldiam=celldiam)
    # nuclei can use grayscale image
    nucmask = segmentNucleus(img[0, :, :], metadata["dxy"], nucdiam=nucdiam)
    woundmask = isolateWoundArea(
        cellmask,
        empty_area_threshold=empty_area_threshold / (metadata["dxy"] ** 2),
    )
    labcells, labnucs, labedge = assignMasks(cellmask, nucmask, woundmask)

    # identify edge cells
    Ncells = labcells.max()

    # save results under subfolder
    outroot = Path(output_path) / output_prefix
    outedge = outroot / "edge"
    outintern = outroot / "nonedge"

    # create folders if they don't exist
    if not outedge.exists():
        outedge.mkdir(exist_ok=True, parents=True)
    if not outintern.exists():
        outintern.mkdir(exist_ok=True, parents=True)
    if not outroot.exists():
        outroot.mkdir(exist_ok=True, parents=True)

    # save overlay image
    overlay = makeRGBComposite(img, log_compress=(False, False, False))
    # then draw annotations on top
    annotated = drawSegmentationBorder(overlay, labcells)
    annotated = drawSegmentationBorder(annotated, labnucs, border_hue=0.15)
    annotated = drawSegmentationBorder(
        annotated, labedge, border_hue=0.65, border_sat=0.2, already_edge=True
    )
    # imsave(outroot / "overlay.png", np.uint8(255 * annotated))

    # get centroids of the cells
    coclist = [[p.label, p.centroid] for p in regionprops(labcells)]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(annotated)

    # label each cell at each centroid
    for (lbl, centroid) in coclist:
        _yc, _xc = centroid
        ax.text(_xc, _yc, f"{lbl}", fontsize=14, color="white")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outroot / "overlay_annotated.png")
    plt.close(fig)

    zoomfactor = 1 / shrink_by

    # crop out / cookie-cut original data
    for i in range(1, Ncells + 1):
        edgeindicator = (labcells == i) & (labedge == i)
        edge_cell = edgeindicator.sum() > 0
        cmask = np.uint8(rescale(labcells == i, zoomfactor))
        nmask = np.uint8(rescale(labnucs == i, zoomfactor))
        emask = np.uint8(rescale(labedge == i, zoomfactor))
        # get bounding box for cell
        rmin, cmin, rmax, cmax = [val for val in regionprops(cmask)[0].bbox]
        # do this on the original data, not the max-projection that was used
        # for doing the segmentation
        # add padding to subregion to give room for masks (so that masks don't
        # end up touching the image border)
        rmin -= 2
        rmax += 2
        cmin -= 2
        cmax += 2

        if img_is_3D:
            subregion = rawimg[:, :, rmin:rmax, cmin:cmax]
        else:
            subregion = rawimg[:, rmin:rmax, cmin:cmax]

        # figure out whether cell is an edge cell
        if edge_cell:
            imsave(
                outedge / f"cell_{i:d}.tif",
                subregion * cmask[None, None, rmin:rmax, cmin:cmax],
                check_contrast=False,
            )
            imsave(
                outedge / f"cell_mask_{i:d}.tif",
                cmask[rmin:rmax, cmin:cmax] * 255,
                check_contrast=False,
            )
            imsave(
                outedge / f"nucleus_mask_{i:d}.tif",
                nmask[rmin:rmax, cmin:cmax] * 255,
                check_contrast=False,
            )
            imsave(
                outedge / f"edge_mask_{i:d}.tif",
                emask[rmin:rmax, cmin:cmax] * 255,
                check_contrast=False,
            )
        else:
            imsave(
                outintern / f"cell_{i:d}.tif",
                subregion * cmask[None, None, rmin:rmax, cmin:cmax],
                check_contrast=False,
            )
            imsave(
                outintern / f"cell_mask_{i:d}.tif",
                cmask[rmin:rmax, cmin:cmax] * 255,
                check_contrast=False,
            )
            imsave(
                outintern / f"nucleus_mask_{i:d}.tif",
                nmask[rmin:rmax, cmin:cmax] * 255,
                check_contrast=False,
            )

    return labcells, labnucs, labedge


def segmentCell(imglist, dxy, celldiam=65.0):
    """segment cell using Cellpose

    Args:
        inputimg (a list of RGB images):
            input images with dimensions Nr x Nc x Nch.
            Only channels 2 & 3 (green & blue) are used.
        celldiam (float): average cell diameter in micron

    Return:
        binary mask of cells
    """

    masks, flows, styles = cytoengine.eval(
        imglist, diameter=celldiam / dxy, resample=True, channels=[2, 3]
    )

    return masks


def segmentNucleus(imglist, dxy, nucdiam=14.0):
    """segment nuclei using Cellpose

    Args:
        inputimg (a list of grayscale images): input images with dimensions Nr x Nc.
        nucdiam (float): average diameter of a nuclei in micron

    Return:
        binary mask of nuclei

    """

    masks, flows, styles, diams = nucengine.eval(
        imglist,
        diameter=nucdiam / dxy,
        resample=True,
        channels=[0, 0],
    )

    return masks


def isolateWoundArea(labcells, empty_area_threshold=1e4):
    """identify 'wound' edge from labelled cells

    Wound edge is defined as a non-cell area that is larger than approximately
    100x100 pixels. The wound is identified as the the dilated wound area
    that intersects with the cells.

    Args:
        labcells (array of int): labelled cell segmentation (from Cellpose)

    Returns:
        binary mask of the wound edge (~2 pixel thick)

    """
    cells = labcells > 0
    wound = ~cells
    # approximately 100x100 pixels are considered "small"
    wound = remove_small_objects(wound, empty_area_threshold)
    # dilate the wound area so that it touches the cells
    expandedwound = binary_dilation(wound, footprint=disk(2))
    return cells & expandedwound


def assignMasks(labcells, labnuclei, woundedge, small_cell_threshold=400):
    """assign labels to wound edge and nuclei based on cell segmentation

    Cells that are located at the boundary (truncated by the border) is thrown
    out. Spurious objects with less than <400 px area (small "cells") area also
    removed by default.

    Args:
        labcells (array of int): labelled cells (cellpose output)
        labnuclei (array of int): labelled nuclei (cellpose output)
        woundedge (array of bool): wound edge mask (generated from `isolateWoundArea`)

    Return:
        Labelled cells, labelled nuclei and labelled wound edge

    """
    # remove cells at the boundary and relabel
    wrk = clear_border(labcells)
    wrk = remove_small_labels(wrk, area_threshold=small_cell_threshold)
    refinedlabcells = label(wrk)
    assignededge = np.uint8(woundedge) * refinedlabcells
    assignednuclei = np.uint8(labnuclei > 0) * refinedlabcells
    return refinedlabcells, assignednuclei, assignededge


def remove_small_labels(labeled_img, area_threshold=400):
    labels = np.unique(labeled_img)
    wrk = np.copy(labeled_img)
    for n in labels:
        roi = labeled_img == n
        objectmask = remove_small_objects(roi, area_threshold).astype(bool)
        wrk[roi] *= objectmask[roi]
    return wrk


def compute_s_values(img, lmlocs):
    """computes `characteristic value` of each local maxima

    Characteristic value is the `curvature` times the mean intensity.
    Calculated according to the Thomann, D, et al. 2002 paper.
    This algorithm is easily extensible to 3D. Just need to form the 3x3
    hessian matrix and compute determinant with a routine.

    Args:
        img (2d-array): input image
        lmlocs (Nx2 array): coordinates of local maxima (Nrows x Ncolumns)

    Returns:
        array of s-values (Nx1 array)

    """
    Nlocs = len(lmlocs)
    s_vals = []
    for i in range(Nlocs):
        rloc, cloc = lmlocs[i, :]
        peakimg = img[rloc - 2 : rloc + 3, cloc - 2 : cloc + 3]
        try:
            dy, dx = np.gradient(peakimg)
        except ValueError:
            print("debug: ", peakimg.shape)
            raise
        dxy, dxx = np.gradient(dx)
        dyy, dyx = np.gradient(dy)
        hess = dxy[2, 2] * dyx[2, 2] - dxx[2, 2] * dyy[2, 2]
        s_vals.append(hess * peakimg.mean())
    return np.array(s_vals)


def find_2d_spots(img):
    """finds local maxima based on local curvature and intensity

    Args:
        img (2d-array): input image

    Returns:
        coordinates local maxima in (row, column) coordinates
    """

    # do initial maxima picking
    maxfiltimg = maximum_filter(img, size=(3, 3))
    local_max = (img == maxfiltimg) & (img > 0)
    # clear borders
    local_max[0:3, :] = False
    local_max[:, 0:3] = False
    local_max[-3:, :] = False
    local_max[:, -3:] = False

    # coordinates of local maxima
    local_max_locs = np.where(local_max)
    local_max_ints = img[local_max_locs]
    top15_thres = np.quantile(local_max_ints, 0.85)

    # redo local maxima picking
    local_max = (img == maxfiltimg) & (img > top15_thres)
    local_max[0:3, :] = False
    local_max[:, 0:3] = False
    local_max[-3:, :] = False
    local_max[:, -3:] = False
    local_max_locs = np.where(local_max)
    local_max_ints = img[local_max_locs]

    # argsort returns lowest -> highest, I like the reverse
    local_max_sorted_ids = np.argsort(local_max_ints)[::-1]
    # sort local maxima from lowest to highest
    # and transpose to get r,c as columns and each maximum along rows
    sorted_max_locs = np.array(
        (
            local_max_locs[0][local_max_sorted_ids],
            local_max_locs[1][local_max_sorted_ids],
        )
    ).T

    svals = compute_s_values(img, sorted_max_locs)
    absvals = np.abs(svals)
    # take the top 2 highest curvature
    sorted_svals_id = np.argsort(absvals)[::-1]
    true_locs = sorted_max_locs[sorted_svals_id[0:2], :]
    top2_absvals = absvals[sorted_svals_id[0:2]]

    # only return one if the other spot is weakly peaked by orders of magnitude
    if (top2_absvals[0] / top2_absvals[1]) > 10.0:
        return true_locs[0, :]
    else:
        return true_locs


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


def find_endpoints(edgeimg, debug=True):
    """finds endpoints of a skeletonized edge

    Usage:
        e1, e2 = find_endpoints(skeletonize(wound_edge))

    Returns:
        list of vector (y,x) for each endpoint

    """
    endpt_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    endpt_response = convolve(edgeimg.astype(np.uint8), endpt_kernel)
    endpts = np.where(endpt_response == 11)

    if debug:
        return endpt_response
    else:
        return [np.array(e) for e in list(zip(*endpts))]


def trace_object_boundary(bwimg):
    """trace binary object boundary

    Using algorithm in Cris Luengo's blog: https://www.crisluengo.net/archives/324/
    Note: assumes there's only one object in image, so it works only with
    isolated masks.

    """
    # first one is to the right, rotating counter clockwise
    directions = np.array(
        [[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
    )
    allindx = np.nonzero(bwimg)
    start_indx = np.array([allindx[0][0], allindx[1][0]])
    sz = bwimg.shape
    cc = []
    coord = start_indx
    ccdir = 0

    while True:
        # increment coordinate with direction
        newcoord = coord + directions[ccdir, :]
        # if new coordinate is within an image and is part of the object
        if (
            np.all(newcoord >= 0)
            and np.all(newcoord < sz)
            and bwimg[newcoord[0], newcoord[1]]
        ):
            # add to chain code
            cc.append(ccdir)
            # assign as current coordinate
            coord = newcoord
            # do a 90-degree turn
            ccdir = (ccdir + 2) % 8
        else:
            # flip direction
            ccdir = (ccdir - 1) % 8

        # if current coordinate is back at the start, then quit
        if np.all(coord == start_indx) and ccdir == 0:
            break

    coord_arr = np.vstack([start_indx[None, :], directions[cc]])

    boundary_id_arr = np.cumsum(coord_arr, axis=0)[0:-1, :]

    yx_indx = (boundary_id_arr[:, 0], boundary_id_arr[:, 1])
    # npixels = boundary_id_arr.shape[0]

    cellboundary = np.zeros_like(bwimg)
    cellboundary[yx_indx] = True

    return cellboundary


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
        out = cv2.resize(img.astype(np.float32), (Lx, Ly), interpolation=cv2.INTER_LINEAR)
        return (out > 0).astype(img.dtype)
    elif img.dtype == "int":
        # used for resizing labeled masks
        out = cv2.resize(img, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        return out
