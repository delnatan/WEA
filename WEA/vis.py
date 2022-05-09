"""
visualization of images
"""
import numpy as np
from matplotlib.colors import (
    rgb_to_hsv,
    hsv_to_rgb,
    LinearSegmentedColormap,
)
from skimage.morphology import binary_erosion, binary_dilation, disk
from scipy.ndimage import map_coordinates, rotate
from colorcet import gouldian

gouldian_cmap = LinearSegmentedColormap.from_list("gouldian", gouldian)


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


def random_colormap(N):
    """generate random colormap for N categories

    Use as matplotlib colormap. 0 is mapped as 'black'

    Args:
        N (int): number of categories

    """
    lo, hi = 0.6, 0.95
    randRGBcolors = [
        (
            np.random.uniform(low=lo, high=hi),
            np.random.uniform(low=lo, high=hi),
            np.random.uniform(low=lo, high=hi),
        )
        for i in range(N)
    ]

    # first color is black (for black background)
    randRGBcolors[0] = [0, 0, 0]
    cmap = LinearSegmentedColormap.from_list("new_map", randRGBcolors, N=N)
    return cmap


def makeRGBComposite(
    img,
    ch_axis=0,
    ch_hues=(195.0, 100.0, 320.0),
    ch_sats=None,
    clip_low=None,
    clip_high=None,
    log_compress=(False, False, False),
):
    """create a color image composite using HSV specifications

    Args:
        img (numpy 3-D array): input multi-channel image
        ch_axis (int): the channel index of multi-channel image
        ch_hues (list of float): ranging from 0-360. Must have equal # as ch
        ch_sats (list of float): ranging from 0-100. Must have equal # as ch
        log_compress (list of bool): whether to do a log compression on channel

    Return:
        color image in (normalized range, [0-1]) RGB space

    """

    Nch = img.shape[ch_axis]

    if ch_sats is None:
        ch_sats = [100.0 for i in range(Nch)]

    if clip_low is None:
        lo = [0.0 for i in range(Nch)]
    else:
        lo = clip_low
    if clip_high is None:
        hi = [100.0 for i in range(Nch)]
    else:
        hi = clip_high

    assert len(ch_hues) == Nch, "Number of channels must match given hues"

    assert (
        len(ch_sats) == Nch
    ), "Number of channels must match given saturations"

    # normalize hues & saturation so that [0,1]
    schues = [c / 360.0 for c in ch_hues]
    scsats = [s / 100.0 for s in ch_sats]

    # create a 'grayscale' rgb image
    rgbimgs = []

    for c in range(Nch):
        indexer = get_indexer(img, ch_axis, c)
        # adjust visual dynamic range
        wrk = img[indexer]

        if wrk.max() - wrk.min() < 1e-6:
            rgbimgs.append(np.stack((wrk,) * 3, axis=-1))
            continue

        _floor = np.percentile(wrk, lo[c])
        _ceiling = np.percentile(wrk, hi[c])
        wrk = np.minimum(np.maximum(wrk, _floor), _ceiling)

        if log_compress[c]:
            # avoid taking the log of zero
            wrk = normalize(np.log(np.maximum(wrk, 1e-8)))
        else:
            wrk = normalize(wrk)

        rgbimgs.append(np.stack((wrk,) * 3, axis=-1))

    # convert to HSV space
    hsvimgs = [rgb_to_hsv(rgbimg) for rgbimg in rgbimgs]
    # assign color to each channel via setting hues & saturation

    for c in range(Nch):
        hsvimgs[c][:, :, 0] = schues[c]
        hsvimgs[c][:, :, 1] = scsats[c]

    # convert back to rgb space
    rgbs = [hsv_to_rgb(hsvimg) for hsvimg in hsvimgs]
    # make the composite by averaging the rgb images
    composite = np.mean(rgbs, axis=0)
    # normalize to each channel maximal values
    chmaxval = composite.max(axis=(0, 1))
    return composite / chmaxval


def drawSegmentationBorder(
    rgbimg,
    labels,
    border_hue=0.0,
    border_sat=1.0,
    thickness=1,
    already_edge=False,
):
    """draw segmentation result as a border"""
    Nlabels = labels.max()
    hsvimg = rgb_to_hsv(rgbimg)

    for n in range(Nlabels):
        mask = labels == (n + 1)
        if not already_edge:
            # create border from mask by doing dilated - eroded masks
            dilated = binary_dilation(mask, footprint=disk(thickness))
            eroded = binary_erosion(mask, footprint=disk(thickness))
            outline = dilated ^ eroded
        else:
            outline = mask
        hsvimg[outline == 1, 0] = border_hue
        hsvimg[outline == 1, 1] = border_sat
        hsvimg[outline == 1, 2] = 1.0

    return hsv_to_rgb(hsvimg)


def normalize(arr):
    """normalizes array from 0 to 1 from its range

    Args:
        arr (array): input array

    Returns:
        normalized array

    """
    arrdrange = arr.max() - arr.min()
    return (arr - arr.min()) / arrdrange


def norm28bit(arr):
    """normalize and convert to 8-bit array

    Input is normalized from 0 to 1 and then multiplied by 255 before
    casting all elements into np.uint8

    Args:
        arr (array): input array

    Returns:
        normalized np.uint8 array
    """
    return np.uint8(normalize(arr) * 255)


def makeCellComposite(img):
    # green - beta-catenin; take the log to compress dynamic range before normalizing
    bcat = norm28bit(np.log(img[1, :, :]))
    # blue - nucleus
    nucleus = norm28bit(img[0, :, :])
    rgbcomp = np.zeros(img.shape[1:] + (3,), dtype=np.uint8)
    rgbcomp[:, :, 1] = bcat
    rgbcomp[:, :, 2] = nucleus
    return rgbcomp


def radial_resampling(img2d, orix, oriy, start_deg=0, n_thetas=360):
    """radially sample an array from a given origin

    The sampling radius is calculated from the longest distance from origin
    to the corners of the image. The longest radius is used for sampling.

    """
    # figure out how long the radius of resampling should be
    corners = np.array(
        [[0, 0], [0, img2d.shape[-2]], [0, img2d.shape[-1]], img2d.shape[-2:]]
    )
    # compute distance from corners to the given origin
    cornerdists = np.sqrt(
        np.sum((corners - np.array([oriy, orix])) ** 2, axis=1)
    )
    rmax = np.ceil(cornerdists.max())
    # setup radial vectors
    radvec = np.arange(rmax)
    thetavec = np.linspace(0, 2 * np.pi, num=n_thetas)
    dtheta = thetavec[1] - thetavec[0]
    offset = n_thetas - int(np.deg2rad(start_deg) / dtheta)
    thetas = np.roll(thetavec, offset)
    yvec = radvec[:, None] * np.sin(thetas[None, :]) + oriy
    xvec = radvec[:, None] * np.cos(thetas[None, :]) + orix

    Nrads = radvec.size
    Nthetas = n_thetas
    yxvec = np.vstack([yvec.ravel(), xvec.ravel()])
    resampled = map_coordinates(
        img2d, yxvec, cval=0.0, prefilter=False
    ).reshape((Nrads, Nthetas))

    return resampled


def visualize_alignment(ax, cell, lo_clip=0.0, hi_clip=100.0):
    oriy, orix = cell.nucleus_centroid
    maxis = cell.compute_migration_axis()
    ax.imshow(cell.composite_with_edge(clip_low=lo_clip, clip_high=hi_clip))
    # ax.imshow(cell.nucleus_mask, alpha=0.2)
    ax.arrow(
        orix,
        oriy,
        maxis[1] - orix,
        maxis[0] - oriy,
        fc="w",
        ec="w",
        head_length=20,
        head_width=15,
        length_includes_head=True,
    )

    try:
        for cpos, nuc, ident, _ in cell.get_mtoc():
            # red if on nucleus
            nuc_id = 1 if nuc else 0

            if ident == "mother":
                col = ("#34cf9d", "r")[nuc_id]
            else:
                col = ("#91ffcc", "r")[nuc_id]
            ax.arrow(
                orix,
                oriy,
                cpos[1] - orix,
                cpos[0] - oriy,
                head_length=15,
                head_width=10,
                length_includes_head=True,
                fc=col,
                ec=col,
            )
    except Exception as e:
        print("DEBUG")
        print(cell.get_mtoc())
        raise e


def visualize_canonized_alignment(ax, cell):
    oriy, orix = cell.nucleus_centroid
    maxis = cell.compute_migration_axis()
    rgbcomp = cell.composite_with_edge()


def make_img_montage(imglist, ncol=5, pad=4):

    isizes = tuple(zip(*[i.shape for i in imglist]))

    rmax = max(isizes[0])
    cmax = max(isizes[1])

    Ncols = min(ncol, len(imglist))
    Nrows = (len(imglist) // ncol) + (len(imglist) % ncol)

    montage_rowsize = Nrows * rmax + pad * (Nrows + 1)
    montage_colsize = Ncols * cmax + pad * (Ncols + 1)

    is_color_image = len(isizes) == 3

    if is_color_image:
        montage_size = (montage_rowsize, montage_colsize, 3)
    else:
        montage_size = (montage_rowsize, montage_colsize)

    montage = np.ones(montage_size, dtype=imglist[0].dtype)

    for i, im in enumerate(imglist):

        r_ = i // Ncols
        c_ = i % Ncols

        if rmax % 2 == 0:
            roffset = 0
        else:
            roffset = 1

        if cmax % 2 == 0:
            coffset = 0
        else:
            coffset = 1

        # image center in montage frame
        rc = (rmax + pad) * r_ + (rmax // 2 + roffset) + pad
        cc = (cmax + pad) * c_ + (cmax // 2 + coffset) + pad

        Ny, Nx = im.shape[0:2]

        if Ny % 2 == 0:
            yoffset = 0
        else:
            yoffset = 1

        if Nx % 2 == 0:
            xoffset = 0
        else:
            xoffset = 1

        ri = rc - (Ny // 2) - yoffset
        rf = rc + (Ny // 2)

        ci = cc - (Nx // 2) - xoffset
        cf = cc + (Nx // 2)

        if is_color_image:
            montage[ri:rf, ci:cf, :] = im
        else:
            montage[ri:rf, ci:cf] = im

    return montage
