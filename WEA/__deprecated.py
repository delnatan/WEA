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
        cellmask, empty_area_threshold=empty_area_threshold / (metadata["dxy"] ** 2),
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
        imglist, diameter=nucdiam / dxy, resample=True, channels=[0, 0],
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
