import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter


def det_norm_hess(img, r=1.8):
    """computes determinant of Hessian / 'curvature' response for spot finding"""
    # s, is the gaussian sigma
    s = np.sqrt(r**2 / 2)

    # t, is scale (or variance)
    t = s * s

    Lxx = gaussian_filter(img, sigma=s, order=(0, 2))
    Lyy = gaussian_filter(img, sigma=s, order=(2, 0))
    Lxy = gaussian_filter(img, sigma=s, order=(1, 1))

    detHimg = t * t * (Lxx * Lyy - Lxy * Lxy)

    return detHimg


def save_as_cellpose_input(layers, outdir, fnout, mode="nuclei"):
    outpath = Path(outdir)
    dapi = [layer.data for layer in layers if layer.name.startswith("dapi")][0]
    tubulin = [layer.data for layer in layers if layer.name.startswith("tubulin")][0]
    cyto = [layer.data for layer in layers if layer.name.startswith("cyto")][0]

    Ny, Nx = dapi.shape

    rgb = np.zeros((Ny, Nx, 3), dtype=np.uint16)

    if mode == "nuclei":
        rgb[:, :, 0] = tubulin
        rgb[:, :, 1] = dapi
    else:
        rgb[:, :, 0] = tubulin
        rgb[:, :, 1] = cyto
        rgb[:, :, 2] = dapi

    tifffile.imwrite(outpath / fnout, rgb)
