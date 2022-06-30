import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter


def det_norm_hess(img, r=1.8):
    """computes determinant of Hessian / 'curvature' response for spot finding"""
    # s, is the gaussian sigma
    s = np.sqrt(r**2 / 2)

    # t, is scale (or variance)
    t = s * s

    Lxx = gaussian_filter(inputimg, sigma=s, order=(0, 2))
    Lyy = gaussian_filter(inputimg, sigma=s, order=(2, 0))
    Lxy = gaussian_filter(inputimg, sigma=s, order=(1, 1))

    detHimg = t * t * (Lxx * Lyy - Lxy * Lxy)

    return detHimg
