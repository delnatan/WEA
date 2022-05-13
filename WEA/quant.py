# some functions for quantitation of circular data
import numpy as np


def mean_resultant_length(data, unit="degree"):
    """ computes mean resultant length from circular data 
    
    The mean resultant length is a measure of dispersion of circular data.
    Circular data is first decomposed into its cartesian components (x, y) via
    sine and cosines, then each components are summed up before taking its 
    L2-norm. This amounts to summing up the circular data as one would to to
    vectors with unit length (e.g. head-to-tail vector addition)
    
    """
    if unit == "degree":
        thetas = np.deg2rad(data)
    elif unit == "radian":
        thetas = data

    cos_part, sine_part = np.cos(thetas), np.sin(thetas)
    csum = np.sum(cos_part)
    ssum = np.sum(sine_part)
    mrl = np.sqrt(csum * csum + ssum * ssum)
    return mrl / data.size


def mean_direction(data, unit="degree"):
    """ computes mean direction from circular data 

    the mean direction is taken as the arctan of the final vector that results
    from adding separate cartesian components of the circular data. arctan2()
    is used to cover all quadrants (ranging from -pi, +pi)

    """
    if unit == "degree":
        thetas = np.deg2rad(data)
    elif unit == "radian":
        thetas = data

    cos_part, sine_part = np.cos(thetas), np.sin(thetas)
    csum = np.sum(cos_part)
    ssum = np.sum(sine_part)

    return np.arctan2(ssum, csum)