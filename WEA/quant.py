# some functions for quantitation of circular data
import numpy as np


def mean_resultant_length(data, unit="degree"):
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

    if unit == "degree":
        thetas = np.deg2rad(data)
    elif unit == "radian":
        thetas = data

    cos_part, sine_part = np.cos(thetas), np.sin(thetas)
    csum = np.sum(cos_part)
    ssum = np.sum(sine_part)

    return np.arctan2(ssum, csum)