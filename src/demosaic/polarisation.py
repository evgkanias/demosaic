"""
Polarisation demosaicing functionality.
"""
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2026, Lund Vision Group, Lund University"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0"
__maintainer__ = "Evripidis Gkanias"

import demosaic.colour as dc
import demosaic.utils as du

import numpy as np


def demosaic(img, method=None, blur_kernel=3):
    """
    Transforms a mosaic image with polarisation filters on it into an image with 4 channels: one per polarisation
    filter. The only supported demosaicing method is 'none' (default).

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    method: callable, str
        the method to use; this could be the name of the method (e.g., 'none') or a custom function.
    blur_kernel: int
        the kernel size of the blur kernel.

    Returns
    -------
    np.ndarray[int, float] | None
    """
    if method is None:
        method = none
    elif isinstance(method, str):
        method = METHODS[method.lower()]

    new_img = method(img)

    if blur_kernel > 0:
        new_img = dc.blur(new_img, blur_kernel)

    return new_img


def none(img):
    """
    This method is doing no interpolation and just splits the image in 4 channels by following the filter pattern.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.

    Returns
    -------
    np.ndarray[int, float] | None
    """
    # the actual pattern of filters does not matter here
    h, w = img.shape[:2]

    new_img = np.zeros((h // 2, w // 2, *img.shape[2:], 4), dtype=img.dtype)
    for i in range(4):
        new_img[..., i] = img[i % 2::2, i // 2::2]
    return new_img


def stokes(img, colour=dc.MONO_000_135_045_090):
    """
    Calculates the Stoke's parameters of a demosaiced image using the pattern indicated by the colour value. Returns an
    image with 3 channels that represent the first 3 of the Stoke's parameters: S0, S1, and S2.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the demosaiced image.
    colour: int
        the colour and polarisation patterns of the mosaic from the colour directory.

    Returns
    -------
    np.ndarray[float]
    """
    pattern = get_pattern(colour)

    if img.dtype == np.uint16:
        int_max = du.UINT16_MAX
    elif img.dtype == np.uint8:
        int_max = du.UINT8_MAX
    else:
        int_max = 1.0

    f = {}
    for i, ang in enumerate(pattern.split("_")):
        f[ang] = np.float32(img[..., i]) / int_max

    i1 = f["000"] + f["090"] + np.finfo(float).eps
    i2 = f["045"] + f["135"] + np.finfo(float).eps

    new_img = np.zeros((*img.shape[:-1], 3), dtype=float)
    new_img[..., 0] = (i1 + i2) / 4.0  # I or S0

    new_img[..., 1] = (f["000"] - f["090"]) / i1  # Q or S1
    new_img[..., 2] = (f["045"] - f["135"]) / i2  # U or S2

    return new_img


def angle(img, colour=None):
    """
    Calculates the angle of linear polarisation (in rads). If the colour is None, it assumes that the input image is the Stoke's
    parameters. If the colour is not none, it assumes that the image is the raw demosaiced image and automatically
    calculates the Stoke's parameters before processing.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the demosaiced image or the Stoke's parameters.
    colour: int | None
        the colour pattern of the mosaic from the colour directory. If None, it assumes that the input image is the raw
        demosaiced image and automatically calculates the Stoke's parameters before processing.

    Returns
    -------
    np.ndarray[float]
    """
    if colour is not None:
        # if colour is not given, the input image is the precomputed stokes parameters
        img = stokes(img, colour)
    return (0.5 * np.arctan2(img[..., 1], img[..., 2])) % np.pi


def degree(img, colour=None):
    """
    Calculates the degree of linear polarisation. If the colour is None, it assumes that the input image is the Stoke's
    parameters. If the colour is not none, it assumes that the image is the raw demosaiced image and automatically
    calculates the Stoke's parameters before processing.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the demosaiced image or the Stoke's parameters.
    colour: int | None
        the colour pattern of the mosaic from the colour directory. If None, it assumes that the input image is the raw
        demosaiced image and automatically calculates the Stoke's parameters before processing.

    Returns
    -------
    np.ndarray[float]
    """
    if colour is not None:
        # if colour is not given, the input image is the precomputed stokes parameters
        img = stokes(img, colour)
    return np.clip(np.sqrt(np.square(img[..., 1]) + np.square(img[..., 2])), 0, 1)


def get_pattern(colour):
    """
    Extracts the filters (angles of polarisers) pattern from a colour code. It returns a single string with the angles
    seperated by underscores ('_').

    Parameters
    ----------
    colour: int
        the colour pattern of the mosaic from the colour directory.

    Returns
    -------
    str
    """
    for var in dir(dc):
        if eval(f"dc.{var}") == colour:
            return "_".join(var.split("_")[1:])
    else:
        return None


METHODS = {
    "none": none
}
