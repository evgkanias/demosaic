"""
High dynamic range functionality.
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
import cv2

GAMMA_CORRECTION = du.config['convert']['gamma']
COLOUR = dc.RG2RGB_000_045_135_090

TONEMAP = cv2.createTonemap(gamma=GAMMA_CORRECTION)


def merge(*img, exposures=None, method=None, gamma_correction=None, ldr=True):
    """
    Merge multiple images of different exposures in a single high dynamic range (HDR) image.

    Parameters
    ----------
    img: list[np.ndarray[float, int]]
        the images to be merged.
    exposures: list[float] | None
        the exposures times. The number of exposures should equal the number of images in img. By default, exposures
        are assumed to be missing.
    method: callable | str
        The HDR method to be used: 'debeyec' (default), 'mertens', or 'robertson'.
    gamma_correction: float | None
        The gamma correction factor to be applied to the HDR image.
    ldr: bool
        whether to return the raw HDR output or to transform it into a lower dynamic range image. Default is True.

    Returns
    -------
    np.ndarray[float, int]
    """
    if method is None:
        method = METHODS["debevec"]
    elif isinstance(method, str) and method in METHODS:
        method = METHODS[method]
    else:
        raise ValueError('Method must be one of {}'.format(METHODS))

    if exposures is not None:
        exposures = exposures.copy()
    if gamma_correction is None:
        tonemap = TONEMAP
    else:
        tonemap = cv2.createTonemap(gamma=gamma_correction)
    hdr = method.process([np.uint8(i * 255) for i in img], times=exposures)

    # LDR and gamma correction
    hdr = tonemap.process(hdr)
    if ldr:
        return ldr16(hdr)
    else:
        return hdr


def ldr16(img):
    """
    Transforms an HDR image (float; values in range [0, 1]) into and LDR image (uint16; values in range [0, 65520]).

    Parameters
    ----------
    img: np.ndarray[float]
        the HDR image to be transformed.

    Returns
    -------
    np.ndarray[int]
    """
    return np.clip(np.nan_to_num(img, nan=0) * du.LDR_MAX, 0, du.LDR_MAX).astype(np.uint16)


METHODS = {
    "debevec": cv2.createMergeDebevec(),
    "mertens": cv2.createMergeMertens(),
    "robertson": cv2.createMergeRobertson()
}
