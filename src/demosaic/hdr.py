import demosaic.io as dio
import demosaic.colour as dc

import numpy as np
import cv2

GAMMA_CORRECTION = dio.config['convert']['gamma']
UINT16_MAX = 65520.0
LDR_MAX = 65535.0
COLOUR = dc.RG2RGB_000_045_135_090

TONEMAP = cv2.createTonemap(gamma=GAMMA_CORRECTION)


def merge(*img, exposures=None, method=None, gamma_correction=None, raw=False):
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
    if raw:
        return hdr
    else:
        return ldr16(hdr)


def ldr16(img):
    return np.clip(np.nan_to_num(img, nan=0) * LDR_MAX, 0, LDR_MAX).astype(np.uint16)


METHODS = {
    "debevec": cv2.createMergeDebevec(),
    "mertens": cv2.createMergeMertens(),
    "robertson": cv2.createMergeRobertson()
}
