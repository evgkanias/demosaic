import demosaic.colour as dc
import demosaic.utils as du

import numpy as np


def demosaic(img, method=None, blur_kernel=3):
    if method is None:
        method = none
    elif isinstance(method, str):
        method = METHODS[method.lower()]

    new_img = method(img)

    if blur_kernel > 0:
        new_img = dc.blur(new_img, blur_kernel)

    return new_img


def none(img):
    # the actual pattern of filters does not matter here
    h, w = img.shape[:2]

    new_img = np.zeros((h // 2, w // 2, *img.shape[2:], 4), dtype=img.dtype)
    for i in range(4):
        new_img[..., i] = img[i % 2::2, i // 2::2]
    return new_img


def stokes(img, colour=dc.MONO_000_135_045_090):
    pattern = get_pattern(colour)

    print(pattern, img.shape)

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

    print(i1.max(), i2.max())

    new_img = np.zeros((*img.shape[:-1], 3), dtype=float)
    new_img[..., 0] = (i1 + i2) / 4.0  # I or S0

    new_img[..., 1] = (f["000"] - f["090"]) / i1  # Q or S1
    new_img[..., 2] = (f["045"] - f["135"]) / i2  # U or S2

    return new_img


def angle(img, colour=None):
    if colour is not None:
        # if colour is not given, the input image is the precomputed stokes parameters
        img = stokes(img, colour)
    return (0.5 * np.arctan2(img[..., 1], img[..., 2])) % np.pi


def degree(img, colour=None):
    if colour is not None:
        # if colour is not given, the input image is the precomputed stokes parameters
        img = stokes(img, colour)
    return np.clip(np.sqrt(np.square(img[..., 1]) + np.square(img[..., 2])), 0, 1)


def get_pattern(colour):
    for var in dir(dc):
        if eval(f"dc.{var}") == colour:
            return "_".join(var.split("_")[1:])
    else:
        return None


METHODS = {
    "none": none
}
