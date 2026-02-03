"""
The Bayer demosaicing functionality.
"""
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2026, Lund Vision Group, Lund University"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0"
__maintainer__ = "Evripidis Gkanias"

import demosaic.colour as dc

import scipy.signal as ss
import numpy as np
import cv2


def demosaic(img, method=None, colour=dc.RG2RGB, blur_kernel=3):
    """
    Transforms a mosaic image with Bayer filters on it into a coloured image as described by the colour value. The
    colour indicates both the Bayer pattern and the result colour encoding. The supported demosaicing methods are:
    'none', 'bilinear', 'malvar' (default), 'fourier', 'cv' (running the innate Open CV method), 'none_pol',
    'bilinear_pol', 'malvar_pol', and 'fourier_pol'. The variants with 'pol' assume that the polarisation filters have
    not been demosaiced beforehand. Custom methods are also supported, and the minimum input should be the mosaic image
    and colour code.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    method: callable, str
        the method to use; this could be the name of the method (e.g., 'none', 'bilinear', 'malvar', 'fourier', 'cv')
        or a custom function.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    blur_kernel: int
        the kernel size of the blur kernel.

    Returns
    -------
    np.ndarray[int, float] | None
    """
    if method is None:
        method = malvar
    elif isinstance(method, str):
        method = METHODS[method.lower()]

    new_img = method(img, colour=colour)

    if blur_kernel > 0:
        new_img = dc.blur(new_img, blur_kernel)

    return new_img


def none(img, colour=dc.RG2RGB, include_polarisation=False):
    """
    This method is doing the minimum interpolation by copying the colour values of neighbouring pixels to infer
    the missing colours in each pixel.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    np.ndarray[int, float] | None
    """
    red, green_0, green_1, blue = split_channels(img, colour, include_polarisation)

    g0 = (0.5 * green_0 + 0.5 * green_1).astype(np.uint16)
    rgb0 = np.transpose([red, g0, blue], axes=(1, 2, 0))

    r20 = red[1::]
    g21 = green_0[1::]
    g1 = 0.5 * g21 + 0.5 * green_1[:-1, :]
    rgb1 = np.transpose([r20, g1, blue[1:]], axes=(1, 2, 0))

    r02 = red[:, 1::]
    g12 = green_1[:, 1::]
    g2 = 0.5 * green_0[:, :-1] + 0.5 * g12
    rgb2 = np.transpose([r02, g2, blue[:, 1:]], axes=(1, 2, 0))

    r22 = red[1::, 1::]
    g3 = 0.5 * g12[:-1, :] + 0.5 * g21[:, :-1]
    rgb3 = np.transpose([r22, g3, blue[1:, 1:]], axes=(1, 2, 0))

    rgb = np.empty((
        rgb0.shape[0] + rgb1.shape[0],
        rgb0.shape[1] + rgb2.shape[1],
        rgb0.shape[2]
    ), dtype=img.dtype)
    rgb[0::2, 0::2] = rgb0
    rgb[1::2, 0::2] = rgb1
    rgb[0::2, 1::2] = rgb2
    rgb[1::2, 1::2] = rgb3

    return to_colour(rgb[..., 0], rgb[..., 1], rgb[..., 2], colour=colour)


def bilinear(img, colour=dc.RG2RGB, include_polarisation=False):
    """
    This method is taking the average value of neighbouring pixels to infer  the missing colours in each pixel.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    np.ndarray[int, float] | None
    """
    red, green_0, green_1, blue = split_channels(img, colour, include_polarisation)

    # read the first 4x4 block
    r00 = red[:-1, :-1]
    g01 = green_0[:-1, :-1]
    r02 = red[:-1, 1:]

    g10 = green_1[:-1, :-1]
    b11 = blue[:-1, :-1]
    g12 = green_1[:-1, 1:]
    b13 = blue[0:-1, 1:]

    r20 = red[1:, :-1]
    g21 = green_0[1:, :-1]
    r22 = red[1:, 1:]
    g23 = green_0[1:, 1:]

    b31 = blue[1:, 0:-1]
    g32 = green_1[1:, 1:]
    b33 = blue[1:, 1:]

    # calculate the missing channels
    g11 = (0.25 * g01 + 0.25 * g10 + 0.25 * g12 + 0.25 * g21).astype(img.dtype)
    r11 = (0.25 * r00 + 0.25 * r02 + 0.25 * r20 + 0.25 * r22).astype(img.dtype)

    b12 = (0.5 * b11 + 0.5 * b13).astype(img.dtype)
    r12 = (0.5 * r02 + 0.5 * r22).astype(img.dtype)

    b21 = (0.5 * b11 + 0.5 * b31).astype(img.dtype)
    r21 = (0.5 * r20 + 0.5 * r22).astype(img.dtype)

    b22 = (0.25 * b11 + 0.25 * b13 + 0.25 * b31 + 0.25 * b33).astype(img.dtype)
    g22 = (0.25 * g12 + 0.25 * g21 + 0.25 * g23 + 0.25 * g32).astype(img.dtype)

    rgb = np.zeros((img.shape[0] - 2, img.shape[1] - 2, 3), dtype=img.dtype)
    rgb[0::2, 0::2, 0] = r11
    rgb[0::2, 0::2, 1] = g11
    rgb[0::2, 0::2, 2] = b11
    rgb[0::2, 1::2, 0] = r12
    rgb[0::2, 1::2, 1] = g12
    rgb[0::2, 1::2, 2] = b12
    rgb[1::2, 0::2, 0] = r21
    rgb[1::2, 0::2, 1] = g21
    rgb[1::2, 0::2, 2] = b21
    rgb[1::2, 1::2, 0] = r22
    rgb[1::2, 1::2, 1] = g22
    rgb[1::2, 1::2, 2] = b22

    return to_colour(rgb[..., 0], rgb[..., 1], rgb[..., 2], colour=colour)


def malvar(img, colour=dc.RG2RGB, include_polarisation=False):
    """
    This method is using a weighted average of neighbouring pixels and its own value to infer the missing colours in
    each pixel. It uses information from all the colours to calculate these values.

    Notes
    -----
    Malvar, H. S., He, L.-W. & Cutler, R. High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color
    Images. 2004 IEEE Int. Conf. Acoust., Speech, Signal Process. 3, III-485-III–488 (2004).

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    np.ndarray[int, float] | None
    """
    red, green_0, green_1, blue = split_channels(np.astype(img, np.float64), colour, include_polarisation)

    # img = np.astype(img, np.float64)
    # r02 = img[0:-5:2, 2:-3:2]
    # g03 = img[0:-5:2, 3:-2:2]
    r02 = red[:-2, 1:-1]
    g03 = green_0[:-2, 1:-1]

    # b11 = img[1:-4:2, 1:-4:2]
    # g12 = img[1:-4:2, 2:-3:2]
    # b13 = img[1:-4:2, 3:-2:2]
    # g14 = img[1:-4:2, 4:-1:2]
    b11 = blue[:-2, :-2]
    g12 = green_1[:-2, 1:-1]
    b13 = blue[:-2, 1:-1]
    g14 = green_1[:-2, 2:]

    # r20 = img[2:-3:2, 0:-5:2]
    # g21 = img[2:-3:2, 1:-4:2]
    # r22 = img[2:-3:2, 2:-3:2]
    # g23 = img[2:-3:2, 3:-2:2]
    # r24 = img[2:-3:2, 4:-1:2]
    # g25 = img[2:-3:2, 5::2]
    r20 = red[1:-1, :-2]
    g21 = green_0[1:-1, :-2]
    r22 = red[1:-1, 1:-1]
    g23 = green_0[1:-1, 1:-1]
    r24 = red[1:-1, 2:]
    g25 = green_0[1:-1, 2:]

    # g30 = img[3:-2:2, 0:-5:2]
    # b31 = img[3:-2:2, 1:-4:2]
    # g32 = img[3:-2:2, 2:-3:2]
    # b33 = img[3:-2:2, 3:-2:2]
    # g34 = img[3:-2:2, 4:-1:2]
    # b35 = img[3:-2:2, 5::2]
    g30 = green_1[1:-1, :-2]
    b31 = blue[1:-1, :-2]
    g32 = green_1[1:-1, 1:-1]
    b33 = blue[1:-1, 1:-1]
    g34 = green_1[1:-1, 2:]
    b35 = blue[1:-1, 2:]

    # g41 = img[4:-1:2, 1:-4:2]
    # r42 = img[4:-1:2, 2:-3:2]
    # g43 = img[4:-1:2, 3:-2:2]
    # r44 = img[4:-1:2, 4:-1:2]
    g41 = green_0[2:, :-2]
    r42 = red[2:, 1:-1]
    g43 = green_0[2:, 1:-1]
    r44 = red[2:, 2:]

    # g52 = img[5::2, 2:-3:2]
    # b53 = img[5::2, 3:-2:2]
    g52 = green_1[2:, 1:-1]
    b53 = blue[2:, 1:-1]

    R00 = 8 * r22
    G00 = 4 * r22 + 2 * (g12 + g21 + g23 + g32) - (r02 + r20 + r24 + r42)
    B00 = 6 * r22 + 2 * (b11 + b13 + b31 + b33) - 1.5 * (r20 + r24 + r02 + r42)

    R01 = 5 * g32 + 4 * (r22 + r42) - (g12 + g52 + g21 + g41 + g23 + g43) + 0.5 * (g30 + g34)
    G01 = 8 * g23
    B01 = 5 * g32 + 4 * (b31 + b33) - (g30 + g34 + g21 + g41 + g23 + g43) + 0.5 * (g12 + g52)

    R10 = 5 * g23 + 4 * (r22 + r24) - (g21 + g25 + g12 + g14 + g32 + g34) + 0.5 * (g03 + g43)
    G10 = 8 * g32
    B10 = 5 * g23 + 4 * (b13 + b33) - (g03 + g43 + g12 + g41 + g32 + g34) + 0.5 * (g21 + g25)

    R11 = 6 * b33 + 2 * (r22 + r24 + r42 + r44) - 1.5 * (b31 + b35 + b13 + b53)
    G11 = 4 * b33 + 2 * (g23 + g32 + g34 + g43) - (b13 + b31 + b35 + b53)
    B11 = 8 * b33

    rgb00 = np.transpose([R00, G00, B00], axes=(1, 2, 0)) / 8
    rgb01 = np.transpose([R01, G01, B01], axes=(1, 2, 0)) / 8
    rgb10 = np.transpose([R10, G10, B10], axes=(1, 2, 0)) / 8
    rgb11 = np.transpose([R11, G11, B11], axes=(1, 2, 0)) / 8

    rgb = np.empty((rgb00.shape[0] + rgb01.shape[0], rgb10.shape[1] + rgb11.shape[1], rgb00.shape[2]),
                   dtype=img.dtype)
    rgb[0::2, 0::2] = np.round(rgb00)
    rgb[1::2, 0::2] = np.round(rgb01)
    rgb[0::2, 1::2] = np.round(rgb10)
    rgb[1::2, 1::2] = np.round(rgb11)

    return to_colour(rgb[..., 0], rgb[..., 1], rgb[..., 2], colour=colour)


def fourier(img, colour=dc.RG2RGB, include_polarisation=False):
    """
    This method is doing a Fourier transform of the image to analyse the colours. This implementation is not complete
    for all the colour patterns.

    Notes
    -----
    Hagen, N., Stockmans, T., Otani, Y. & Buranasiri, P. Fourier-domain filtering analysis for color-polarization camera
    demosaicking. Appl. Opt. 63, 2314 (2024).

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    NotImplementedError
    """
    n_x, n_y = img.shape[:2]

    x = np.arange(n_x)[:, None]
    y = np.arange(n_y)[None, :]

    # mu_x = get_mu_x(x, colour, include_polarisation)
    # mu_y = get_mu_x(y, colour, include_polarisation)
    #
    # mu_r = get_red_mask(img, colour=colour, include_polarisation=include_polarisation) > 0
    # mu_g = get_green_mask(img, colour=colour, include_polarisation=include_polarisation) > 0
    # mu_b = get_blue_mask(img, colour=colour, include_polarisation=include_polarisation) > 0
    #
    # mu_w = 1.0
    # mu_u = mu_x * mu_y
    # mu_v = 0.5 * (mu_y - mu_y)

    p_x, p_y = n_x // 2, n_y // 2
    m_x, m_y = p_x // 2, p_y // 2  # mask size

    f2 = np.fft.fftshift(np.fft.fft2(img))

    # import matplotlib.pyplot as plt
    #
    # f2abs = np.where(abs(f2) > 0, np.log(abs(f2)), 0)
    # plt.figure('log(abs(fft_img))')
    # plt.imshow(f2abs, extent=[-1, 1, -1, 1], aspect='auto')
    # plt.xticks([-1, -0.5, 0, 0.5, 1], ['-1', '-1/2', '0', '1/2', '1'])
    # plt.yticks([-1, -0.5, 0, 0.5, 1], ['-1', '-1/2', '0', '1/2', '1'])
    # # draw_rgbpol_fft_circles()
    # plt.colorbar()
    # plt.xlabel('x-axis frequencies (Nyquist units)')
    # plt.ylabel('y-axis frequencies (Nyquist units)')
    # plt.show()

    mask = __create_mask_function(n_x, n_y, n_x // 4, n_y // 4)

    c00 = np.fft.ifft2(np.fft.ifftshift(f2 * mask))
    c10 = np.fft.ifft2(np.fft.ifftshift(np.roll(f2, -m_x, axis=0) * mask))
    cm10 = np.fft.ifft2(np.fft.ifftshift(np.roll(f2, m_x, axis=0) * mask))
    c01 = np.fft.ifft2(np.fft.ifftshift(np.roll(f2, -m_y, axis=1) * mask))
    c0m1 = np.fft.ifft2(np.fft.ifftshift(np.roll(f2, m_y, axis=1) * mask))
    c02 = np.fft.ifft2(np.fft.ifftshift(np.roll(f2, -p_y, axis=1) * mask))
    c20 = np.fft.ifft2(np.fft.ifftshift(np.roll(f2, p_x, axis=0) * mask))

    c11 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, -m_x, axis=0), -m_y, axis=1) * mask))
    cm1m1 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, m_x, axis=0), m_y, axis=1) * mask))
    c1m1 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, -m_x, axis=0), m_y, axis=1) * mask))
    cm11 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, m_x, axis=0), -m_y, axis=1) * mask))

    c12 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, -m_x, axis=0), -p_y, axis=1) * mask))
    cm12 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, m_x, axis=0), -p_y, axis=1) * mask))
    c21 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, -p_x, axis=0), -m_y, axis=1) * mask))
    c2m1 = np.fft.ifft2(np.fft.ifftshift(np.roll(np.roll(f2, -p_x, axis=0), m_y, axis=1) * mask))

    rs0 = np.real(2*c00 - (1+1j)*c10 - c1m1 - (1-1j)*cm10 - cm11 + 1j*cm1m1 - 1j*c11 + (1-1j)*c0m1 + (1+1j)*c01)
    rs1 = 0.5 * np.real(
        2 * c20 - 2 * c02 + 2j * c1m1 - 2j * cm11 - (1 - 1j) * c10 + (1 + 1j) * c21 + (1 - 1j) * c2m1 - (1 + 1j) * cm10
        - (1 - 1j) * c01 - (1 + 1j) * c0m1 + (1 + 1j) * c12 + (1 - 1j) * cm12)
    rs2 = 0.5 * np.real(
        -2 * c20 - 2 * c02 + 2 * c11 + 2 * cm1m1 + (1 - 1j) * c10 - (1 + 1j) * c21 - (1 - 1j) * c2m1 + (1 + 1j) * cm10
        - (1 - 1j) * c01 - (1 + 1j) * c0m1 + (1 + 1j) * c12 + (1 - 1j) * cm12)

    gs0 = np.real((2 * c00) + 1j * c11 + c1m1 + cm11 - 1j * cm1m1)
    # gsp = real((2 * c20) + c11 - (1j * c1m1) + (1j * cm11) + cm1m1)
    # gsm = real((2 * c02) + c11 + (1j * c1m1) - (1j * cm11) + cm1m1)
    gs1 = np.real(c20 - c02 - 1j * c1m1 + 1j * cm11)
    gs2 = np.real(-c20 - c02 - c11 - cm1m1)

    bs0 = np.real(2 * c00 - 1j * c11 - c1m1 - cm11 + 1j * cm1m1 - (1 + 1j) * c01 - (1 - 1j) * c0m1 + (1 + 1j) * c10 + (
                1 - 1j) * cm10)
    # Bsp = real(2*c20 - c11 + 1j*c1m1 - 1j*cm11 - cm1m1 + (1-1j)*c10 - (1+1j)*c21 - (1-1j)*c2m1 + (1+1j)*cm10)
    # Bsm = real(2*c02 - c11 - 1j*c1m1 + 1j*cm11 - cm1m1 - (1-1j)*c01 - (1+1j)*c0m1 + (1+1j)*c12 + (1-1j)*cm12)
    bs1 = 0.5 * np.real(
        2 * c20 - 2 * c02 + 2j * c1m1 - 2j * cm11 + (1 - 1j) * c10 - (1 + 1j) * c21 - (1 - 1j) * c2m1 + (1 + 1j) * cm10
        + (1 - 1j) * c01 + (1 + 1j) * c0m1 - (1 + 1j) * c12 - (1 - 1j) * cm12)
    bs2 = 0.5 * np.real(
        -2 * c20 - 2 * c02 + 2 * c11 + 2 * cm1m1 - (1 - 1j) * c10 + (1 + 1j) * c21 + (1 - 1j) * c2m1 - (1 + 1j) * cm10
        + (1 - 1j) * c01 + (1 + 1j) * c0m1 - (1 + 1j) * c12 - (1 - 1j) * cm12)

    ## Prevent any divide-by-small number problems.
    rns1, rns2 = rs1 / (rs0 + np.finfo(float).eps), rs2 / (rs0 + np.finfo(float).eps)
    gns1, gns2 = gs1 / (gs0 + np.finfo(float).eps), gs2 / (gs0 + np.finfo(float).eps)
    bns1, bns2 = bs1 / (bs0 + np.finfo(float).eps), bs2 / (bs0 + np.finfo(float).eps)

    (n_x, n_y) = rs0.shape
    rgb_s0 = np.zeros((n_x, n_y, 3), 'float32')
    rgb_s0[:, :, 0] = rs0
    rgb_s0[:, :, 1] = gs0
    rgb_s0[:, :, 2] = bs0

    rgb_ns1 = np.zeros((n_x, n_y, 3), 'float32')
    rgb_ns1[:, :, 0] = rns1
    rgb_ns1[:, :, 1] = gns1
    rgb_ns1[:, :, 2] = bns1

    rgb_ns2 = np.zeros((n_x, n_y, 3), 'float32')
    rgb_ns2[:, :, 0] = rns2
    rgb_ns2[:, :, 1] = gns2
    rgb_ns2[:, :, 2] = bns2

    raise NotImplementedError("The Fourier method is not yet implemented.")


def __create_mask_function(n_x, n_y, m_x, m_y, mask_type='rect'):
    p_x, p_y = n_x // 2, n_y // 2
    mask = np.zeros((n_x, n_y), dtype=float)

    if mask_type == 'rect':
        mask[p_x - (m_x // 2):p_x + (m_x // 2), p_y - (m_y // 2):p_y + (m_y // 2)] = 1.0
        return mask
    elif mask_type == 'supergauss':
        w_x = ss.general_gaussian(m_x, p=3.0, sig=m_x / 2.9)
        w_y = ss.general_gaussian(m_y, p=3.0, sig=m_y / 2.9)
    else:
        w_x = eval(f'np.{mask_type}')(m_x)
        w_y = eval(f'np.{mask_type}')(m_y)

    window_2d = np.outer(w_x, w_y)
    mask[p_x - (m_x // 2):p_x + (m_x // 2), p_y - (m_y // 2)] = window_2d

    return mask


def get_red_mask(shape, colour=dc.RG2RGB, include_polarisation=True):
    """
    Calculates the mask that includes only the red filters in the mosaic based on the Bayer colour pattern and on
    whether polarisation was already demosaiced.

    Parameters
    ----------
    shape: list[int] | tuple[int]
        the image shape.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    np.ndarray[bool]
    """
    x = np.arange(shape[0])[:, None]
    y = np.arange(shape[1])[None, :]

    mu_x = __get_mu_x(x, colour, include_polarisation)
    mu_y = __get_mu_x(y, colour, include_polarisation)

    if dc.from_rggb(colour):
        mu = .25 * (1 + mu_x) * (1 + mu_y)
    elif dc.from_bggr(colour):
        mu = .25 * (1 - mu_x) * (1 - mu_y)
    elif dc.from_grbg(colour):
        mu = .25 * (1 - mu_x) * (1 + mu_y)
    else:
        mu = .25 * (1 + mu_x) * (1 - mu_y)

    return mu > 0.5


def get_blue_mask(shape, colour=dc.RG2RGB, include_polarisation=True):
    """
    Calculates the mask that includes only the blue filters in the mosaic based on the Bayer colour pattern and on
    whether polarisation was already demosaiced.

    Parameters
    ----------
    shape: list[int] | tuple[int]
        the image shape.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    np.ndarray[bool]
    """
    x = np.arange(shape[0])[:, None]
    y = np.arange(shape[1])[None, :]

    mu_x = __get_mu_x(x, colour, include_polarisation)
    mu_y = __get_mu_x(y, colour, include_polarisation)

    if dc.from_rggb(colour):
        mu = .25 * (1 - mu_x) * (1 - mu_y)
    elif dc.from_bggr(colour):
        mu = .25 * (1 + mu_x) * (1 + mu_y)
    elif dc.from_grbg(colour):
        mu = .25 * (1 + mu_x) * (1 - mu_y)
    else:
        mu = .25 * (1 - mu_x) * (1 + mu_y)

    return np.round(mu)


def get_green_mask(shape, colour=dc.RG2RGB, include_polarisation=True):
    """
    Calculates the mask that includes only the green filters in the mosaic based on the Bayer colour pattern and on
    whether polarisation was already demosaiced.

    Parameters
    ----------
    shape: list[int] | tuple[int]
        the image shape.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    np.ndarray[bool]
    """
    x = np.arange(shape[0])[:, None]
    y = np.arange(shape[1])[None, :]

    mu_x = __get_mu_x(x, colour, include_polarisation)
    mu_y = __get_mu_x(y, colour, include_polarisation)

    if dc.from_rggb(colour) or dc.from_bggr(colour):
        mu = .5 * (1 - mu_x * mu_y)
    else:
        mu = .5 * (1 + mu_x * mu_y)

    return np.round(mu)


def __get_mu_x(x, colour, include_polarisation=True):
    if dc.from_polarised(colour) and include_polarisation:
        return np.sqrt(2) * np.cos(np.pi / 4 * (2 * x - 1))
    else:
        return np.cos(np.pi * x)


def split_channels(img, colour=dc.RG2RGB, include_polarisation=True):
    """
    Based on the colour code and polarisation information, it splits the image mosaic into the separate channels:
    red, green_0, green_1, and blue.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    colour: int
        the colour pattern of the mosaic from the colour directory.
    include_polarisation: bool
        whether to take into account that polarisation filters are also included. This makes each colout filter to
        occupy 2x2 pixels instead of a single pixel and is handled differently. If the polarisation was demosaiced
        beforehand, this should be False.

    Returns
    -------
    tuple[np.ndarray[int, float], np.ndarray[int, float], np.ndarray[int, float], np.ndarray[int, float]]
    """
    red_mask = get_red_mask(img.shape, colour=colour, include_polarisation=include_polarisation)
    green_mask = get_green_mask(img.shape, colour=colour, include_polarisation=include_polarisation)
    blue_mask = get_blue_mask(img.shape, colour=colour, include_polarisation=include_polarisation)

    red = np.reshape(img[red_mask], (img.shape[0] // 2, -1))
    green = np.reshape(img[green_mask], (img.shape[0], -1))
    green_0 = green[0::2]
    green_1 = green[1::2]
    blue = np.reshape(img[blue_mask], (img.shape[0] // 2, -1))

    return red, green_0, green_1, blue


def to_colour(red, green, blue, colour=dc.RG2RGB):
    """
    Combines the red, green and blue channels into a single colour image based on the colour code.

    Parameters
    ----------
    red: np.ndarray[int, float]
        the red channel.
    green: np.ndarray[int, float]
        the green channel.
    blue: np.ndarray[int, float]
        the blue channel.
    colour: int
        the colour pattern of the mosaic from the colour directory.

    Returns
    -------
    np.ndarray[int, float]
    """

    if dc.to_bgr(colour):
        img = np.transpose([blue, green, red], axes=(1, 2, 0))
    elif dc.to_rgb(colour):
        img = np.transpose([red, green, blue], axes=(1, 2, 0))
    else:
        img = ((red + green + blue) / 3.0).astype(red.dtype)

    return img


def cv(img, colour=dc.RG2RGB):
    """
    Uses the OpenCV library to convert the Bayer-mosaic image into a colour image.

    Parameters
    ----------
    img: np.ndarray[int, float]
        the 2D array of a mosaic image.
    colour: int
        the colour pattern of the mosaic from the colour directory.

    Returns
    -------
    np.ndarray[int, float] | None
    """
    return cv2.cvtColor(img, colour)


METHODS = {
    "none": none,
    "none_pol": lambda *args, **kwargs: none(*args, **kwargs, include_polarisation=True),
    "bilinear": bilinear,
    "bilinear_pol": lambda *args, **kwargs: bilinear(*args, **kwargs, include_polarisation=True),
    "malvar": malvar,
    "malvar_pol": lambda *args, **kwargs: malvar(*args, **kwargs, include_polarisation=True),
    "fourier": fourier,
    "fourier_pol": lambda *args, **kwargs: fourier(*args, **kwargs, include_polarisation=True),
    "cv": cv
}
