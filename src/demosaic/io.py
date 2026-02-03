"""
Input and output functions.
"""
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2026, Lund Vision Group, Lund University"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0"
__maintainer__ = "Evripidis Gkanias"

import demosaic.utils as utils

import numpy as np
import PIL.Image as Image
import piexif
import yaml
import os
import re
import loguru as lg


def get_raw_image_files(directory):
    """
    Returns the raw image files from a directory based on their name patterns. Images what are named as 'image#.tiff'
    or 'image_#.tiff' are supposed to be raw, where # is a number. It returns a list of names of the raw image files,
    sorted in ascending order.

    Parameters
    ----------
    directory: str
        the path of the directory to search for image files.

    Returns
    -------
    list[str]
    """
    return sorted([f for f in os.listdir(directory) if re.match(r'image\d+\.tiff$', f)])


def read_raw_images(directory):
    """
    Reads the raw image files from a directory based on their name patterns. Images what are named as 'image#.tiff' or
    'image_#.tiff' are supposed to be raw, where # is a number. It returns a tuple of the image data, metadata, and
    file names.

    Parameters
    ----------
    directory: str
        the path of the directory to search for image files.

    Returns
    -------
    tuple[list[np.ndarray[int]], list[dict], list[str]]
    """
    files = get_raw_image_files(directory)
    images = []
    metas = []
    for fname in files:
        path = os.path.join(directory, fname)

        img = load_image(path, normalise=True)

        if img is None:
            lg.logger.error(f"Image not found or not single channel: {path}")
            continue

        meta_dict = load_meta(path)
        images.append(img[:, ::-1])
        metas.append(meta_dict)
    return images, metas, files


def load_image(path, normalise=True):
    """
    Loads an image from a file. If normalise is set to True, the image is normalised to [0, 1].

    Parameters
    ----------
    path: str
        the path of the image to load.
    normalise: bool
        whether to normalise the image to [0, 1]. Default is True.

    Returns
    -------
    np.ndarray[float, int]
    """
    img = Image.open(path)

    if normalise:
        assert img.mode in ['I;16', 'L'], f'Image mode is {img.mode}, expected single channel image.'

        # check image mode
        if img.mode == 'I;16':
            dtype = np.float64
            max_val = utils.LDR_MAX
        else:  # img.mode == 'L':
            dtype = np.float32
            max_val = utils.UINT8_MAX

        img = np.array(img, dtype=dtype) / max_val

    return img


def load_meta(path):
    """
    Loads the EXIF metadata from an image file.

    Parameters
    ----------
    path: str
        the path of the image to load its metadata.

    Returns
    -------
    dict[str, Any]
    """
    return get_meta(piexif.load(path))


def save_image(path, img, meta=None):
    """
    Saves an image and its metadata to a file. This function supports unit16 values only for single channel images. RGB
    images are transformed into uint8.

    Parameters
    ----------
    path: str
        the path of the image to save.
    img: np.ndarray[float, int]
        the pixel values of the image.
    meta: dict[str, Any]
        the metadata of the image. Default is None.
    """
    lg.logger.debug(f'Saving image...')
    # transform uint16 to uint8
    # Pillow does not support uint16 RGB images
    # OpenCV does not support EXIF metadata
    # Tifffile does not support EXIF but supports custom metadata
    if img.dtype == np.dtypes.UInt16DType() and img.ndim > 2:
        lg.logger.warning(f"Pillow does not support 16-bit RGB images. Reducing resolution to 8 bits for image {path}")
        img_bytes = np.uint8(img / 256)
        mode = 'RGB'
    else:
        if img.ndim > 2:
            mode = 'RGB'
        elif img.dtype == np.dtypes.UInt16DType():
            mode = 'I;16'  # non-RGB images can be 16-bit long
        elif img.dtype == np.dtypes.UInt8DType():
            mode = 'L'
        else:
            mode = None
        img_bytes = img
    pil_img = Image.fromarray(img_bytes[:, ::-1], mode=mode)
    exif = None
    if meta is not None:
        exif = get_exif_bytes(meta)
    pil_img.save(path, exif=exif, compression=None)
    lg.logger.debug(f'Saved image at {path}')


def get_exif_bytes(meta):
    """
    Transforms the dictionary of metadata into an array of EXIF bytes to accompany an image file.

    Parameters
    ----------
    meta: dict[str, Any]
        the metadata of the image.

    Returns
    -------
    bytes
    """
    exif = {"0th": {}, "Exif": {}, "GPS": {}}

    # Camera info
    exif["0th"][piexif.ImageIFD.Make] = meta["CameraMaker"]
    exif["0th"][piexif.ImageIFD.Model] = meta['CameraModel']
    exif["0th"][piexif.ImageIFD.UniqueCameraModel] = f"{meta['CameraMaker']} {meta['CameraModel']}"
    exif["0th"][piexif.ImageIFD.DateTime] = meta['DateTime']
    exif["0th"][piexif.ImageIFD.Software] = meta.get("Software", "HDR-LUCID-CAMERA")
    exif["0th"][piexif.ImageIFD.Artist] = meta.get("Artist", "Evripidis Gkanias")
    exif["0th"][piexif.ImageIFD.Copyright] = meta.get("Copyright", "2025, Lund Vision Group")

    # Exposure and image info
    exif["Exif"][piexif.ExifIFD.BodySerialNumber] = meta['CameraSerialNumber']
    exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = meta['DateTime']
    if "ExposureTime" in meta:
        exif["Exif"][piexif.ExifIFD.ExposureTime] = rational(meta['ExposureTime'], 1000000)
    exif["Exif"][piexif.ExifIFD.BrightnessValue] = rational(meta['CameraBrightness'], 10)
    exif["Exif"][piexif.ExifIFD.ISOSpeedRatings] = int(meta['ISOSpeedRatings'])
    exif["Exif"][piexif.ExifIFD.FNumber] = rational(meta['FNumber'], 10)
    exif["Exif"][piexif.ExifIFD.FocalLength] = rational(meta['FocalLength'], 10)
    exif["Exif"][piexif.ExifIFD.LensMake] = meta['LensMaker']
    exif["Exif"][piexif.ExifIFD.LensModel] = meta['LensModel']
    exif["Exif"][piexif.ExifIFD.Temperature] = rational(meta['CameraTemperature'], 10)

    if 'Gamma' in meta:
        exif["Exif"][piexif.ExifIFD.Gamma] = rational(meta['Gamma'], 10)

    # GPS info
    deg_lat, min_lat, sec_lat, sig_lat = dms(meta['Latitude'])
    deg_lon, min_lon, sec_lon, sig_lon = dms(meta['Longitude'])

    exif["GPS"] = {
        piexif.GPSIFD.GPSLatitudeRef: 'N' if sig_lat >= 0 else 'S',
        piexif.GPSIFD.GPSLatitude: (rational(deg_lat, 1), rational(min_lat, 1), rational(sec_lat, 100)),
        piexif.GPSIFD.GPSLongitudeRef: 'E' if sig_lon >= 0 else 'W',
        piexif.GPSIFD.GPSLongitude: (rational(deg_lon, 1), rational(min_lon, 1), rational(sec_lon, 100)),
        piexif.GPSIFD.GPSAltitudeRef: 0 if meta['Altitude'] >= 0 else 1,
        piexif.GPSIFD.GPSAltitude: rational(abs(meta['Altitude']), 1)
    }

    exif_bytes = piexif.dump(exif)
    return exif_bytes


def get_meta(exif):
    """
    Transforms EXIF data into a metadata dictionary.

    Parameters
    ----------
    exif: dict
        the EXIF data.

    Returns
    -------
    dict[str, Any]
    """
    # Camera info
    meta = {
        "CameraMaker": exif["0th"][piexif.ImageIFD.Make],
        "CameraModel": exif["0th"][piexif.ImageIFD.Model],
        "DateTime": exif["Exif"][piexif.ExifIFD.DateTimeOriginal],
        "Software": exif["0th"][piexif.ImageIFD.Software],
        "Artists": exif["0th"][piexif.ImageIFD.Artist],
        "Copyright": exif["0th"][piexif.ImageIFD.Copyright],

        # Exposure and image info
        "CameraSerialNumber": exif["Exif"][piexif.ExifIFD.BodySerialNumber],
        "ExposureTime": unrational(*exif["Exif"][piexif.ExifIFD.ExposureTime]),
        "CameraBrightness": unrational(*exif["Exif"][piexif.ExifIFD.BrightnessValue]),
        "ISOSpeedRatings": exif["Exif"][piexif.ExifIFD.ISOSpeedRatings],
        "FNumber": unrational(*exif["Exif"][piexif.ExifIFD.FNumber]),
        "FocalLength": unrational(*exif["Exif"][piexif.ExifIFD.FocalLength]),
        "LensMaker": exif["Exif"][piexif.ExifIFD.LensMake],
        "LensModel": exif["Exif"][piexif.ExifIFD.LensModel],
        "CameraTemperature": unrational(*exif["Exif"][piexif.ExifIFD.Temperature]),

        # GPS info
        "Latitude": undms(*([unrational(*val) for val in exif["GPS"][piexif.GPSIFD.GPSLatitude]] +
                            [1.0 if exif["GPS"][piexif.GPSIFD.GPSLatitudeRef] == b'N' else -1])),
        "Longitude": undms(*([unrational(*val) for val in exif["GPS"][piexif.GPSIFD.GPSLongitude]] +
                             [1.0 if exif["GPS"][piexif.GPSIFD.GPSLongitudeRef] == b'E' else -1])),
        "Altitude": unrational(*exif["GPS"][piexif.GPSIFD.GPSAltitude]) * (
            1 if exif["GPS"][piexif.GPSIFD.GPSAltitudeRef] >= 0 else -1)
    }

    if piexif.ExifIFD.Gamma in exif["Exif"]:
        meta['Gamma'] = unrational(*exif["Exif"][piexif.ExifIFD.Gamma])

    return meta


def save_metadata(dir_path, meta):
    """
    Saves the metadata in a file called 'info.txt' in the given directory. The format of the metadata matches the one of
    a YAML file.

    Parameters
    ----------
    dir_path: str
        the directory to save the metadata in.
    meta: dict[str, Any]
        the metadata dictionary.
    """
    path = os.path.join(dir_path, 'info.txt')
    with open(path, 'w') as f:
        yaml.safe_dump(meta, f)
    lg.logger.debug(f'Saved metadata at: {path}.')


def dms(decimal):
    """
    Converts a decimal number that indicates an angle (degrees) of an Earth's coordinate to a tuple of
    degrees, minutes, seconds and sign. T

    Parameters
    ----------
    decimal: float
        the decimal number of the angle in degrees.

    Returns
    -------
    tuple[int, int, int, int]
    """
    sign = np.sign(decimal)
    decimal_degrees = abs(decimal)

    degrees = int(decimal_degrees)
    decimal_part = decimal_degrees - degrees

    minutes = int(decimal_part * 60)
    decimal_minutes = (decimal_part * 60) - minutes

    seconds = round(decimal_minutes * 60)

    return degrees, minutes, seconds, sign


def undms(degrees, minutes, seconds, sign):
    """
    Convert DMS (degrees, minutes, seconds, sign) back to decimal degrees. The result angle (in degrees) is rounded to
    4 decimal digits.

    Parameters
    ----------
    degrees: int
    minutes: int
    seconds: int
    sign: int

    Returns
    -------
    float
    """
    decimal = float(np.round(abs(degrees) + minutes / 60.0 + seconds / 3600.0, decimals=4))
    return sign * decimal


def rational(value, precision=100000):
    """
    Converts a real number (value) into a rational tuple based on the precision. The precision sets how many digits of
    the value to keep. The value is multiplied with its precision and all the remaining digits are discarded. The result
    is two integers (rational tuple): the outcome of the multiplication (numerator) and the precision (denominator).

    Parameters
    ----------
    value: float
        the number to convert.
    precision: int
        the precision of the number.

    Returns
    -------
    tuple[int, int]
    """
    numerator = int(np.round(value * precision))
    denominator = precision
    return numerator, denominator


def unrational(*rational_tuple):
    """
    Convert a rational tuple (numerator, denominator) back to float.

    Parameters
    ----------
    rational_tuple: tuple[int, int]

    Returns
    -------
    float
    """
    numerator, denominator = rational_tuple
    return float(numerator) / float(denominator)

