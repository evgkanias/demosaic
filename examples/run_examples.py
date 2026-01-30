import demosaic as dm

import datetime as dt
import numpy as np
import cv2
import os

DATASET_PATH = os.path.abspath(os.path.join("..", "..", "compass-bee-dance", "data", "poughon2024"))


# Function to map angle and DoLP to HSV and then to BGR
def angle_dolp_to_rgb(angle_channel, dolp_channel):
    hue = np.round((np.nan_to_num(angle_channel, nan=0) % 180) / 180.0 * 179).astype(np.uint8)
    value = (np.clip(np.nan_to_num(dolp_channel, nan=0), 0, 1) * 255).astype(np.uint8)
    hsv = np.zeros((*hue.shape, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = value
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


if __name__ == '__main__':
    session = "2022-07-06"
    img_path = os.path.join(DATASET_PATH, f"{session}_raw.npy")
    ano_path = os.path.join(DATASET_PATH, f"{session}_raw_annotations.npy")
    print(img_path)

    data = np.load(img_path, allow_pickle=True)
    annotations = np.load(ano_path, allow_pickle=True)
    img_dict = {
        'img': [img for img in data[:, 0]],
        'exp': [exp for exp in data[:, 1]],
        'tim': [dt.datetime.strptime(tim, '%Y-%m-%dT%H-%M-%S') for tim in data[:, 2]],
        'typ': [typ for typ in data[:, 3]],  # numbers correspond to fixed or auto exposure values
        'lab': [cla for cla in annotations[:, 1]],
    }

    dm_pol = dm.polarisation.demosaic(img_dict['img'][2])
    dm_img = dm.bayer.demosaic(dm_pol[..., 0], colour=dm.colour.RG2RGB_135_000_090_045, method='fourier', blur_kernel=1)
    dm_sto = dm.polarisation.stokes(dm_pol, colour=dm.colour.RG2RGB_135_000_090_045)
    dm_deg = dm.polarisation.degree(dm_sto)
    dm_ang = dm.polarisation.angle(dm_sto)

    # cv2.imshow("Raw random image", dm_pol[..., 0])
    cv2.imshow("Bayer random image", (dm_img / 256).astype(np.uint8))
    # cv2.imshow("Stokes", (dm_sto[..., 2] * 256).astype(np.uint8))
    cv2.imshow("Polarisation", angle_dolp_to_rgb(np.degrees(dm_ang), dm_deg))
    cv2.waitKey(0)
