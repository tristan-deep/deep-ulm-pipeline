"""Saving localizations to hdf5 file
Author: Tristan Stevens

Requirements:
 - pip install h5py
"""

from pathlib import Path

import h5py
import numpy as np


def save_localizations(out_file_path, px, pz, fr_num, intensity, image_shape):
    """Save localizations to hdf5 file.
    Args:
        out_file_path (str): The path to the output hdf5 file.
        px (ndarray): The x-coordinates of the localizations.
        pz (ndarray): The z-coordinates of the localizations.
        fr_num (ndarray): The frame numbers of the localizations.
        intensity (ndarray): The intensity values of the localizations.
        image_shape (tuple): The shape of the image (z, x).

    Raises:
        AssertionError: If the input arrays have invalid shapes or values.

    Returns:
        None
    """

    assert px.shape == pz.shape == fr_num.shape == intensity.shape
    assert len(px.shape) == 1
    assert len(image_shape) == 2

    # make sure pz and px are in limits of image_shape
    assert np.all(pz < image_shape[0]), f"{pz.max()} >= {image_shape[0]}"
    assert np.all(px < image_shape[1]), f"{px.max()} >= {image_shape[1]}"
    assert np.all(pz >= 0), "pz must be all positive"
    assert np.all(px >= 0), "px must be all positive"

    # make sure fr_num is 0-indexed
    assert np.all(fr_num >= 0), "fr_num must be all positive"
    assert fr_num.min() == 0, "fr_num must be 0-indexed"

    localizations = np.stack([pz, px, fr_num, intensity], axis=1)

    index_keys = {
        "pz": 0,
        "px": 1,
        "fr_num": 2,
        "intensity": 3,
    }
    dtypes = {
        "px": "float32",
        "pz": "float32",
        "fr_num": "int32",
        "intensity": "float32",
    }
    out_file_path = Path(out_file_path)
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file_path, "w") as h5file:
        for key, index in index_keys.items():
            h5file.create_dataset(key, data=localizations[:, index], dtype=dtypes[key])
        h5file.create_dataset("image_shape", data=image_shape, dtype="int32")

    print(f"Succesfully saved tracks to {out_file_path}")


if __name__ == "__main__":
    out_file_path = ".temp/localizations.hdf5"

    # localizations are saved independently of upscaling factor, and normalized
    # to the original image size. This means that the localizations are
    # saved in the same coordinate system as the B-mode image. As localizations are
    # done on the upscaled image, the localizations here will be floats after normalization.

    px = ...  # make sure it is a 1D array within 0 and image_shape[1]
    pz = ...  # make sure it is a 1D array within 0 and image_shape[0]
    fr_num = ...  # make sure it is a 1D array and 0-indexed
    intensity = ...  # make sure it is a 1D array

    image_shape = (
        ...
    )  # make sure it is a tuple of length 2 with size of (z, x) dimensions

    # some dummy data as example and to test the saving script
    # please remove this when you have your own data
    ### >>>>> begin of dummy data <<<<< ###
    px = np.array([0.1, 2.1, 3, 4.5, 5.7, 6.2, 6.3])
    pz = np.array([0.1, 2.1, 3, 4.5, 5.7, 6.2, 6.3])
    fr_num = np.array([0, 0, 0, 1, 1, 2, 2])
    intensity = np.array([1, 1, 1, 1, 1, 1, 1])

    image_shape = (10, 10)
    ### >>>>> end of dummy data <<<<< ###

    save_localizations(out_file_path, px, pz, fr_num, intensity, image_shape)
