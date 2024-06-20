"""Convert tracks to density and velocity maps.

Author: Tristan Stevens
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.interpolate import interp1d

from utils import array_idx_to_list, list_to_array_idx


def sub2ind(array_shape, rows, cols):
    """Convert subscripts to linear indices.
    Args:
        array_shape: shape of array
        rows: row subscripts
        cols: column subscripts
    """
    return rows * array_shape[1] + cols


def smooth(array, smooth_window_size):
    """Smooth the data using a window with requested size.

    Args:
        array (ndarray): 1-D array containing the data to be smoothed
        smooth_window_size (int): window size of smoothening kernel
            must be odd number.

    Returns:
        ndarray: smoothened array
    """
    assert smooth_window_size % 2 == 1, "smooth_window_size must be odd"
    assert len(array.shape) == 1, "smooth only accepts 1 dimension arrays."

    out0 = (
        np.convolve(array, np.ones(smooth_window_size, dtype=int), "valid")
        / smooth_window_size
    )
    r = np.arange(1, smooth_window_size - 1, 2)
    start = np.cumsum(array[: smooth_window_size - 1])[::2] / r
    stop = (np.cumsum(array[:-smooth_window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def interpolate(x, interpolateFactor: float = 0.1):
    """1D linear interpolation."""
    n = int(1 / interpolateFactor - 1)
    x_i = np.zeros(len(x) * (n + 1) - n)
    f = interp1d(np.arange(len(x)), x)
    x_i = f(np.linspace(0, len(x) - 1, len(x_i)))
    return x_i


def postprocess_tracks(
    raw_tracks: Union[list, np.ndarray],
    interpolation: bool = True,
    smooth_factor: int = None,
    min_length: int = 1,
    max_linking_distance: int = 20,
    upsampling_factor: int = 1,
    interp_exp: float = 0.8,
    index_keys: dict = None,
):
    """Post process tracks.

    1. Smoothen tracks
    2. Interpolate

    Args:
        raw_tracks (list or ndarray): input tracks. either list of tracks for
            each separate bubble or ndarray with shape:
            [pos_z, pos_x, vel_z, vel_x, frame_number, bubble_id]
        interpolation (bool): Perform interpolation
        smooth_factor (int): Amount of smoothening to apply
        min_length (int): minimum length of tracks to keep
        max_linking_distance (int): maximum distance between two points to link them
        upsampling_factor (int): amount of upsampling
        interp_exp (float): interpolation exponent
        index_keys (dict): dictionary containing the indices of the columns in the tracks.
            If None, the default is used:
                index_keys = {
                    "pz": 0,
                    "px": 1,
                    "vz": 2,
                    "vx": 3,
                    "fr_num": 4,
                    "track_id": 5,
                    "py": 6,
                }
    Returns:
        list or ndarray: post processed tracks. same format as input, but with
            many more points due to interpolation.
    """
    assert isinstance(
        raw_tracks, (list, np.ndarray)
    ), f"raw_tracks must be list or ndarray, not {type(raw_tracks)}"

    assert smooth_factor is None or smooth_factor % 2 == 1, "smooth_factor must be odd"

    interp_factor = 1 / max_linking_distance / upsampling_factor * interp_exp
    if index_keys is None:
        index_keys = {
            "pz": 0,
            "px": 1,
            "vz": 2,
            "vx": 3,
            "fr_num": 4,
            "track_id": 5,
            "py": 6,
        }

    smooth_keys = ["pz", "px", "vz", "vx"]
    interpolate_keys = ["pz", "px", "vz", "vx", "fr_num"]

    converted_to_list = False
    if isinstance(raw_tracks, np.ndarray):
        # sort raw_tracks on track_id, but also on frame number as we are going to interpolate
        # between points in the same track and we want to keep the order of the points
        sorted_indices = np.lexsort(
            (raw_tracks[:, index_keys["fr_num"]], raw_tracks[:, index_keys["track_id"]])
        )
        raw_tracks = raw_tracks[sorted_indices]
        raw_tracks = array_idx_to_list(
            raw_tracks, raw_tracks[:, index_keys["track_id"]]
        )
        converted_to_list = True

    tracks_out = []
    for track in tqdm.tqdm(raw_tracks, desc="Post processing tracks..."):
        N = len(track)
        track_ids = track[:, index_keys["track_id"]]
        frame_numbers = track[:, index_keys["fr_num"]]
        unique_track_ids = set(track_ids)
        assert len(unique_track_ids) == 1, "track_id must be unique for each track"
        track_id = unique_track_ids.pop()
        if N > min_length - 1:
            # Smooth
            if smooth_factor:
                # must be smaller than N and uneven
                a = smooth_factor if smooth_factor <= N else N - (1 - N % 2)
                tempList = []
                for index_key in smooth_keys:
                    index = index_keys[index_key]
                    tempList.append(smooth(track[:, index], int(a)))
                tempList.append(frame_numbers)
                tempList.append(track_ids)
                track = np.stack(tempList, axis=-1)

            # Interpolate
            if interpolation:
                tempList = []
                for index_key in interpolate_keys:
                    index = index_keys[index_key]
                    tempList.append(interpolate(track[:, index], interp_factor))
                new_track_length = len(tempList[0])
                track_ids = np.ones(new_track_length) * track_id
                tempList.append(track_ids)
                track = np.stack(tempList, axis=-1)

            # add py as zeros for now
            track = np.concatenate([track, np.zeros((len(track), 1))], axis=-1)

            tracks_out.append(track)

    # new order of columns is [pz, px, vz, vx, fr_num, track_id, py]
    # convert back to input order
    current_order = ["pz", "px", "vz", "vx", "fr_num", "track_id", "py"]
    tracks_out = [
        track[:, [current_order.index(key) for key in index_keys.keys()]]
        for track in tracks_out
    ]

    if converted_to_list:
        tracks_out = list_to_array_idx(tracks_out)
        # remove last idx dim as it is already present as a column (track_id)
        tracks_out = tracks_out[:, :-1]

    return tracks_out


def tracks_to_map(
    tracks: Union[list, np.ndarray],
    size: tuple,
    smooth_window_size: int = 9,
    plot_maps: bool = False,
    index_keys: dict = None,
):
    """Convert tracks to density and velocity maps.
    Args:
        tracks (list or np.ndarray): list of tracks with arrays with shape (N, 6) with N being
            the number of points in the track or ndarray with shape (N, 6) with N being the
            of total points in all tracks combined. The columns are expected to be:
                (z, x, vel_z, vel_x, frame, id)

        size (tuple): size of the output map
        smooth_window_size (int): window size of smoothening kernel
            must be odd number.
        plot_maps (bool): plot maps using matplotlib
        index_keys (dict): dictionary containing the indices of the columns in the tracks.
            If None, the default is used:
                index_keys = {
                    "pz": 0,
                    "px": 1,
                    "vz": 2,
                    "vx": 3,
                    "fr_num": 4,
                    "track_id": 5,
                    "py": 6,
                }
    Returns:
        dict: dictionary containing the maps. has keys:
            - density_map
            - vel_z_map
            - vel_x_map
            - vel_norm_map
            - vel_mean_map
    """
    assert isinstance(
        tracks, (list, np.ndarray)
    ), f"raw_tracks must be list or ndarray, not {type(tracks)}"
    if index_keys is None:
        index_keys = {
            "pz": 0,
            "px": 1,
            "vz": 2,
            "vx": 3,
            "fr_num": 4,
            "track_id": 5,
            "py": 6,
        }

    if isinstance(tracks, np.ndarray):
        # sort raw_tracks on track_id
        sorted_indices = np.lexsort(
            (tracks[:, index_keys["fr_num"]], tracks[:, index_keys["track_id"]])
        )
        tracks = tracks[sorted_indices]
        tracks = array_idx_to_list(tracks, tracks[:, index_keys["track_id"]])

    delta = 0

    maps_out = {
        "density_map": np.zeros(size).flatten(),
        "vel_z_map": np.zeros(size).flatten(),
        "vel_x_map": np.zeros(size).flatten(),
        "vel_norm_map": np.zeros(size).flatten(),
        "vel_mean_map": np.zeros(size).flatten(),
    }

    for track in tqdm.tqdm(tracks, desc="Converting tracks to maps..."):
        # round pixel position into [z,x] index_keys
        pos_z_round = np.round(track[:, index_keys["pz"]] + delta).astype(int)
        pos_x_round = np.round(track[:, index_keys["px"]] + delta).astype(int)

        vel_z = track[:, index_keys["vz"]]
        vel_x = track[:, index_keys["vx"]]

        N = len(track)
        # must be smaller than N and uneven
        a = smooth_window_size if smooth_window_size <= N else N - (1 - N % 2)
        vel_norm = smooth(
            np.linalg.norm(track[:, [index_keys["vz"], index_keys["vx"]]], axis=1), a
        )

        vel_mean = vel_norm * np.sign(np.mean(track[:, index_keys["vz"]]))

        # remove out of grid bubbles (ie. the grid is too small)
        idx = np.where(
            (pos_z_round >= 0)
            & (pos_z_round < size[0])
            & (pos_x_round >= 0)
            & (pos_x_round < size[1])
        )
        ind = np.ravel_multi_index(
            np.array([pos_x_round[idx], pos_z_round[idx]]),
            (size[1], size[0]),
            order="F",
        )

        # Tracks are counted only once per pixel, the unique keeps only 1 point per pixel
        ind, idxs = np.unique(ind, return_index=True)  # .astype(int)

        maps_out["density_map"][ind] += 1
        maps_out["vel_z_map"][ind] += vel_z[idx[0][idxs]]
        maps_out["vel_x_map"][ind] += vel_x[idx[0][idxs]]
        maps_out["vel_norm_map"][ind] += vel_norm[idx[0][idxs]]
        maps_out["vel_mean_map"][ind] += vel_mean[idx[0][idxs]]

    idx = np.where(maps_out["density_map"] > 0)
    for key, value in maps_out.items():
        if key == "density_map":
            continue
        maps_out[key][idx] = value[idx] / maps_out["density_map"][idx]

    for key, value in maps_out.items():
        maps_out[key] = value.reshape(size)

    # plot all maps
    if plot_maps:
        fig, axs = plt.subplots(1, len(maps_out.keys()), figsize=(15, 5))
        for i, (key, value) in enumerate(maps_out.items()):
            axs[i].imshow(value)
            axs[i].set_title(key)
            axs[i].axis("off")
    return maps_out
