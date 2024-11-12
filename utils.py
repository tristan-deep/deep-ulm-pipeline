"""Utilities

Author: Tristan Stevens
"""

import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def yellow(text):
    """Color text yellow."""
    return f"\033[33m{text}\033[0m"


def check_monotonic_increase(arr):
    """Check if array is monotonically increasing."""
    return np.all(np.diff(arr) >= 0)


def array_idx_to_list(array, indices):
    """Split input array into list according to indices.

    Args:
        array (ndarray): Any input array.
        indices (array): 1D indices array. Should have same size
            as first axis of input array.

    Returns:
        List: list of ndarrays split by indices.

    Examples:
        In this example array and indices are the same to
        clearly see the functionality of this function.
        >>> x = np.array([
            0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4]
        )
        >>> array_idx_to_list(x, x)
        [array([0, 0, 0]),
        array([1, 1, 1, 1, 1, 1]),
        array([2, 2]),
        array([3]),
        array([4, 4, 4, 4])]
    """
    assert check_monotonic_increase(
        indices
    ), "Indices need to be monotonically increasing"
    list_out = np.array_split(array, np.where(np.diff(indices))[0] + 1, axis=0)
    return list_out


def list_to_array_idx(lst):
    """
    Converts a list of arrays into a single numpy array with an additional column
    indicating the original index of each element in the list.

    Args:
        lst (list): A list of numpy arrays.

    Returns:
        numpy.ndarray: A numpy array with two columns. The first column contains all
        elements from the input list concatenated together, and the second column
        contains the index of the original array that each element came from.

    Example:
        >>> lst = [np.array([1, 2, 3]), np.array([4, 5])]
        >>> list_to_array_idx(lst)
        array([[1, 0],
               [2, 0],
               [3, 0],
               [4, 1],
               [5, 1]])
    """
    if len(lst) == 0:
        raise ValueError("Input list must not be empty")
    assert isinstance(lst, list), "Input must be a list"
    assert all(
        isinstance(a, np.ndarray) for a in lst
    ), "All elements of the list must be numpy arrays"
    assert all(
        l.shape[1:] == lst[0].shape[1:] for l in lst
    ), "All arrays must have the same shape except for the first dimension"
    lengths = [len(l) for l in lst]
    idx = np.repeat(np.arange(len(lengths)), lengths)
    # add appropriate number of singleton dims to idx if necessary
    idx = np.reshape(idx, idx.shape + (1,) * (lst[0].ndim - 1))
    lst = np.concatenate(lst)
    if lst.ndim == 1:
        idx = np.expand_dims(idx, axis=1)
        lst = np.expand_dims(lst, axis=1)
    return np.hstack([lst, idx])


def append_counters(counter):
    """Append multiple counters after eachother.

    Will output purely monotonically ascending list.


    Args:
        counter (ndarray): List with ascending counters.

    Returns:
        (ndarray): Monotonically ascending list.

    Examples:
        [1, 2, 3, 3, 4, 5, 5, 1, 2, 2]
        -> [1, 2, 3, 3, 4, 5, 5, 6, 7, 7]
    """
    counter = np.array(counter)
    max_idx = np.max(counter)
    diff = counter[:-1] - counter[1:]
    idx = np.where(diff > 0)[0] + 1
    for i in idx:
        counter[i:] += max_idx
    return counter


def translate(array, range_from, range_to):
    """Map values in array from one range to other.

    Args:
        array (ndarray): input array.
        range_from (Tuple): lower and upper bound of original array.
        range_to (Tuple): lower and upper bound to which array should be mapped.

    Returns:
        (ndarray): translated array
    """
    leftMin, leftMax = range_from
    rightMin, rightMax = range_to
    if leftMin == leftMax:
        return np.ones_like(array) * rightMax

    # Convert the left range into a 0-1 range (float)

    if not isinstance(array, np.ndarray):
        array = array.numpy()

    valueScaled = (array - leftMin) / (leftMax - leftMin)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * (rightMax - rightMin))


def set_random_seed(seed=None):
    """Set random seed to all random generators."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    return seed


def make_unique_path(save_dir):
    """Create unique directory from save_dir ussing incremental suffix."""
    save_dir = Path(save_dir)
    try:
        save_dir.mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        unique_dir_found = False
        post_fix = 0
        while not unique_dir_found:
            try:
                Path(str(save_dir) + f"_{post_fix}").mkdir(exist_ok=False, parents=True)
                unique_dir_found = True
                save_dir = Path(str(save_dir) + f"_{post_fix}")
            except FileExistsError:
                post_fix += 1
    return save_dir


def generate_fake_bmode_data(
    shape, normalization_range=(0, 1), num_blobs=10, std_dev_range=(1, 5)
):
    """Quickly generate some fake B-mode data for testing purposes."""
    # Create an empty array
    data = np.zeros(shape)

    for _ in range(num_blobs):
        # Randomly choose the center of the blob
        center = np.random.randint(0, shape[0], size=2)
        # Randomly choose the standard deviation of the blob
        std_dev = np.random.uniform(*std_dev_range)

        # Create a Gaussian blob
        for i in range(shape[0]):
            for j in range(shape[1]):
                data[i, j] += np.exp(
                    -((i - center[0]) ** 2 + (j - center[1]) ** 2) / (2 * std_dev**2)
                )

    # Normalize the data
    data_min, data_max = normalization_range
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data * (data_max - data_min) + data_min

    return data


def find_peaks_tensorflow(data, thres):
    """Find peaks in a 2D image."""
    tmp_data = data

    # peak detection using max pooling
    # pad ones instead of zeros to avoid boundary artifact
    in_height, in_width = data.shape[1:3]
    stride_height = stride_width = 1
    filter_height = filter_width = 3
    if in_height % stride_height == 0:
        pad_along_height = max(filter_height - stride_height, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_height), 0)
    if in_width % stride_width == 0:
        pad_along_width = max(filter_width - stride_width, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_width), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    tmp_data_padded = tf.pad(tmp_data, paddings, mode="CONSTANT", constant_values=1)
    maxpool_data = tf.nn.pool(
        tmp_data_padded,
        window_shape=(filter_height, filter_width),
        pooling_type="MAX",
        padding="VALID",
    )
    binary_map = tf.logical_and(
        tf.equal(tmp_data, maxpool_data), tf.greater(tmp_data, thres)
    )
    peaks = tf.where(binary_map)

    intensity = tf.gather_nd(data, peaks)
    peaks = peaks.numpy().astype("float64")

    peaks[:, 3] = peaks[:, 0]
    peaks[:, 0] = intensity

    sr_map = np.array(binary_map).astype("int64")
    sr_map = np.sum(sr_map, axis=0).squeeze()
    return peaks, sr_map
