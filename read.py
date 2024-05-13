"""Read classes for localization and track files (hdf5)

Author: Tristan Stevens
"""

import copy
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import scipy.io as sio
import yaml

from tracks_to_map import postprocess_tracks, tracks_to_map
from utils import array_idx_to_list, list_to_array_idx, yellow


class ReadH5:
    def __init__(self, file_path):
        """Open a .h5 file for reading.

        Args:
            file_path :  path to the .h5 HDF5 file

        """
        self.file_path = Path(file_path)
        self.h5f = h5py.File(self.file_path, "r")

    def get_extension(self):
        return self.file_path.suffix

    def __getitem__(self, i, keys=None):
        if keys is None:
            return self._get(i=i, group=self.h5f)
        else:
            return self._get_from_keys(i=i, keys=keys, group=self.h5f)

    def _get_from_keys(self, i, keys, group=None):
        alist = list()
        for key in keys:
            alist.append(group[key][i])
        return alist

    def get_all(self):
        return self._get(i=None, group=self.h5f)

    def _get(self, i=None, group=None):
        alist = []
        for key in group.keys():
            sub_group = group.get(key)
            if isinstance(sub_group, h5py.Group):
                output = self._get(i, sub_group)
            elif isinstance(sub_group, h5py.Dataset):
                if i is None:
                    # get all using ':'
                    output = sub_group[:]
                else:
                    output = sub_group[i]
            else:
                raise ValueError(f"{type(group)}")
            alist.append(output)
        return alist

    def keys(self):
        return self.h5f.keys()

    def summary(self):
        self.h5f.visititems(print)
        # self.h5f.visititems(self._visit_func)

    @staticmethod
    def _visit_func(name, node):
        print(f"{node.name}: ")

    def frame_as_first(self, frames):
        """permute the dataset to have the frame indices as the first dimension

        args:
            numpy array with frame indices as last dimension

        Returns:
            numpy array of shape num_frames x ....
        """

        # always start with frame dim:
        last_dim = len(np.shape(frames)) - 1
        order = (last_dim,) + tuple(np.arange(0, last_dim))
        frames = np.array(frames).transpose(order)
        return frames

    def __len__(self):
        return len(self.h5f[list(self.keys())[0]])

    def close(self):
        """Close the .h5 HDF5 file for reading.

        Returns:
            void
        """

        self.h5f.close()


def save_dict_to_file(filename, dic):
    """Save dict to .mat or .h5"""

    filetype = Path(filename).suffix
    assert filetype in [".mat", ".h5"]

    if filetype == ".h5":
        with h5py.File(filename, "w") as h5file:
            recursively_save_dict_contents_to_group(h5file, "/", dic)
    elif filetype == ".mat":
        sio.savemat(filename, dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """Save dict contents to group"""
    for key, item in dic.items():
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            h5file[path + key] = item


def load_dict_from_file(filename, squeeze=True):
    """dict from file"""
    filetype = Path(filename).suffix
    assert filetype in [".mat", ".h5"]

    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file, "/", squeeze)


def recursively_load_dict_contents_from_group(h5file, path, squeeze=True):
    """Load dict from contents of group"""
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if squeeze:
                ans[key] = np.squeeze(item[()])
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans


def set_dict_as_attr(obj, dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            set_dict_as_attr(obj, value)
        setattr(obj, key, value)
    return


class ReadLocs(ReadH5):
    """Read class for reading / saving and handling of localizations."""

    def __init__(
        self,
        file_path=None,
        locs=None,
        init_index_order=None,
        image_shape=None,
    ):
        if file_path:
            super().__init__(file_path)
            assert (
                locs is None and init_index_order is None and image_shape is None
            ), "Either file_path or localizations (with init_index_order and image_shape) must be provided, not both"
            self.init_index_order = ["intensity", "pz", "px", "fr_num"]
            self.locs = None
            keys = list(self.keys())
            keys.remove("image_shape")
            self._image_shape = self.h5f["image_shape"][:]
        else:
            assert (
                locs is not None
            ), "Either file_path or localizations must be provided"
            assert (
                init_index_order is not None
            ), "init_index_order must be provided alongside localizations"
            assert (
                image_shape is not None
            ), "image_shape must be provided alongside localizations"
            self.locs = locs
            self.init_index_order = init_index_order
            self._image_shape = image_shape
            keys = self.init_index_order

        assert (
            len(self._image_shape) == 2
        ), f"image_shape must be (z, x) -> (height, width), got {self._image_shape}"

        # dictionary with keys and indices
        self._index_keys = dict(zip(keys, range(len(keys))))

        if self.locs is not None:
            self._assert_locs_image_shape()

        self.dtypes = {
            "intensity": "float32",
            "px": "float32",
            "pz": "float32",
            "fr_num": "int32",
        }

        assert len(self.index_keys) == len(self.dtypes) == len(self.init_index_order), (
            "index_keys, dtypes and init_index_order must have same length, got lengths "
            f"{len(self.index_keys)}, {len(self.dtypes)}, {len(self.init_index_order)}"
        )

    def save(self, file_path):
        """Save localizations to file.
        Args:
            file_path: Path to save file.
        """
        with h5py.File(file_path, "w") as h5file:
            for key, index in self.index_keys.items():
                h5file.create_dataset(
                    key, data=self.locs[:, index], dtype=self.dtypes[key]
                )
            h5file.create_dataset("image_shape", data=self.image_shape, dtype="int32")
        print(f"Saved localizations to {yellow(file_path)}")

    def read(self):
        """Read localizations from file.

        Returns:
            localizations: ndarray of shape (N, 4) with columns
                [intensity, pz, px, fr_num]
        """
        n_columns = len(self.index_keys)
        n_locs = len(self.h5f["intensity"])
        self.locs = np.zeros((n_locs, n_columns), dtype=np.float32)
        for key, index in self.index_keys.items():
            self.locs[:, index] = self.h5f[key][:]

        # sort based on frame numbers
        idx = np.argsort(self.locs[:, self.index_keys["fr_num"]])
        self.locs = self.locs[idx]

        self._assert_locs_image_shape()
        return self.locs

    def __len__(self):
        """Number of localizations"""
        if self.locs is None:
            return None
        return len(self.locs)

    @property
    def index_keys(self):
        """Index keys."""
        return self._index_keys

    @index_keys.setter
    def index_keys(self, value):
        """Set index_keys."""
        if isinstance(value, dict):
            self._index_keys = value
        elif isinstance(value, list):
            self._index_keys = dict(zip(value, range(len(value))))
        else:
            raise ValueError("index_keys must be dict or list")

    @property
    def num_frames(self):
        """Number of frames."""
        if self.locs is None:
            return None
        return len(np.unique(self.locs[:, self.index_keys["fr_num"]]))

    @property
    def image_shape(self):
        """Image shape, (z, x) -> (height, width)"""
        return self._image_shape

    def _assert_locs_image_shape(self):
        """Assert that localizations are within image shape."""
        if self.locs is None:
            return
        locs = self.locs
        if locs[:, self.index_keys["px"]].max() > self.image_shape[1]:
            raise ValueError(
                f"Localizations are outside of image shape: {self.image_shape}"
            )
        if locs[:, self.index_keys["pz"]].max() > self.image_shape[0]:
            raise ValueError(
                f"Localizations are outside of image shape: {self.image_shape}"
            )


class ReadTracks(ReadH5):
    """Read class for reading / saving and handling of tracks."""

    def __init__(
        self,
        file_path=None,
        tracks=None,
        init_index_order=None,
        image_shape=None,
    ):
        if file_path:
            super().__init__(file_path)
            assert (
                tracks is None and init_index_order is None and image_shape is None
            ), "Either file_path or tracks (with init_index_order and image_shape) must be provided, not both"
            self.init_index_order = ["pz", "px", "vz", "vx", "fr_num", "track_id", "py"]
            self.tracks = None
            keys = list(self.keys())
            keys.remove("image_shape")
            self._image_shape = self.h5f["image_shape"][:]
        else:
            assert tracks is not None, "Either file_path or tracks must be provided"
            assert (
                init_index_order is not None
            ), "init_index_order must be provided alongside tracks"
            assert (
                image_shape is not None
            ), "image_shape must be provided alongside tracks"
            self.tracks = tracks
            self.init_index_order = init_index_order
            self._image_shape = image_shape
            keys = self.init_index_order

        assert (
            len(self._image_shape) == 2
        ), f"image_shape must be (z, x) -> (height, width), got {self._image_shape}"

        # dictionary with keys and indices
        self._index_keys = dict(zip(keys, range(len(keys))))

        if self.tracks is not None:
            self._assert_tracks_image_shape()

        self.tracks_post = None

        self.dtypes = {
            "px": "float32",
            "pz": "float32",
            "py": "float32",
            "vx": "float32",
            "vz": "float32",
            "fr_num": "int32",
            "track_id": "int32",
        }

        assert len(self.index_keys) == len(self.dtypes) == len(self.init_index_order), (
            "index_keys, dtypes and init_index_order must have same length, got lengths "
            f"{len(self.index_keys)}, {len(self.dtypes)}, {len(self.init_index_order)}"
        )

    def save(self, file_path):
        """Save tracks to file.
        Args:
            file_path: Path to save file.
        """
        if isinstance(self.tracks, list):
            tracks = self.list_to_array(self.tracks)
        else:
            tracks = self.tracks
        with h5py.File(file_path, "w") as h5file:
            for key, index in self.index_keys.items():
                h5file.create_dataset(
                    key, data=tracks[:, index], dtype=self.dtypes[key]
                )
            h5file.create_dataset("image_shape", data=self.image_shape, dtype="int32")
        print(f"Saved tracks to {yellow(file_path)}")

    def read(self):
        """Read tracks from file.

        Returns:
            tracks: ndarray of shape (N, 7) with columns
                [pz, px, vz, vx, fr_num, track_id, py]
        """
        n_columns = len(self.index_keys)
        n_tracks = len(self.h5f["track_id"])
        self.tracks = np.zeros((n_tracks, n_columns), dtype=np.float32)
        for key, index in self.index_keys.items():
            self.tracks[:, index] = self.h5f[key][:]

        self.transpose(*self.init_index_order)

        # sort on fr_num and track_id using lexisort
        sort_idx = np.lexsort(
            (
                self.tracks[:, self.index_keys["track_id"]],
                self.tracks[:, self.index_keys["fr_num"]],
            )
        )
        self.tracks = self.tracks[sort_idx]

        self._assert_tracks_image_shape()
        return self.tracks

    def array_to_list(self, tracks_array):
        """Convert tracks in array format to list format."""
        # sort on track_id
        sort_idx = np.argsort(tracks_array[:, self.index_keys["track_id"]])
        tracks_array = tracks_array[sort_idx]
        tracks_list = array_idx_to_list(
            tracks_array, tracks_array[:, self.index_keys["track_id"]]
        )
        return tracks_list

    def list_to_array(self, tracks_list):
        """Convert tracks in list format to array format."""
        tracks_array = list_to_array_idx(tracks_list)
        return tracks_array

    def __len__(self):
        """Number of localizations"""
        if self.tracks is None:
            return None
        return len(self.tracks)

    @property
    def index_keys(self):
        """Index keys."""
        return self._index_keys

    @index_keys.setter
    def index_keys(self, value):
        """Set index_keys."""
        if isinstance(value, dict):
            self._index_keys = value
        elif isinstance(value, list):
            self._index_keys = dict(zip(value, range(len(value))))
        else:
            raise ValueError("index_keys must be dict or list")

    @property
    def n_tracks(self):
        """Number of tracks."""
        if self.tracks is None:
            return None
        return len(np.unique(self.tracks[:, self.index_keys["track_id"]]))

    @property
    def n_frames(self):
        """Number of frames."""
        if self.tracks is None:
            return None
        return len(np.unique(self.tracks[:, self.index_keys["fr_num"]]))

    @property
    def image_shape(self):
        """Image shape, (z, x) -> (height, width)"""
        return self._image_shape

    def transpose(self, *axes):
        """Transposes the order of columns in the tracks array.

        Args:
            *axes: Order of columns in the tracks array. Can be either
                integers or strings. If strings, the index_keys are used
                to convert them to integers.

        Raises:
            ValueError: If not all axes are integers or strings.

        Returns:
            ndarray: Tracks of shape (n_tracks, 6) with the new order of columns.
        """
        if all(isinstance(ax, int) for ax in axes):
            # If all axes are integers, transpose the array and update the index_keys
            pass
        elif all(isinstance(ax, str) for ax in axes):
            # If all axes are strings, convert them to integers using the index_keys
            axes = [self.index_keys[ax] for ax in axes]
        else:
            raise ValueError("All axes must be either all integers or all strings")

        self.tracks = self.tracks[:, axes]

        # list out keys and values separately
        key_list = list(self.index_keys.keys())
        val_list = list(self.index_keys.values())

        # reorder index_keys, by looping over new order defined by axes
        self.index_keys = dict(
            zip(
                [key_list[val_list.index(ax)] for ax in axes],
                range(len(axes)),
            )
        )

        return self.tracks

    def filter(self, tracks, key: str, value: Union[float, list]):
        """Filter tracks based on key and value.

        Args:
            key (str): key to filter on.
            value (float, list): value or list of values to filter on.

        Returns:
            ndarray: tracks filtered on key and value.
        """
        if isinstance(value, (list, np.ndarray)):
            tracks = tracks[np.isin(tracks[:, self.index_keys[key]], value)]
            return tracks
        else:
            tracks = tracks[tracks[:, self.index_keys[key]] == value]
            return tracks

    def _assert_tracks_image_shape(self):
        """Assert that tracks are within image shape."""
        if self.tracks is None:
            return
        if isinstance(self.tracks, list):
            tracks = self.list_to_array(self.tracks)
        else:
            tracks = self.tracks
        if tracks[:, self.index_keys["px"]].max() > self.image_shape[1]:
            raise ValueError(f"Tracks are outside of image shape: {self.image_shape}")
        if tracks[:, self.index_keys["pz"]].max() > self.image_shape[0]:
            raise ValueError(f"Tracks are outside of image shape: {self.image_shape}")

    def postprocess(
        self,
        min_length: int = 20,
        max_linking_distance: int = 10,
        upsampling_factor: int = 1,
        smooth_factor: int = None,
        **kwargs,
    ):
        """Post process tracks."""
        self.tracks_post = postprocess_tracks(
            self.tracks,
            min_length=min_length,
            max_linking_distance=max_linking_distance,
            upsampling_factor=upsampling_factor,
            smooth_factor=smooth_factor,
            index_keys=self.index_keys,
            **kwargs,
        )
        return self.tracks_post

    def tracks_to_maps(
        self,
        upsampling_factor: int = 1,
        min_length: int = 20,
        max_linking_distance: int = 10,
        smooth_factor: int = None,
    ):
        """Converts tracks to maps.

        If the tracks_post attribute is None, it falls back to using the tracks attribute.
        The resulting maps include density_map, vel_x_map, vel_z_map, and vel_norm_map.

        Args:
            upsampling_factor (int): Factor to upsample the maps with.
            min_length (int): minimum length of tracks to keep
            max_linking_distance (int): maximum distance between two points to link them
            smooth_factor (int): Amount of smoothening to apply

        Returns:
            A tuple containing the density_map, vel_x_map, vel_z_map, and vel_norm_map.
        """
        if self.tracks_post is None:
            print("Cannot find tracks_post. Post processing tracks.")
            tracks_post = self.postprocess(
                min_length=min_length,
                max_linking_distance=max_linking_distance,
                upsampling_factor=upsampling_factor,
                smooth_factor=smooth_factor,
            )
        else:
            tracks_post = self.tracks_post

        if isinstance(tracks_post, list):
            upsampling_factor_array = np.ones(tracks_post[0].shape[-1])
            upsampling_factor_array[[self.index_keys["px"], self.index_keys["pz"]]] = (
                upsampling_factor
            )
            tracks_post = [track * upsampling_factor_array for track in tracks_post]

        elif isinstance(tracks_post, np.ndarray):
            tracks_post[
                :, [self.index_keys["px"], self.index_keys["pz"]]
            ] *= upsampling_factor
        else:
            raise ValueError(
                f"tracks_post must be list or ndarray, got {type(tracks_post)}"
            )

        target_shape = np.array(self.image_shape) * upsampling_factor

        maps = tracks_to_map(
            tracks_post,
            target_shape,
            index_keys=self.index_keys,
        )

        return (
            maps["density_map"],
            maps["vel_x_map"],
            maps["vel_z_map"],
            maps["vel_norm_map"],
        )


class ReadMaps(ReadH5):
    def __init__(self, file_path):
        super().__init__(file_path)

    def read(self):
        self.maps = self.h5f["maps"]
        self.ulm_den = self.maps["ulm_den"][:]
        self.ulm_vz = self.maps["ulm_vz"][:]
        self.ulm_v = self.maps["ulm_v"][:]

        return self.ulm_den, self.ulm_vz, self.ulm_v

    @property
    def shape(self):
        return self.ulm_den.shape

    def __len__(self):
        return 3


class Config(dict):
    """Config class.

    This Config class extends a normal dictionary with easydict such that
    values can be accessed as class attributes. Furthermore it enables
    saving and loading to a yaml.

    """

    def __init__(self, dictionary=None, **kwargs):
        if dictionary is None:
            dictionary = {}
        if kwargs:
            dictionary.update(**kwargs)
        for k, v in dictionary.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__:
            if not (k.startswith("__") and k.endswith("__")):
                if k not in ["update", "serialize", "deep_copy", "save_to_yaml"]:
                    setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        else:
            value = self.__class__(value) if isinstance(value, dict) else value
        super().__setattr__(name, value)
        self[name] = value

    def update(self, override_dict):
        for name, value in override_dict.items():
            setattr(self, name, value)

    def serialize(self):
        """Serialize config object to dictionary"""
        dictionary = {}
        for key, value in self.items():
            if isinstance(value, Config):
                dictionary[key] = value.serialize()
            elif isinstance(value, Path):
                dictionary[key] = str(value)
            else:
                dictionary[key] = value
        return dictionary

    def deep_copy(self):
        """Deep copy"""
        return Config(copy.deepcopy(self.serialize()))

    def save_to_yaml(self, path):
        """Save config contents to yaml"""
        with open(Path(path), "w", encoding="utf-8") as save_file:
            yaml.dump(
                self.serialize(),
                save_file,
                default_flow_style=False,
                sort_keys=False,
            )


def load_config_from_yaml(path, loader=yaml.FullLoader):
    """Load config object from yaml file
    Args:
        path (str): path to yaml file.
        loader (yaml.Loader, optional): yaml loader. Defaults to yaml.FullLoader.
            for custom objects, you might want to use yaml.UnsafeLoader.
    Returns:
        Config: config object.
    """
    with open(Path(path), "r", encoding="utf-8") as file:
        dictionary = yaml.load(file, Loader=loader)
    if dictionary:
        return Config(dictionary)
    else:
        return {}
