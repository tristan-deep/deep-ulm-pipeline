"""Main entry point of running inference on localizations.

Performs the following steps:
- Load localization data from hdf5 file
- Track localizations using track parameters from config file
- Save raw tracks to file
- Postprocess tracks
- Convert tracks to maps
- Visualize and save maps

Example usage:

```bash
python main.py -c ./configs/tracking_config.yaml
```

Args:
    -c (--config): path to tracking config file.
    -s (--save): Save raw tracks to file
    --skip: Skip folders that already have tracks.

Note:
    Make sure the localizations are stored in the correct format (hdf5)
    see read.py ReadLocs class for that and also have only a single
    localization file per folder.

Author: Tristan Stevens
"""

import argparse
import shutil
from pathlib import Path

from config import Config, load_config_from_yaml
from locs_to_tracks import locs_to_tracks
from read import ReadLocs, ReadTracks
from utils import make_unique_path, yellow
from visualization import show_ulm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/tracking_config.yaml",
        help="path to tracking config file.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_false",
        default=True,
        help="Save raw tracks to file",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        default=False,
        help="Skip folders that already have tracks.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_path = args.config
    config = load_config_from_yaml(config_path)
    config = Config(config)

    print(f"Loaded config: {yellow(config_path)}")

    folders = config.localization_folder
    if isinstance(folders, str):
        folder = Path(folders)
        # check if folder is a folder or a file
        if folder.is_file():
            folders = [folder.parent]
        elif folder.is_dir():
            files = list(folder.rglob("*.hdf5")) + list(folder.rglob("*.h5"))
            folders = [file.parent for file in files]
            # omit any folder that has a subfolder tracks
            folders = [
                folder
                for folder in folders
                if not any("tracks" in part for part in folder.parts)
            ]
        else:
            raise ValueError(f"Could not find {folder}")
    else:
        folders = [Path(folder) for folder in folders]

    print(f"Found {yellow(len(folders))} files to track")

    for data_dir in folders:
        try:
            if args.skip:
                # check if an tracks.hdf5 is already present in the folder
                track_files = list(
                    [file.name for file in data_dir.rglob("tracks.hdf5")]
                    + [file.name for file in data_dir.rglob("tracks.h5")]
                )
                if len(track_files) != 0:
                    print(f"Skipping {data_dir} because it already has tracks.")
                    continue

            files = list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5"))
            assert (
                len(files) != 0
            ), f"Could not find localization file (hdf5) in {yellow(data_dir)}"
            assert (
                len(files) == 1
            ), f"Found multiple hdf5 files in {data_dir}, please only have a single localization hdf5 file per folder."

            data_file = files[0]

            print(f"\nLoading localizations from: {yellow(data_file)}\n")

            save_dir = data_dir / config.save_folder_name
            save_dir = make_unique_path(save_dir)
            save_dir = save_dir / config.tracking.track_mode
            save_dir.mkdir(exist_ok=False, parents=True)

            data = ReadLocs(data_file)
            data.read()

            config.image_shape = data.image_shape
            config.super_res_shape = data.image_shape * config.tracking.upscale

            # Copy configs over to save directory
            shutil.copy(config_path, save_dir)

            print("Tracking...")
            raw_tracks = locs_to_tracks(
                data.locs,
                index_keys=data.index_keys,
                track_mode=config.tracking.track_mode,
                out_format="list",
                mode_loc=config.tracking.mode_MB_loc,
                mode_vel=config.tracking.mode_vel,
                min_length=config.tracking.min_length,
                frame_rate=config.tracking.frame_rate,
                max_frame_skipped=config.tracking.max_frame_skipped,
                max_linking_distance=config.tracking.max_linking_distance,
            )

            tracks_obj = ReadTracks(
                tracks=raw_tracks,
                image_shape=data.image_shape,
                init_index_order=[
                    "pz",
                    "px",
                    "vz",
                    "vx",
                    "fr_num",
                    "track_id",
                    "py",
                ],
            )

            # Save raw tracks
            if args.save:
                # save_tracks(raw_tracks, output_dir=save_dir, filetype="h5")
                # print(f"Succesfully saved raw tracks to: {save_dir}")
                tracks_obj.save(save_dir / "tracks.h5")

            # Postprocess
            if config.tracking.postprocess:
                # from utils.utils import list_to_array_idx
                # tracks = list_to_array_idx(tracks)
                tracks_obj.postprocess(
                    interpolation=config.tracking.interpolation,
                    smooth_factor=config.tracking.smooth_factor,
                    min_length=config.tracking.min_length,
                    upsampling_factor=config.tracking.upscale,
                )
            (
                ulm_den,
                ulm_vz,
                ulm_vx,
                ulm_v,
            ) = tracks_obj.tracks_to_maps(upsampling_factor=config.tracking.upscale)

            show_ulm(
                ulm_den=ulm_den,
                ulm_vz=ulm_vz,
                ulm_vx=ulm_vx,
                ulm_v=ulm_v,
                save_dir=save_dir,
                axis=False,
                colorbar=False,
                **config.visualization,
            )
        except Exception as e:
            print(f"Failed to track {data_file} because of: {e}")
            raise e


if __name__ == "__main__":
    main()
