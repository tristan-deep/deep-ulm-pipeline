"""Generate tracks from localizations.

Author: Tristan Stevens
"""

import numpy as np
import tqdm

from tracking.kalman_filter.tracker import Tracker as KF_tracker
from tracking.kalmannet.tracker import Tracker as Knet_tracker
from tracking.simple.tracker import Tracker as simple_tracker
from utils import array_idx_to_list, list_to_array_idx


def locs_to_tracks(
    localizations,
    index_keys: list,
    track_mode: str,
    out_format: str,
    mode_loc: str,
    mode_vel: str,
    track_id: int = None,
    min_length: int = 1,
    reverse: bool = False,
    frame_rate: float = 1.0,
    max_frame_skipped: int = 3,
    max_linking_distance: int = 10,
):
    track_mode = str(track_mode).lower()
    assert track_mode in ["simple", "kf", "knet"], "Unknown track_mode"
    assert out_format in ["array", "list"], "Unknown out_format"
    assert mode_loc in ["localization", "state"], "Unknown mode_MB_loc"
    assert mode_vel in ["calculate", "state"], "Unknown mode_vel"

    assert (
        len(index_keys) == localizations[0].shape[0]
    ), "index_keys must have the same length as the number of columns in localizations"

    if isinstance(localizations, list):
        localizations = list_to_array_idx(localizations)

    localizations = localizations[
        :,
        [
            index_keys["fr_num"],
            index_keys["pz"],
            index_keys["px"],
            index_keys["intensity"],
        ],
    ]

    # Parameters
    # frame nr is in column 0, in the matlab file column 3
    min_frame = min(localizations[:, 0])
    # Renormalizes to take into account that not all matrices start with frame number 1
    localizations[:, 0] = localizations[:, 0] - min_frame
    num_frames = len(np.unique(localizations[:, 0]))
    localizations = localizations.astype("float32")
    return_track_id = bool(track_id)

    # Tracking
    dt = 1 / frame_rate
    if track_mode == "simple":
        tracker = simple_tracker(
            max_linking_distance, max_frame_skipped, dt=dt, trackId=track_id
        )  # Initialize tracker
    elif track_mode == "kf":
        tracker = KF_tracker(
            max_linking_distance, max_frame_skipped, dt=dt, trackId=track_id
        )  # Initialize tracker
    elif track_mode == "knet":
        tracker = Knet_tracker(
            max_linking_distance, max_frame_skipped, dt=dt, trackId=track_id
        )  # Initialize tracker
    else:
        raise (
            NotImplementedError(
                "Unknown track_mode, please set track_mode to 'simple', 'KF' or 'Knet'"
            )
        )

    # convert to list for faster indexing
    localizations_list = array_idx_to_list(localizations, localizations[:, 0])

    for i in tqdm.tqdm(range(num_frames)):
        if reverse:
            i = int(num_frames - min_frame) - i

        centers = localizations_list[i]

        if len(centers) > 0:
            tracker.update(centers)

    tracks_out = tracker.retrieveTracks(
        mode_format=out_format,
        mode_MB_loc=mode_loc,
        mode_vel=mode_vel,
        min_length=min_length,
    )
    # Set trackId for new file
    track_id = tracker.tracks[-1].trackId

    if return_track_id:
        return tracks_out, track_id
    else:
        return tracks_out
