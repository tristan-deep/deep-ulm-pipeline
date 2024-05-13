import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from tracking.kalman_filter.kalman_filter import KalmanFilter


class Tracks(object):
    """docstring for Tracks"""

    def __init__(self, detection, trackId, dt=1, stateVariance=50, method="Velocity"):
        super(Tracks, self).__init__()
        detection_in, detection = self.split_detection(detection)
        # Initialize Kalman Filter
        self.KF = KalmanFilter(dt=dt, stateVariance=stateVariance, method=method)
        self.KF.predict()
        self.detection = detection
        self.KF.correct(np.matrix(detection[:2]).reshape(2, 1))
        self.KF.predict()
        self.KF.correct(np.matrix(detection[:2]).reshape(2, 1))
        self.KF.predict()
        # Set class attributes
        self.prediction = detection[:2].reshape(1, 2)
        self.trackId = trackId
        self.skipped_frames = 0
        self.detectionsHist = [detection_in]  # [detection]
        self.statesHist = [self.KF.state]
        self.predictionsHist = [self.prediction]

    def update(self, detection):
        detection_in, detection = self.split_detection(detection)
        # Update KF
        self.KF.correct(np.matrix(detection[:2]).reshape(2, 1))
        self.statesHist.append(self.KF.state)
        if self.detectionsHist[-1][0] != detection_in[0]:
            self.detectionsHist.append(detection_in)
        self.prediction = np.array(self.KF.predict()).reshape(1, 2)
        self.predictionsHist.append(self.prediction)

    def getPrediction(self, k=5):
        if not len(self.detectionsHist):
            out = self.detection
        elif len(self.detectionsHist) < k:
            out = self.detectionsHist[-1][1:3]
        else:
            out = (self.prediction + self.detectionsHist[-1][1:3]) / 2
        return out

    def split_detection(self, detection_in):
        return detection_in, detection_in[1:3]


class Tracker(object):
    """
    Class for the Kalman Filter based tracking of MBs

    """

    def __init__(
        self, dist_threshold, max_frame_skipped, dt=1.0, trackId=0, initWindow=5
    ):
        """
        Init function of tracker

        Parameters
        ----------
        dist_threshold : float
            maximum distance between two consecutive MBs on a track.
        max_frame_skipped : int
            maximum number of frames a track is allowed to skip.
        dt : float, optional
            time interval, change in time. The default is 1.0.
        trackId : int, optional
            The id which should be given to the first track. This parameter can be used to avoid having
            tracks with identical trackIds when the tracker is used on multiple files. The default is 0.
        initWindow : int, optional
            Initialization window. The default is 5.

        Returns
        -------
        None.

        """
        super(Tracker, self).__init__()
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.dt = dt
        self.trackId = trackId if trackId is not None else 0
        self.tracks = []
        self.costHist = []
        self.trackArchive = []
        self.initWindow = initWindow  # Initialization window

    def update(self, detectionsInput):
        """
        Function to update the Tracker. The function takes in the detections and determines the
        cost (Eucledian distance) between the detections and the predicted MB locations. With this
        cost the detections will be assigned to tracks with the Hungarian linker which is implemented
        with scipy's linear sum assignment function. The assignment if rejected if the distance between
        the assigned MB and the last MB on the track exceeds the distance threshold.

        Parameters
        ----------
        detectionsInput : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if detectionsInput.shape[1] >= 4:
            detections = detectionsInput[:, [1, 2]].copy()
        elif detectionsInput.shape[1] == 3:
            detections = detectionsInput[:, :2].copy()
        else:
            detections = detectionsInput.copy()

        if len(self.tracks) == 0:
            for i in range(detections.shape[0]):
                track = Tracks(detectionsInput[i], self.trackId, dt=self.dt)
                self.trackId += 1
                self.tracks.append(track)
        else:
            N = len(self.tracks)
            M = len(detections)
            cost = []

            # for i in range(N):
            #     diff = np.linalg.norm(self.tracks[i].getPrediction(self.initWindow) - detections.reshape(-1,2), axis=1)
            #     cost.append(diff)

            # cost = np.array(cost)
            # cost = cost/np.max(cost)
            # self.costHist.append(cost)
            predictions = np.array(
                [
                    track.getPrediction(self.initWindow).squeeze()
                    for track in self.tracks
                ]
            )
            cost = pairwise_distances(predictions, detections.reshape(-1, 2))
            row, col = linear_sum_assignment(cost)  # Hungarian linker
            assignment = [-1] * N
            for i in range(len(row)):
                assignment[row[i]] = col[i]

            un_assigned_tracks = []

            for i in range(len(assignment)):
                if assignment[i] != -1 and len(self.tracks[i].detectionsHist) > 0:
                    distance = np.linalg.norm(
                        detections[assignment[i]]
                        - self.tracks[i].detectionsHist[-1][1:3]
                    )  # Distance to previous point
                    if (
                        distance > self.dist_threshold
                    ):  # Reject assignment if distance is larger than distance threshold
                        assignment[i] = -1
                        un_assigned_tracks.append(i)

                if assignment[i] == -1:
                    self.tracks[i].skipped_frames += 1

            for i in range(len(assignment)):
                if assignment[i] != -1:
                    self.tracks[i].skipped_frames = 0
                    self.tracks[i].update(detectionsInput[assignment[i]])  # update
                    # self.tracks[i].detectionsHist.append(detectionsInput[assignment[i]])

            # Delete tracks with too many skipped frames
            del_tracks = []
            for i in range(len(self.tracks)):
                if self.tracks[i].skipped_frames > self.max_frame_skipped:
                    del_tracks.append(i)

            if len(del_tracks) > 0:
                for i in range(len(del_tracks), 0, -1):
                    j = del_tracks[i - 1]
                    self.trackArchive.append(
                        self.tracks[j]
                    )  # Save tracks before deleting
                    del self.tracks[j]
                    del assignment[j]

            # Assign new tracks to unassigned MBs
            for i in range(len(detections)):
                if i not in assignment:
                    track = Tracks(detectionsInput[i], self.trackId, dt=self.dt)
                    # track.detectionsHist.append(detectionsInput[i])
                    self.trackId += 1
                    self.tracks.append(track)

    def retrieveTracks(
        self,
        mode_format="array",
        mode_MB_loc="localization",
        mode_vel="state",
        min_length=1,
    ):
        """
        This function retrieves tracks from the Tracker object.

        Parameters
        ----------
        mode_format : str, optional
            Determines in which datatype the tracks will be returned, if set to 'array' the function will
            return a (6xN) numpy array, with columns [pz, px, v_z, v_x, fr_num, trackId]. If set to
            'list' the function will return a list of tracks. Each track is a (6xM) array, with columns:
            [pz, px, v_z, v_x, fr_num] and with M the number of MBs on the track. The default is 'array'.
        mode_MB_loc : str, optional
            Determines which coordinates to take for the MB position. If set to 'localization' the function
            will use the localized MB coordinates. If set to 'state' the coordinates will be taken from the
            Kalman states history. If set to 'initState' the function will take the localized coordinates
            in a specified initialization window (of length k), after this window the function will take
            the coordinates from the KF states history. The default is 'localization'.
        mode_vel : str, optional
            Determine which MB velocities to take, which can be either from the states history or it can be
            calculated from the MB positions. The default is 'localization'.. The default is 'state'.
        min_length : int, optional
            The minimal length a track should be. The default is 1.

        Returns
        -------
        combinedTracks : np.array or list
            The function returns either a list or a numpy ndarray containing all the tracks.

        """
        tracks = self.tracks + self.trackArchive
        combinedTracks = []
        k = self.initWindow  # Initialization window
        for track in tracks:
            N = min(
                len(track.statesHist), len(track.detectionsHist)
            )  # Number of points on track
            if N > min_length - 1:
                track.detectionsHist = np.array(track.detectionsHist)
                track.statesHist = np.array(track.statesHist)
                trackNew = np.zeros((N, track.detectionsHist.shape[1] + 3))
                # Set MB coordinates
                if mode_MB_loc == "localization":  # Take MB positions from localization
                    trackNew[:, :2] = (
                        track.detectionsHist[:N, 1:3].copy().squeeze()
                    )  # z,x
                elif mode_MB_loc == "initState":
                    if N > k:
                        trackNew[:k, :2] = (
                            track.detectionsHist[:k, 1:3].squeeze().copy()
                        )
                        trackNew[k:, :2] = (
                            track.statesHist[k:N, [0, 2]].squeeze().copy()
                        )
                    else:
                        trackNew[:N, :2] = (
                            track.detectionsHist[:N, 1:3].squeeze().copy()
                        )
                elif mode_MB_loc == "state":  # Take MB positions from states history
                    trackNew[:, :2] = track.statesHist[:N, [0, 2]].squeeze().copy()
                else:
                    raise (
                        NotImplementedError(
                            "mode_MB_loc not specified, please set mode_MB_loc to one of the following: 'localization', 'state', 'initState'"
                        )
                    )

                if N > 1:
                    # Set MB velocities
                    if mode_vel == "state":  # Take velocity from states history
                        trackNew[:, 2:4] = (
                            track.statesHist[:N, [1, 3]].copy().squeeze()
                        )  # z, x
                    elif (
                        mode_vel == "initState"
                    ):  # Take velocity from states after initialization window
                        if N > k:
                            trackNew[1:k, 2:4] = (
                                track.detectionsHist[1:k, 1:3]
                                - track.detectionsHist[: k - 1, 1:3]
                            ) / (
                                np.expand_dims(
                                    track.detectionsHist[1:k, 0]
                                    - track.detectionsHist[: k - 1, 0],
                                    axis=1,
                                )
                                * self.dt
                            )  # Second part for including missing frames
                            trackNew[0, 2:4] = np.mean(trackNew[1:k, 2:4], axis=0)
                            trackNew[k:, 2:4] = (
                                track.statesHist[k:N, [1, 3]].copy().squeeze()
                            )
                        else:
                            trackNew[1:N, 2:4] = (
                                track.detectionsHist[1:N, 1:3]
                                - track.detectionsHist[: N - 1, 1:3]
                            ) / (
                                np.expand_dims(
                                    track.detectionsHist[1:N, 0]
                                    - track.detectionsHist[: N - 1, 0],
                                    axis=1,
                                )
                                * self.dt
                            )  # Second part for including missing frames
                            trackNew[0, 2:4] = np.mean(trackNew[1:N, 2:4], axis=0)
                    elif (
                        mode_vel == "calculate"
                    ):  # Calculate velocity from MB positions
                        # if N > 2:
                        trackNew[1:, 2:4] = (
                            track.detectionsHist[1:N, 1:3]
                            - track.detectionsHist[: N - 1, 1:3]
                        ) / (
                            np.expand_dims(
                                track.detectionsHist[1:N, 0]
                                - track.detectionsHist[: N - 1, 0],
                                axis=1,
                            )
                            * self.dt
                        )  # Second part for including missing frames
                        if N > 4:
                            trackNew[0, 2:4] = np.mean(trackNew[1:5, 2:4], axis=0)
                        else:
                            trackNew[0, 2:4] = np.mean(trackNew[1:, 2:4], axis=0)
                    else:
                        raise (
                            NotImplementedError(
                                "mode_vel not specified, please set mode_vel to one of the following: 'state', 'initState', 'calculate'"
                            )
                        )

                # Set frame number
                trackNew[:, 4] = track.detectionsHist[:N, 0].copy()
                # Set track id
                trackNew[:, 5] = track.trackId

                if (
                    trackNew.shape[1] > 6
                ):  # Copy other info (actual trackId and actual speed)
                    trackNew[:, 6:] = track.detectionsHist[:N, 3:].copy()

                if mode_format == "array":
                    if not len(combinedTracks):
                        combinedTracks = trackNew.copy()
                    else:
                        combinedTracks = np.concatenate(
                            [combinedTracks, trackNew], axis=0
                        )
                elif mode_format == "list":  # List of tracks
                    combinedTracks += [trackNew]
                else:
                    raise (
                        NotImplementedError(
                            "mode_format not specified, please set mode_format to one of the following: 'array', 'list'"
                        )
                    )

        return combinedTracks
