"""
==============================================================================
	Eindhoven University of Technology
==============================================================================
	Source Name   : kalmanFilter.py
					Simulation file
	Author(s)     : Thierry Voskuil
	Modified by   : -
	Date          : 15/04/2022
==============================================================================
"""

import numpy as np


class KalmanFilter(object):
    """
    KalmanFilter 2D
    State vector X = (x, v_x, y, v_y)
    """

    def __init__(self, dt=1, stateVariance=1, measurementVariance=1, method="Velocity"):
        super(KalmanFilter, self).__init__()
        self.method = method
        self.stateVariance = stateVariance
        self.measurementVariance = measurementVariance
        self.dt = dt
        self.initModel()

    def initModel(self):
        """init function to initialize the model"""
        if self.method == "Acceleration":
            self.U = 1
        else:
            self.U = 0
        self.A = np.matrix(
            [[1, self.dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]]
        )

        # If acceleration is used
        self.B = np.matrix([[self.dt**2 / 2], [self.dt], [self.dt**2 / 2], [self.dt]])

        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.P = np.matrix(self.stateVariance * np.identity(self.A.shape[0]))
        self.R = np.matrix(self.measurementVariance * np.identity(self.H.shape[0]))

        self.Q = np.matrix(
            [
                [self.dt**4 / 4, self.dt**3 / 2, 0, 0],
                [self.dt**3 / 2, self.dt**2, 0, 0],
                [0, 0, self.dt**4 / 4, self.dt**3 / 2],
                [0, 0, self.dt**3 / 2, self.dt**2],
            ]
        )

        self.erroCov = self.P
        self.state = np.matrix(
            [[0], [0], [0], [0]]
        )  # check what happens when v_0 = 0 (now 1)

    def predict(self):
        """Predict function which predicst next state based on previous state"""
        self.predictedState = self.A * self.state + self.B * self.U
        self.predictedErrorCov = self.A * self.erroCov * self.A.T + self.Q
        temp = np.asarray(self.predictedState)
        return temp[0], temp[2]  # x,y

    def correct(self, currentMeasurement):
        """Correct function which correct the states based on measurements"""
        # H = HJacobian(self.x, *args)
        self.kalmanGain = (
            self.predictedErrorCov
            * self.H.T
            * np.linalg.pinv(self.H * self.predictedErrorCov * self.H.T + self.R)
        )
        # hx = Hx(self.x, *hx_args)
        # self.y = residual(z, hx)
        # self.x = self.x + dot(self.K, self.y)
        self.state = self.predictedState + self.kalmanGain * (
            currentMeasurement - (self.H * self.predictedState)
        )

        self.erroCov = (
            np.identity(self.P.shape[0]) - self.kalmanGain * self.H
        ) * self.predictedErrorCov
