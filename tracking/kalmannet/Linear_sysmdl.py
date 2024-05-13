# Imports
# %%
import time

import matplotlib.pyplot as plt
import numpy as np


class timer(object):
    def __init__(self, k=3, print_t=0):
        self.t0 = time.time()
        self.k = k
        self.P = print_t

    def set_t(self):
        self.t0 = time.time()

    def time(self, print_text=""):
        if self.P:
            print("Timer ", print_text, " = ", round(time.time() - self.t0, self.k))
        self.t0 = time.time()


def smooth(a, WSZ):  # place in utils
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), "valid") / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[: WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


# %%
class SystemModel:

    def __init__(
        self,
        F=None,
        q=0.5,
        H=None,
        r=0.1,
        T=100,
        dt=1e-2,
        outlier_p=0,
        rayleigh_sigma=10000,
    ):

        self.outlier_p = outlier_p
        self.rayleigh_sigma = rayleigh_sigma
        self.dt = dt
        self.T = T
        ####################
        ### Motion Model ###
        ####################
        if F is None:
            self.F = np.array(
                [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]],
                dtype="float32",
            )
        else:
            self.F = F.astype("float32")
        self.m = self.F.shape[0]

        self.q = q
        self.Q = q * q * np.eye(self.m)

        #########################
        ### Observation Model ###
        #########################
        if H is None:
            self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype="float32")
        else:
            self.H = H.astype("float32")
        self.n = self.H.shape[0]

        self.r = r
        self.R = r * r * np.eye(self.n)

        # Init sequence
        self.m1x_0 = np.zeros((4, 1))
        self.m2x_0 = np.zeros((4))

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):
        self.q = q
        self.Q = q * q * np.eye(self.m).astype("float32")
        self.r = r
        self.R = r * r * np.eye(self.n).astype("float32")

    def UpdateCovariance_Matrix(self, Q, R):
        self.Q = Q.astype("float32")
        self.R = R.astype("float32")

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        x = []
        y = []

        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        # Outliers
        if self.outlier_p > 0:
            b_matrix = np.random.binomial(1, self.outlier_p, T).astype("float32")

        # Generate Sequence Iteratively
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################
            # Process Noise
            if self.q == 0:
                xt = self.F @ self.x_prev
            else:
                xt = self.F @ self.x_prev
                mean = np.zeros([self.m])
                eq = np.random.multivariate_normal(mean, Q_gen, 1).T.astype("float32")
                eq = np.reshape(eq[:], [self.m, 1])
                xt = np.add(xt, eq)  # Additive Process Noise

            ################
            ### Emission ###
            ################
            # Observation Noise
            if self.r == 0:
                yt = self.H @ xt
            else:
                yt = self.H @ xt
                mean = np.zeros([self.n])
                er = np.random.multivariate_normal(mean, R_gen, 1).T.astype("float32")
                er = np.reshape(er[:], [self.n, 1])
                yt = np.add(yt, er)  # Additive Observation Noise

            # Outliers
            if self.outlier_p > 0:
                if b_matrix[t] != 0:
                    btdt = self.rayleigh_sigma * np.sqrt(
                        -2 * np.log(np.random.uniform(size=(self.n, 1)))
                    )
                    yt = np.add(yt, btdt)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            x.append(np.squeeze(xt))

            # Save Current Observation to Trajectory Array
            y.append(np.squeeze(yt))

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt
        self.x = np.squeeze(np.stack(x, axis=1))
        self.y = np.squeeze(np.stack(y, axis=1))

    ######################
    ### Generate Batch ###
    ######################

    def GenerateBatch(
        self,
        size,
        T,
        randomInit=True,
        seqInit=False,
        p_range=[0, 100],
        v_range=[-1000, 1000],
        smooth_b=True,
    ):

        Input = []
        Target = []

        ### Generate Examples
        initConditions = self.m1x_0
        for i in range(0, size):
            # Generate Sequence

            # Randomize initial conditions to get a rich dataset
            if randomInit:
                initConditions = np.zeros(self.m1x_0.shape)
                initConditions[[0, 2]] = np.random.uniform(
                    p_range[0], p_range[1], initConditions[[0, 2]].shape
                )
                initConditions[[1, 3]] = np.random.uniform(
                    v_range[0], v_range[1], initConditions[[1, 3]].shape
                )

            if seqInit:
                initConditions = self.x_prev
                if (i * T % T) == 0:
                    initConditions = np.zeros(self.m1x_0.shape)

            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            if smooth_b:
                # Training sequence input
                Input.append(np.array([smooth(part, 15) for part in self.y]))
                # Training sequence output
                Target.append(np.array([smooth(part, 15) for part in self.x]))
            else:
                # Training sequence input
                Input.append(self.y)
                # Training sequence output
                Target.append(self.x)

        self.Input = np.squeeze(np.stack(Input))
        self.Target = np.squeeze(np.stack(Target))

    def GenerateTrainTestValData(
        self, N_train, N_test, N_val, Nt=100, p_range=[0, 100], v_range=[-1000, 1000]
    ):
        sizes = [N_train, N_val, N_test]
        print(v_range)
        data = []
        for i in range(3):
            self.GenerateBatch(
                sizes[i], Nt, randomInit=True, p_range=p_range, v_range=v_range
            )
            data += [self.Input] + [self.Target]
        return data
