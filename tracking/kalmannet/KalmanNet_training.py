# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:02:28 2022

@author: s153325
"""

"""# **Class: KalmanNet**"""

import tensorflow as tf


class KalmanNetNN(tf.keras.Model):  # torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    #############
    ### Build ###
    #############
    def Build(self, config, ssModel=None):

        self.InitSystemDynamics(ssModel.F, ssModel.H)

        self.hidden_dim_factor = config.hidden_dim_factor
        self.N_neur_L1 = config.N_neur_L1
        self.N_neur_L2 = config.N_neur_L2
        self.batch_size = config.batch_size  # Batch Size
        self.dropout = config.dropout

        # Number of neurons in the 1st hidden layer
        H1_KNet = (self.m + self.n) * (self.N_neur_L1) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (self.m * self.n) * (self.N_neur_L2)

        self.InitKGainNet(H1_KNet, H2_KNet, n_layers=config.n_layers)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2, n_layers=1):

        # Input Dimensions
        D_in = self.m + self.n + self.n  # x(t-1), y(t), ^y(t)
        self.y_prev = 0

        # # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################

        # Linear Layer
        self.KG_l1 = tf.keras.layers.Dense(
            units=H1, use_bias=True, input_shape=(None, D_in, 1)
        )
        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = tf.keras.layers.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * self.hidden_dim_factor
        # Number of Layers
        self.n_layers = n_layers
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for Hidden State
        self.hn = tf.random.normal(
            (self.seq_len_hidden, self.batch_size, self.hidden_dim)
        )

        # tf.keras.layers.GRU(self.hidden_dim, return_sequences=True,  return_state=True, stateful=True)
        # tf.keras.layers.GRU(self.hidden_dim, return_sequences=False,  return_state=True, stateful=True)
        # Iniatialize GRU Layer
        self.GRU = tf.keras.layers.GRU(
            self.hidden_dim, return_sequences=False, return_state=True, stateful=True
        )  # ,input_shape=(None,None,96))
        # self.GRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim, return_sequences=True,  return_state=True, stateful=True))
        # d = {}

        # for i in range(1, self.n_layers+1):
        #     d['GRU_%02d' % i] = tf.keras.layers.GRU(self.hidden_dim, return_sequences=False,  return_state=True, stateful=True)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = tf.keras.layers.Dense(units=H2, use_bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = tf.keras.layers.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = tf.keras.layers.Dense(
            units=D_out, use_bias=True
        )  # torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = F
        self.m = self.F.shape[0]

        # Set Observation Matrix
        self.H = H
        self.n = self.H.shape[0]

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0=None, batch_size=0, reset_hidden_state=True):
        if not batch_size:
            batch_size = self.batch_size
        if M1_0 is None:
            self.m1x_prior = tf.zeros((batch_size, 4, 1))
            self.m1x_posterior = tf.zeros((batch_size, 4, 1))
        else:
            self.m1x_prior = M1_0
            self.m1x_posterior = M1_0

        if reset_hidden_state:
            self.hn = tf.random.normal(
                (self.seq_len_hidden, batch_size, self.hidden_dim)
            )

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):  # predict step

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = tf.linalg.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = tf.linalg.matmul(self.H, self.m1x_prior)

    def predict(self):
        return tf.squeeze(self.H @ (self.F @ self.m1x_posterior))

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        # Feature 4: x_t|t - x_t|t-1
        dm1x = self.m1x_posterior - self.m1x_prev_prior

        # print(dm1x.shape, y.shape)
        dm1x_reshape = tf.reshape(dm1x, (y.shape[0], 4))
        dm1x_norm = tf.nn.l2_normalize(dm1x_reshape, axis=1, epsilon=1e-12)

        # Feature 2: yt - y_t+1|t
        dm1y = y - tf.reshape(self.m1y, (y.shape[0], 2))
        dm1y_norm = tf.nn.l2_normalize(dm1y, axis=1, epsilon=1e-12)

        # KGain Net Input

        # Feature 1: yt - y_t-1
        dm1y_f1 = y - self.y_prev
        dm1y_norm_f1 = tf.nn.l2_normalize(dm1y, axis=1, epsilon=1e-12)
        self.y_prev = y

        KGainNet_in = tf.concat([dm1y_norm, dm1y_norm_f1, dm1x_norm], axis=1)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)

        # Reshape Kalman Gain to a Matrix
        self.KGain = tf.reshape(KG, (y.shape[0], self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = tf.transpose(tf.expand_dims(y, axis=1), perm=(0, 2, 1))
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        INOV = tf.linalg.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        return tf.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        # L1_in = self.KG_in(KGainNet_in);
        if len(KGainNet_in.shape) == 1:
            KGainNet_in = tf.transpose(tf.expand_dims(KGainNet_in, axis=1))
        L1_out = self.KG_l1(KGainNet_in)
        # L1_out = self.KG_l1(KGainNet_in);
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = tf.transpose(tf.expand_dims(La1_out, axis=0), perm=(1, 0, 2))

        shape = (KGainNet_in.shape[0], self.hidden_dim)
        GRU_out, self.hn = self.GRU(GRU_in, initial_state=tf.reshape(self.hn, shape))

        GRU_out_reshape = tf.reshape(GRU_out, shape)

        ####################
        ### Hidden Layer ###
        ####################

        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def call(self, yt):
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        self.GRU.reset_states()
