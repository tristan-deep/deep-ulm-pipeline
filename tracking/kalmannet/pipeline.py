import time

import numpy as np
import tensorflow as tf
import wandb


# %%
class dotdict(dict):  # remove if in utils
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Pipeline_KF:

    def __init__(self, modelName, save_model=True):
        super().__init__()
        self.modelName = modelName
        self.save_model = save_model

    def save(self, add_params_name=True, additional_name=None):
        if add_params_name:
            additional_name = (
                "_L1_"
                + ("%02d" % self.model.N_neur_L1)
                + "_L2_"
                + ("%02d" % self.model.N_neur_L2)
                + "_HN_"
                + ("%02d" % self.model.hidden_dim_factor)
            )
        elif additional_name is None:
            additional_name = ""
        self.model_name = self.wd / "checkpoints" / (self.modelName + additional_name)
        if self.save_model:
            self.model.save_weights(self.model_name)

    def load(self, checkpoint="KNet_TF_mdl_06_14_22", load_params_from_name=True):
        model_name = "./checkpoints/" + checkpoint
        self.model.load_weights(model_name)
        if load_params_from_name:
            params = ["L1", "L2", "HN"]
            [L1, L2, HN] = [
                int(checkpoint[checkpoint.find(p) + 3 : checkpoint.find(p) + 5])
                for p in params
            ]
            self.model.Build(
                self.ssModel,
                N_neur_L1=L1,
                N_neur_L2=L2,
                hidden_dim_factor=HN,
                batch_size=64,
            )

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, config, wd=""):
        self.N_Epochs = config.epochs  # Number of Training Epochs
        self.N_B = config.batch_size  # Number of Samples in Batch
        self.learningRate = config.learning_rate  # Learning Rate

        # MSE LOSS Function
        self.loss_fn = (
            tf.keras.losses.MeanSquaredError()
        )  # nn.MSELoss(reduction='mean')
        self.config = config
        self.wd = wd
        # if config.wandb:
        #     self.data_config = {
        #         "n_train": 5000,    # number of train samples
        #         "n_test": 1000,      # number of test samples
        #         "n_val": 1000,       # number of validation samples
        #         "n_t": 100,          # number of timesteps
        #         "dt": 1e-2,
        #         "method": "both",    # 'nonlinear', 'linear' or 'both'
        #         "save": True
        #         }
        #     self.data_config = dotdict(self.data_config)
        # else:
        self.data_config = dotdict(config.data_config)
        self.dt = self.data_config.dt

        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learningRate
        )  # , weight_decay=self.weightDecay)
        self.train_acc_metric = (
            tf.keras.metrics.MeanSquaredError()
        )  # SparseCategoricalAccuracy()
        set_loss_weight = True  # config.loss_weight
        if set_loss_weight:
            self.loss_w = tf.eye(4) * tf.constant([1, self.dt, 1, self.dt])
        else:
            self.loss_w = tf.cast(tf.eye(4), tf.float32)

    # @tf.function
    def train_step(self, x, x_target):
        with tf.GradientTape() as tape:
            x_pred_list = []
            y_pred_list = []
            for i in range(x.shape[2]):
                # Predict corrected state
                x_pred_list.append(self.model(x[:, :, i], training=True))
                # Predict next observation
                y_pred_list.append(self.model.predict())
            x_pred = tf.stack(x_pred_list, axis=2)
            y_pred = tf.squeeze(tf.stack(y_pred_list, axis=2))
            y_target = tf.stack(
                [x_target[:, 0, :], x_target[:, 2, :]], axis=1
            )  # check dimensions
            # Calculate losses
            vars = self.model.trainable_weights
            if self.config.L2_lambda:
                loss_L2 = (
                    tf.add_n([tf.nn.l2_loss(v) for v in vars if "bias" not in v.name])
                    * self.config.L2_lambda
                )  # L2 loss
            else:
                loss_L2 = 0
            loss_y = self.loss_fn(y_pred[:, :, :-1], y_target[:, :, 1:])
            loss_value = (
                self.loss_fn(self.loss_w @ x_target, self.loss_w @ x_pred) + loss_L2
            )

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # gradients = [(tf.clip_by_value(grad, -1., 1.)) for grad in grads]
        # or:
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        grads = [
            tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in grads
        ]  # [(tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad), val) for grad,val in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(self.loss_w @ x_target, self.loss_w @ x_pred)
        return loss_value, loss_y

    def NNtrain(self, train_input, train_target, cv_input, cv_target):

        self.N_E = self.data_config.n_train
        self.N_CV = self.data_config.n_val

        MSE_cv_linear_epoch_list = []
        MSE_cv_dB_epoch_list = []

        MSE_train_linear_epoch_list = []
        MSE_train_dB_epoch_list = []

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        start_time = time.time()
        for ti in range(0, self.N_Epochs):

            #     #################################
            #     ### Validation Sequence Batch ###
            #     #################################

            epoch_time = time.time()
            MSE_cv_linear_batch_list = []
            MSE_cv_pred_batch_list = []
            for j in range(0, self.N_CV // self.N_B):
                idx1 = j * self.N_B
                idx2 = (j + 1) * self.N_B
                y_cv = cv_input[idx1:idx2, :, :]
                # init_seq = tf.constant([y_cv[0, 0].numpy(), 0., y_cv[1, 0].numpy(), 0.], shape=(4,1))
                self.model.InitSequence()

                # x_out_cv = tf.zeros((self.ssModel.m, self.ssModel.T))
                x_out_list = []
                y_out_list = []
                for t in range(self.ssModel.T):
                    # Predict corrected state
                    x_out_list.append(self.model(y_cv[:, :, t], training=False))
                    # Predict next observation
                    y_out_list.append(self.model.predict())
                    # print(self.loss_fn(self.loss_w @ tf.expand_dims(x_out_list[-1],axis=2), self.loss_w @ tf.expand_dims(cv_target[idx1:idx2, :, t],axis=2)).numpy())
                x_out_cv = tf.stack(x_out_list, axis=2)
                y_out_cv = tf.squeeze(tf.stack(y_out_list, axis=2))
                y_target = tf.stack(
                    [cv_target[idx1:idx2, 0, :], cv_target[idx1:idx2, 2, :]], axis=1
                )
                loss_y_prediction = self.loss_fn(
                    y_out_cv[:, :, :-1], y_target[:, :, 1:]
                ).numpy()  #
                # Compute Training Loss
                MSE_cv_loss_batch = self.loss_fn(
                    self.loss_w @ x_out_cv, self.loss_w @ cv_target[idx1:idx2, :, :]
                ).numpy()
                MSE_cv_linear_batch_list.append(MSE_cv_loss_batch)
                MSE_cv_pred_batch_list.append(loss_y_prediction)

            # Average
            MSE_cv_linear_epoch_list.append(
                tf.math.reduce_mean(tf.convert_to_tensor(MSE_cv_linear_batch_list))
            )
            MSE_cv_dB_epoch_list.append(10 * tf_log10(MSE_cv_linear_epoch_list[-1]))

            val_prediction_loss = tf.math.reduce_mean(
                tf.convert_to_tensor(MSE_cv_pred_batch_list)
            )

            if MSE_cv_dB_epoch_list[-1] < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = MSE_cv_dB_epoch_list[-1]
                self.MSE_cv_idx_opt = ti
                self.save()
                if self.config.wandb:
                    wandb.save("best_model.h5")

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Init Hidden State
            # self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0
            MSE_train_linear_batch_list = []
            MSE_train_pred_batch_list = []
            for j in range(0, self.N_E // self.N_B):  # batch
                idx1 = j * self.N_B
                idx2 = (j + 1) * self.N_B

                x_training = train_input[idx1:idx2, :, :]
                y_training = train_target[idx1:idx2, :, :]
                # init_seq = tf.constant([y_training[0, 0].numpy(), 0., y_training[1, 0].numpy(), 0.], shape=(4,1))
                self.model.InitSequence()  # self.ssModel.m1x_0 init_seq

                # Compute Training Loss
                LOSS, pred_loss = self.train_step(
                    x_training, y_training
                )  # exclude init sequence
                MSE_train_linear_batch_list.append(LOSS.numpy())
                MSE_train_pred_batch_list.append(pred_loss.numpy())
                # if self.config.wandb:
                #     wandb.log({"batch loss": LOSS})
                Batch_Optimizing_LOSS_sum += LOSS

            # Average
            MSE_train_linear_epoch_list.append(
                tf.math.reduce_mean(tf.convert_to_tensor(MSE_train_linear_batch_list))
            )  # torch.mean(MSE_train_linear_batch)
            MSE_train_dB_epoch_list.append(
                10 * tf_log10(MSE_train_linear_epoch_list[-1])
            )  # torch.log10(self.MSE_train_linear_epoch[ti])
            train_prediction_loss = tf.math.reduce_mean(
                tf.convert_to_tensor(MSE_train_pred_batch_list)
            )
            t = (time.time() - epoch_time) / 60

            ########################
            ### Training Summary ###
            ########################
            # print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]")
            print("-----------------------------------------------------------------")
            print(
                ti,
                " | MSE :",
                round(MSE_train_dB_epoch_list[ti].numpy(), 4),
                "[dB]",
                "MSE Validation :",
                round(MSE_cv_dB_epoch_list[ti].numpy(), 4),
                "[dB]",
            )
            if (
                np.isnan(MSE_train_dB_epoch_list[ti].numpy())
                or np.isnan(MSE_cv_dB_epoch_list[ti].numpy())
                or MSE_cv_dB_epoch_list[ti].numpy() == np.inf
                or MSE_cv_dB_epoch_list[ti].numpy() == np.inf
            ):
                print("Numerical instability..")
                # print(self.model.trainable_weights)
                return

            if ti > 1:
                d_train = MSE_train_dB_epoch_list[ti] - MSE_train_dB_epoch_list[ti - 1]
                d_cv = MSE_cv_dB_epoch_list[ti] - MSE_cv_dB_epoch_list[ti - 1]
                print(
                    "diff MSE Training :",
                    round(d_train.numpy(), 4),
                    "[dB]",
                    "diff MSE Val :",
                    round(d_cv.numpy(), 4),
                    "[dB]",
                )

            print(
                "Optimal idx:",
                self.MSE_cv_idx_opt,
                "Optimal :",
                round(self.MSE_cv_dB_opt.numpy(), 4),
                "[dB]  | Time : ",
                round(t, 2),
                "minutes ",
            )
            # print('----------------------------------------------------------')
            if self.config.wandb:
                # Logging
                wandb.log(
                    {
                        "train_loss": MSE_train_linear_epoch_list[-1].numpy(),
                        "train_loss_dB": MSE_train_dB_epoch_list[-1].numpy(),
                        "train_prediction_loss": train_prediction_loss.numpy(),
                        "val_loss": MSE_cv_linear_epoch_list[-1].numpy(),
                        "val_loss_dB": MSE_cv_dB_epoch_list[-1].numpy(),
                        "val_prediction_loss": val_prediction_loss.numpy(),
                    }
                )
        t = (time.time() - start_time) / 60
        print("Done training, elapsed time : ", round(t, 2), " minutes")

        self.MSE_cv_linear_epoch = tf.convert_to_tensor(MSE_cv_linear_epoch_list)
        self.MSE_cv_dB_epoch = tf.convert_to_tensor(MSE_cv_dB_epoch_list)
        self.MSE_train_linear_epoch = tf.convert_to_tensor(MSE_cv_linear_epoch_list)
        self.MSE_train_dB_epoch = tf.convert_to_tensor(MSE_cv_dB_epoch_list)

    def NNTest(self, n_Test, test_input, test_target):
        start = time.time()
        self.N_T = n_Test
        MSE_test_linear_arr_list = []
        MSE_test_pred_batch_list = []

        # MSE LOSS Function
        loss_fn = tf.keras.losses.MeanSquaredError()

        for j in range(0, self.N_T // self.N_B):
            idx1 = j * self.N_B
            idx2 = (j + 1) * self.N_B
            y_mdl_tst = test_input[idx1:idx2, :, :]

            self.model.InitSequence()  # self.ssModel.m1x_0)

            x_out_list = []
            y_out_list = []
            for t in range(self.ssModel.T):
                # Predict corrected state
                x_out_list.append(self.model(y_mdl_tst[:, :, t], training=False))
                # Predict next observation
                y_out_list.append(self.model.predict())
            x_out_test = tf.stack(x_out_list, axis=2)
            y_out_test = tf.squeeze(tf.stack(y_out_list, axis=2))
            y_target = tf.stack(
                [test_target[idx1:idx2, 0, :], test_target[idx1:idx2, 2, :]], axis=1
            )
            # Compute Training Loss
            loss_y_prediction = self.loss_fn(
                y_out_test[:, :, :-1], y_target[:, :, 1:]
            ).numpy()  #
            loss = self.loss_fn(
                self.loss_w @ x_out_test, self.loss_w @ test_target[idx1:idx2, :, :]
            ).numpy()
            MSE_test_linear_arr_list.append(loss)
            MSE_test_pred_batch_list.append(loss_y_prediction)

        end = time.time()
        t = end - start
        self.MSE_test_linear_arr = tf.convert_to_tensor(MSE_test_linear_arr_list)
        # Average
        self.MSE_test_linear_avg = tf.math.reduce_mean(
            (self.MSE_test_linear_arr)
        )  # torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * tf_log10(
            self.MSE_test_linear_avg
        )  # torch.log10(self.MSE_test_linear_avg)
        # Standard deviation
        self.MSE_test_dB_std = tf.math.reduce_std(
            self.MSE_test_linear_arr
        )  # torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * tf_log10(
            self.MSE_test_dB_std
        )  # torch.log10(self.MSE_test_dB_std)

        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg.numpy(), "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.MSE_test_dB_std.numpy(), "[dB]")
        # Print Run Time
        print("Inference Time:", round(t, 2), " seconds")

        return [
            self.MSE_test_linear_arr,
            self.MSE_test_linear_avg,
            self.MSE_test_dB_avg,
            x_out_test,
        ]


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
