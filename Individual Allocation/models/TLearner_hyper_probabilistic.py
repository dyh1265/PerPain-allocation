import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from utils.set_seed import setSeed
import keras_tuner as kt
from utils.layers import FullyConnected
# from models.CausalModel import CausalModel
from models.CausalModel import *
import keras_tuner as kt
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping
import os, sys
import tensorflow_probability as tfp
tf.get_logger().setLevel(logging.ERROR)
import tensorflow_probability as tfp
tfd = tfp.distributions
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
import json
from os.path import exists

os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt

plt.show()
import json
from os.path import exists
import shutil


def callbacks(rlr_monitor):
    cbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(monitor=rlr_monitor, factor=0.5, patience=5, verbose=0, mode='auto',
                          min_delta=0., cooldown=0, min_lr=1e-8),
        EarlyStopping(monitor='val_regression_probabilistic_loss', patience=40, min_delta=0., restore_best_weights=True)
    ]
    return cbacks


class HyperTLearnerProb(kt.HyperModel, CausalModel):
    class HyperTLearnerProb:
        """
        A hypermodel implementation for a T-Learner with probabilistic regression capabilities.

        This class extends `kt.HyperModel` and `CausalModel` to provide a hyperparameter-tunable 
        T-Learner model for causal inference tasks. It allows for the customization of model 
        parameters and training configurations.

        Attributes:
            params (dict): A dictionary containing model parameters such as learning rate (`lr`) 
                           and batch size (`batch_size`).
            name (str): The name of the model. Defaults to 't-learner'.

        Methods:
            build(hp):
                Builds and compiles the T-Learner probabilistic model using the specified 
                hyperparameters and configurations.

                Args:
                    hp: A `kt.HyperParameters` object for managing hyperparameter tuning.

                Returns:
                    A compiled instance of `TLearnerProbModel`.

            fit(hp, model, *args, **kwargs):
                Fits the compiled model to the provided training data.

                Args:
                    hp: A `kt.HyperParameters` object for managing hyperparameter tuning.
                    model: The compiled `TLearnerProbModel` instance.
                    *args: Positional arguments to be passed to the `fit` method of the model.
                    **kwargs: Keyword arguments to be passed to the `fit` method of the model.

                Returns:
                    The history object resulting from the model's `fit` method.
        """
    def __init__(self, params, name='t-learner'):
        super().__init__()
        self.params = params
        self.name = name

    def build(self, hp):
        momentum = 0.9

        model = TLearnerProbModel(
            name=self.name,
            hp=hp,
            params=self.params,
        )

        model.compile(optimizer=SGD(learning_rate=self.params['lr'], nesterov=True, momentum=momentum),
                      loss=self.regression_probabilistic_loss,
                      metrics=self.regression_probabilistic_loss, run_eagerly=False
                      )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class TLearnerProbModel(Model):
    """
    TLearnerProbModel is a neural network model designed for probabilistic predictions 
    with support for hyperparameter tuning and optional triplet modeling.

    Attributes:
        params (dict): A dictionary containing model parameters such as activation 
            function, dropout rate, kernel initializer, and whether to use triplet modeling.
        model_name (str): The name of the model.
        n_hidden_0 (int): Number of fully connected layers for the first prediction head.
        hidden_y0 (int): Number of units in each hidden layer for the first prediction head.
        n_hidden_1 (int): Number of fully connected layers for the second prediction head.
        hidden_y1 (int): Number of units in each hidden layer for the second prediction head.
        n_hidden_2 (int, optional): Number of fully connected layers for the third prediction 
            head (used if triplet modeling is enabled).
        hidden_y2 (int, optional): Number of units in each hidden layer for the third prediction 
            head (used if triplet modeling is enabled).
        phi (None): Placeholder for additional functionality (currently unused).
        dropout (bool): Indicates whether dropout is applied based on the dropout rate.
        pred_y0 (FullyConnected): Fully connected network for the first prediction head.
        pred_y1 (FullyConnected): Fully connected network for the second prediction head.
        pred_y2 (FullyConnected, optional): Fully connected network for the third prediction 
            head (used if triplet modeling is enabled).
        flatten (layers.Flatten): Flatten layer for input processing.

    Methods:
        __init__(name, params, hp, *args, **kwargs):
            Initializes the TLearnerProbModel with the specified parameters and hyperparameters.

        call(inputs, training=False):
            Performs a forward pass through the model, generating predictions for each head 
            and concatenating the results.

            Args:
                inputs (Tensor): Input tensor to the model.
                training (bool): Flag indicating whether the model is in training mode.

            Returns:
                Tensor: Concatenated predictions from all heads.
    """
    def __init__(
            self,
            name,
            params,
            hp,
            *args,
            **kwargs,
    ):
        super(TLearnerProbModel, self).__init__(*args, **kwargs)
        self.params = params
        self.model_name = name
        self.n_hidden_0 = hp.Int('n_hidden_0', min_value=2, max_value=5, step=1)
        self.hidden_y0 = hp.Int('hidden_y0', min_value=4, max_value=128, step=4)
        self.n_hidden_1 = hp.Int('n_hidden_1', min_value=2, max_value=5, step=1)
        self.hidden_y1 = hp.Int('hidden_y1', min_value=4, max_value=128, step=4)
        if self.params['triplet']:
            self.n_hidden_2 = hp.Int('n_hidden_2', min_value=2, max_value=5, step=1)
            self.hidden_y2 = hp.Int('hidden_y2', min_value=4, max_value=128, step=4)
        self.phi = None

        if self.params['dropout_rate'] > 0:
            self.dropout = True
        else:
            self.dropout = False

        self.pred_y0 = FullyConnected(n_fc=self.n_hidden_0, hidden_phi=self.hidden_y0,
                                      final_activation=self.params['activation'], out_size=2,
                                      dropout=self.dropout, dropout_rate=self.params['dropout_rate'],
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y0')

        self.pred_y1 = FullyConnected(n_fc=self.n_hidden_1, hidden_phi=self.hidden_y1,
                                      final_activation=self.params['activation'], out_size=2,
                                      dropout=self.dropout, dropout_rate=self.params['dropout_rate'],
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y1')
        if self.params['triplet']:
            self.pred_y2 = FullyConnected(n_fc=self.n_hidden_2, hidden_phi=self.hidden_y2,
                                          final_activation=self.params['activation'], out_size=2,
                                          dropout=self.dropout, dropout_rate=self.params['dropout_rate'],
                                          kernel_init=self.params['kernel_init'],
                                          kernel_reg=regularizers.l2(.01), name='pred_y2')


        self.flatten = layers.Flatten()

    def call(self, inputs, training=False):
        x = inputs
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        y2_pred = self.pred_y2(x)
        # Concatenate the result and return
        concat_pred =  tf.concat([[y0_pred[:, 0], y0_pred[:, 1]],
                            [y1_pred[:, 0], y1_pred[:, 1]],
                            [y2_pred[:, 0], y2_pred[:, 1]]], axis=1)

        return concat_pred


class TLearnerHyperProb(CausalModel):
    """
    TLearnerHyperProb is a subclass of CausalModel designed for hyperparameter tuning and training 
    of probabilistic models in a causal inference setting. It provides methods for hyperparameter 
    search and model fitting, as well as evaluation of the trained model.

    Attributes:
        params (dict): A dictionary containing model and training parameters.
        directory_name (str): Directory name for saving tuner results.
        project_name (str): Project name for the tuner.
        num (int): Counter for tracking the number of models trained.

    Methods:
        __init__(params):
            Initializes the TLearnerHyperProb instance with the given parameters.

        fit_tuner(seed=0, **kwargs):
            Performs hyperparameter tuning using the Keras Tuner library.
            
            Args:
                seed (int): Random seed for reproducibility.
                **kwargs: Additional arguments, including:
                    - x: Input features.
                    - y: Target labels.
                    - t: Treatment indicators.

        fit_model(seed=0, count=0, **kwargs):
            Fits the model using the best hyperparameters obtained from the tuner.
            
            Args:
                seed (int): Random seed for reproducibility.
                count (int): Counter for tracking the number of models trained.
                **kwargs: Additional arguments, including:
                    - x: Input features.
                    - y: Target labels.
                    - t: Treatment indicators.
                    - checkpoint: Whether to load weights from a checkpoint.
                    - checkpoint_name: Name of the checkpoint file.

            Returns:
                model: The trained model.

        evaluate(x_test, model):
            Evaluates the trained model on test data.
            
            Args:
                x_test: Test input features.
                model: The trained model to be evaluated.

            Returns:
                Predictions made by the model on the test data.
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name = None
        self.num = 0

    def fit_tuner(self, seed=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)

        directory_name = 'params_' + self.params['tuner_name'] + '/' + self.params['dataset_name']
        setSeed(seed)

        project_name = self.params["model_name"]

        hp = kt.HyperParameters()

        self.directory_name = directory_name
        self.project_name = project_name

        hypermodel = HyperTLearnerProb(params=self.params, name='gnn_tarnet_search')
        objective = kt.Objective("regression_probabilistic_loss", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='regression_probabilistic_loss', patience=5)]
        tuner.search(x, yt, epochs=30, validation_split=0.0, callbacks=[stop_early], verbose=self.params['verbose'])

        return

    def fit_model(self, seed=0, count=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)
        setSeed(seed)

        tuner = self.params['tuner'](
            HyperTLearnerProb(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {
                                'n_hidden_0': self.params['n_hidden_0'],
                               'hidden_y0': self.params['hidden_y0'],
                               'n_hidden_1': self.params['n_hidden_1'],
                               'hidden_y1': self.params['hidden_y1']}
            if self.params['triplet']:
                best_hps.values['n_hidden_2'] = self.params['n_hidden_2']
                best_hps.values['hidden_y2'] = self.params['hidden_y2']
        else:
            self.params['n_hidden_0'] = best_hps.get('n_hidden_0')
            self.params['hidden_y0'] = best_hps.get('hidden_y0')
            self.params['n_hidden_1'] = best_hps.get('n_hidden_1')
            self.params['hidden_y1'] = best_hps.get('hidden_y1')
            if self.params['triplet']:
                self.params['n_hidden_2'] = best_hps.get('n_hidden_2')
                self.params['hidden_y2'] = best_hps.get('hidden_y2')

        model = tuner.hypermodel.build(best_hps)
        stop_early = [
            ReduceLROnPlateau(monitor='regression_probabilistic_loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=0., cooldown=0, min_lr=1e-8),
            EarlyStopping(monitor='regression_probabilistic_loss', patience=self.params['patience'], restore_best_weights=True)]
        # kwargs['checkpoint'] = None
        if kwargs['checkpoint'] is not None:
            # dummy_data = tf.zeros((x.shape[0], x.shape[1], x.shape[2]))  # replace `input_shape` with the shape of your input data
            dummy_data = tf.zeros((x.shape[0], x.shape[1])) # replace `input_shape` with the shape of your input data
            _ = model(dummy_data)
            model.load_weights(kwargs['checkpoint_name'])
        else:
            model.fit(x=x, y=yt,
                      validation_split=0.0,
                      callbacks=stop_early,
                      epochs=self.params['epochs'],
                      verbose=self.params['verbose'],
                      batch_size=self.params['batch_size'])
            # save the model
            model.save_weights(kwargs['checkpoint_name'])

        if self.num == 0:
            print(model.summary())
            self.sparams = f""" 
             n_hidden_0 = {best_hps.get('n_hidden_0')} n_hidden_1 = {best_hps.get('n_hidden_1')}
             hidden_y0 = {best_hps.get('hidden_y0')}  hidden_y1 = {best_hps.get('hidden_y1')}
         hidden_y2 = {best_hps.get('hidden_y2')}  n_hidden_y2 = {best_hps.get('n_hidden_2')}"""
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                              {self.sparams}""")
        return model

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)

