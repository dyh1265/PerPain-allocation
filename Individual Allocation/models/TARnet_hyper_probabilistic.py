import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from models.CausalModel import *
from utils.layers import FullyConnected
import keras_tuner as kt
from utils.callback import callbacks
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN,ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from os.path import exists
import shutil


class HyperTarnetProb(kt.HyperModel, CausalModel):
    class HyperTarnetProb:
        """
        A hypermodel class for building and training a probabilistic TARNet model 
        using Keras Tuner (kt) and a causal inference framework.

        This class extends `kt.HyperModel` and `CausalModel` to define a 
        hyperparameter-tunable TARNet model with probabilistic loss functions.

        Attributes:
            params (dict): A dictionary of model parameters, including learning rate 
                           (`lr`) and batch size (`batch_size`).

        Methods:
            build(hp):
                Builds and compiles the TARNet model with the specified hyperparameters.
                
                Args:
                    hp (kt.HyperParameters): A Keras Tuner HyperParameters object 
                                             for managing hyperparameter tuning.

                Returns:
                    keras.Model: A compiled Keras model instance.

            fit(hp, model, *args, **kwargs):
                Trains the TARNet model using the provided data and parameters.
                
                Args:
                    hp (kt.HyperParameters): A Keras Tuner HyperParameters object 
                                             for managing hyperparameter tuning.
                    model (keras.Model): The compiled Keras model to train.
                    *args: Positional arguments passed to the `model.fit` method.
                    **kwargs: Keyword arguments passed to the `model.fit` method.

                Returns:
                    keras.callbacks.History: A history object containing training 
                                             metrics and loss values.
        """
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = TarnetModelProb(name='tarnet', params=self.params, hp=hp)
        optimizer = SGD(learning_rate=self.params['lr'], momentum=0.9)
        model.compile(optimizer=optimizer, run_eagerly=False,
                      loss=self.regression_probabilistic_loss,
                      metrics=self.regression_probabilistic_loss)
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class TarnetModelProb(Model):
    """
    TarnetModelProb is a probabilistic model that uses fully connected layers to predict multiple outputs 
    (y0, y1, y2) based on the input features. The architecture is designed to allow hyperparameter tuning 
    for the number of fully connected layers and the size of hidden layers.

    Attributes:
        params (dict): A dictionary containing model parameters such as dropout rate, kernel initializer, 
                       activation function, and L2 regularization strength.
        n_fc (int): Number of fully connected layers in the shared feature extractor.
        hidden_phi (int): Size of the hidden layers in the shared feature extractor.
        fc (FullyConnected): Shared feature extractor network.
        n_fc_y0 (int): Number of fully connected layers in the y0 prediction network.
        hidden_y0 (int): Size of the hidden layers in the y0 prediction network.
        pred_y0 (FullyConnected): Fully connected network for predicting y0.
        n_fc_y1 (int): Number of fully connected layers in the y1 prediction network.
        hidden_y1 (int): Size of the hidden layers in the y1 prediction network.
        pred_y1 (FullyConnected): Fully connected network for predicting y1.
        n_fc_y2 (int): Number of fully connected layers in the y2 prediction network.
        hidden_y2 (int): Size of the hidden layers in the y2 prediction network.
        pred_y2 (FullyConnected): Fully connected network for predicting y2.

    Methods:
        call(inputs, training=True):
            Performs a forward pass through the model. The input is processed by the shared feature extractor 
            and then passed to the individual prediction networks for y0, y1, and y2. The predictions are 
            concatenated and returned as the output.

            Args:
                inputs (Tensor): Input tensor to the model.
                training (bool): Whether the model is in training mode.

            Returns:
                Tensor: Concatenated predictions for y0, y1, and y2.
    """
    def __init__(self, name, params, hp, **kwargs):
        super(TarnetModelProb, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=2, max_value=128, step=4)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.hidden_phi, final_activation='elu', dropout=True,
                                      dropout_rate=params['dropout_rate'],
                                 out_size=self.hidden_phi, kernel_init=params['kernel_init'], kernel_reg=None,
                                 name='fc')
        self.n_fc_y0 = hp.Int('n_fc_y0', min_value=2, max_value=10, step=1)
        self.hidden_y0 = hp.Int('hidden_y0', min_value=4, max_value=128, step=4)
        self.pred_y0 = FullyConnected(n_fc=self.n_fc_y0, hidden_phi=self.hidden_y0,
                                      final_activation=params['activation'], out_size=2,
                                      kernel_init=params['kernel_init'], dropout=True,
                                      dropout_rate=params['dropout_rate'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y0')

        self.n_fc_y1 = hp.Int('n_fc_y1', min_value=2, max_value=10, step=1)
        self.hidden_y1 = hp.Int('hidden_y1', min_value=4, max_value=128, step=4)
        self.pred_y1 = FullyConnected(n_fc=self.n_fc_y1, hidden_phi=self.hidden_y1,
                                      final_activation=params['activation'], out_size=2,
                                      kernel_init=params['kernel_init'], dropout=True,
                                      dropout_rate=params['dropout_rate'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y1')

        self.n_fc_y2 = hp.Int('n_fc_y2', min_value=2, max_value=10, step=1)
        self.hidden_y2 = hp.Int('hidden_y2', min_value=4, max_value=128, step=4)
        self.pred_y2 = FullyConnected(n_fc=self.n_fc_y2, hidden_phi=self.hidden_y2,
                                      final_activation=params['activation'], out_size=2,
                                      kernel_init=params['kernel_init'], dropout=True,
                                      dropout_rate=params['dropout_rate'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y2')

    def call(self, inputs, training=True):
        x = self.fc(inputs)
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        y2_pred = self.pred_y2(x)
        concat_pred = tf.concat([[y0_pred[:, 0], y0_pred[:, 1]],
                                 [y1_pred[:, 0], y1_pred[:, 1]],
                                 [y2_pred[:, 0], y2_pred[:, 1]]], axis=1)
        return concat_pred


class TARnetHyperProb(CausalModel):
    """
    TARnetHyperProb is a subclass of CausalModel designed to implement a hyperparameter-tuned TARNet 
    (Transformation-based Adversarial Representation Network) with probabilistic loss. This class 
    provides methods for hyperparameter tuning, model training, and evaluation.

    Attributes:
        params (dict): A dictionary containing model parameters and configurations.
        directory_name (str): Directory name for saving tuner results.
        project_name (str): Project name for the tuner.
        best_hps (kt.HyperParameters): Best hyperparameters obtained from the tuner.
        num (int): Counter to track the number of models trained.

    Methods:
        fit_tuner(x, y, t, seed):
            Tunes the hyperparameters of the TARNet model using Keras Tuner.
            
            Args:
                x (np.ndarray): Input features.
                y (np.ndarray): Outcome variable.
                t (np.ndarray): Treatment variable.
                seed (int): Random seed for reproducibility.

        fit_model(x, y, t, count, seed):
            Trains the TARNet model using the best hyperparameters obtained from the tuner.
            
            Args:
                x (np.ndarray): Input features.
                y (np.ndarray): Outcome variable.
                t (np.ndarray): Treatment variable.
                count (int): Counter for tracking training iterations.
                seed (int): Random seed for reproducibility.
            
            Returns:
                model (tf.keras.Model): Trained TARNet model.

        evaluate(x_test, model):
            Evaluates the trained TARNet model on test data.
            
            Args:
                x_test (np.ndarray): Test input features.
                model (tf.keras.Model): Trained TARNet model.
            
            Returns:
                np.ndarray: Predictions made by the model on the test data.
    """

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name = None
        self.best_hps = None
        self.num = 0

    def fit_tuner(self, x, y, t, seed):
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)
        directory_name = 'params_' + self.params['tuner_name'] + '/' + self.params['dataset_name']

        if self.dataset_name == 'gnn':
            directory_name = directory_name + f'/{self.params["model_name"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = self.params["model_name"]

        hp = kt.HyperParameters()

        self.directory_name = directory_name
        self.project_name = project_name

        hypermodel = HyperTarnetProb(params=self.params)
        objective = kt.Objective("regression_probabilistic_loss", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        yt = tf.concat([y, t], axis=1)
        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='regression_probabilistic_loss', patience=5)]
        tuner.search(x, yt, epochs=30, validation_split=0.0, callbacks=[stop_early], verbose=self.params['verbose'])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {'n_fc': self.params['n_fc'],
                                'hidden_phi': self.params['hidden_phi'],
                               'n_fc_y0': self.params['n_fc_y0'],
                               'hidden_y0': self.params['hidden_y0'],
                               'n_fc_y1': self.params['n_fc_y1'],
                               'hidden_y1': self.params['hidden_y1'],
                               'n_fc_y2': self.params['n_fc_y2'],
                               'hidden_y2': self.params['hidden_y2']
                               }
        self.best_hps = best_hps

        return

    def fit_model(self, x, y, t, count, seed):
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)

        tuner = kt.RandomSearch(
            HyperTarnetProb(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = self.best_hps
        model = tuner.hypermodel.build(best_hps)
        yt = tf.concat([y, t], axis=1)
        model.fit(x=x, y=yt,
                  callbacks=callbacks('regression_probabilistic_loss'),
                  validation_split=0.0,
                  epochs=self.params['epochs'],
                  batch_size=self.params['batch_size'],
                  verbose=self.params['verbose'])

        self.sparams = f"""n_fc={best_hps.get('n_fc')} hidden_phi = {best_hps.get('hidden_phi')}
              hidden_y1 = {best_hps.get('hidden_y1')} n_fc_y1 = {best_hps.get('n_fc_y1')}
              hidden_y2 = {best_hps.get('hidden_y2')} n_fc_y2 = {best_hps.get('n_fc_y2')}
              hidden_y0 = {best_hps.get('hidden_y0')}  n_fc_y0 = {best_hps.get('n_fc_y0')}"""


        # if count == 0 and self.folder_ind == 0:
        if self.num ==0:
            print(f"""The hyperparameter search is complete. The optimal hyperparameters are
                   {self.sparams}""")
            print(model.summary())

        return model

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)



