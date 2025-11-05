from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from scipy.special import expit
from utils.set_seed import *
import json
from os.path import exists
import shutil
import warnings, logging
import scipy.stats
from datetime import datetime
import keras_tuner as kt
from codecarbon import EmissionsTracker
from matplotlib import pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').disabled = True



class CausalModel:
    """
    CausalModel is a class designed to handle causal inference tasks, including training, evaluation, 
    and probabilistic regression loss computation. It also provides utilities for hyperparameter tuning 
    and tracking emissions during model training.

    Attributes:
        dataset_name (str): Name of the dataset being used.
        num (int): A parameter specifying the number of something (context-specific).
        params (dict): Dictionary containing various parameters for the model.
        binary (bool): Indicates if the model is binary.
        sparams (str): Placeholder for additional parameters (default is an empty string).
        emission_test (list): List to store emissions during testing.
        emission_train (list): List to store emissions during training.
        folder_ind (int or None): Index for folder identification (default is None).
        sum_size (int or None): Placeholder for sum size (default is None).
        num_train (int): Number of training samples.

    Methods:
        __init__(params):
            Initializes the CausalModel instance with the given parameters.

        setSeed(seed):
            Sets the random seed for reproducibility across various libraries and frameworks.

        train_and_evaluate(pehe_list_train, pehe_list_test, ate_list_train, ate_list_test, **kwargs):
            Placeholder method for training and evaluating the model.

        regression_probabilistic_loss(concat_true, concat_pred):
            Computes the generalized probabilistic regression loss for multiple treatments.

            Args:
                concat_true (tf.Tensor): Tensor of shape (batch_size, 2) containing factual outcomes 
                                         and treatment indices.
                concat_pred (list of tf.Tensor): List containing two tensors for predicted means 
                                                 and standard deviations.

            Returns:
                tf.Tensor: Mean loss over the batch.

        define_tuner(hypermodel, hp, objective, directory_name, project_name):
            Defines a hyperparameter tuner based on the specified tuner name in the parameters.

            Args:
                hypermodel: The hypermodel to be tuned.
                hp: Hyperparameters for the tuner.
                objective: Objective for the tuner to optimize.
                directory_name (str): Directory to save tuner results.
                project_name (str): Name of the tuning project.

            Returns:
                Tuner: Configured tuner instance.

        get_trackers(count):
            Sets up emissions trackers for monitoring and logging emissions during training and testing.

            Args:
                count (int): Counter to determine if existing files should be deleted.

            Returns:
                tuple: A tuple containing the test and train emissions trackers.
    """
    def __init__(self, params):
        self.dataset_name = params['dataset_name']
        self.num = params['num']
        self.params = params
        self.binary = params['binary']
        self.sparams = ""
        self.emission_test = list()
        self.emission_train = list()
        self.folder_ind = None
        self.sum_size = None
        self.num_train = params['num_train']

    @staticmethod
    def setSeed(seed):
        os.environ['PYTHONHASHSEED'] = '0'

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    def train_and_evaluate(self, pehe_list_train, pehe_list_test, ate_list_train, ate_list_test, **kwargs):
        pass



    def regression_probabilistic_loss(self, concat_true, concat_pred):
        """
        Generalized probabilistic regression loss for n_treatments.

        concat_true: (batch_size, 2)
            [:,0] = factual outcome
            [:,1] = treatment index (0..n_treatments-1)
        concat_pred: (2, batch_size * n_treatments)
            concat_pred[0] = flattened means
            concat_pred[1] = flattened stds
        """
        y_true = concat_true[:, 0]                           # (batch,)
        t_true = tf.cast(concat_true[:, 1], tf.int32)        # (batch,)

        batch_size = tf.shape(concat_true)[0]
        n_treatments = tf.shape(concat_pred)[1] // batch_size

        # Reshape to (batch, n_treatments)
        y_pred_mean = tf.reshape(concat_pred[0], (batch_size, n_treatments))
        y_pred_std  = tf.reshape(concat_pred[1], (batch_size, n_treatments))

        # Gather the prediction for the factual treatment
        pred_mean_sel = tf.gather(y_pred_mean, t_true, batch_dims=1)  # (batch,)
        pred_std_sel  = tf.gather(y_pred_std,  t_true, batch_dims=1)  # (batch,)

        # Define Normal distribution
        y_pred_dist = tfd.Normal(loc=pred_mean_sel, scale=pred_std_sel)

        # Loss: mix of NLL and MSE
        alpha = self.params.get("alpha", 1.0)
        loss = alpha * (-y_pred_dist.log_prob(y_true)) \
            + (1 - alpha) * tf.square(y_true - pred_mean_sel)

        # Mean over batch
        return tf.reduce_mean(loss)



    def define_tuner(self, hypermodel, hp, objective, directory_name, project_name):
        if self.params['tuner_name'] == 'hyperband':
            tuner = self.params['tuner'](
                hypermodel=hypermodel,
                objective=objective,
                directory=directory_name,
                max_epochs=50,
                tuner_id='2',
                overwrite=False,
                hyperparameters=hp,
                project_name=project_name,
                seed=0)
        else:
            tuner = self.params['tuner'](
                hypermodel=hypermodel,
                objective=objective,
                directory=directory_name,
                tuner_id='2',
                overwrite=False,
                hyperparameters=hp,
                project_name=project_name,
                max_trials=self.params['max_trials'],
                seed=0)
        return tuner

    def get_trackers(self, count):
        folder_path = './Emissions/' + self.params['model_name'] + '/'
        if self.params['defaults']:
            folder_path += 'default/'
        else:
            folder_path += self.params['tuner_name'] + '/'

        file_path_train = self.params['model_name'] + '_' + self.params['dataset_name'] + '_train.csv'
        file_path_test = self.params['model_name'] + '_' + self.params['dataset_name'] + '_test.csv'

        if self.params['dataset_name'] == 'gnn':
            file_path_train = self.params['model_name'] + '_' + self.params['dataset_name'] + '_train_' + str(
                self.folder_ind) + '.csv'
            file_path_test = self.params['model_name'] + '_' + self.params['dataset_name'] + '_test_' + str(
                self.folder_ind) + '.csv'

        file_exists_train = exists(folder_path + file_path_train)
        file_exists_test = exists(folder_path + file_path_test)
        folder_exists = exists(folder_path)

        if file_exists_train:
            # delete file
            if count == 0:
                os.remove(folder_path + file_path_train)

        if file_exists_test:
            # delete file
            if count == 0:
                os.remove(folder_path + file_path_test)
        # check if folder exists
        if not folder_exists:
            os.makedirs(folder_path)
        tracker_train = EmissionsTracker(project_name=self.params['model_name'], output_dir=folder_path,
                                         output_file=file_path_train, log_level="critical")
        tracker_test = EmissionsTracker(project_name=self.params['model_name'], output_dir=folder_path,
                                        output_file=file_path_test, log_level="critical")

        return tracker_test, tracker_train
