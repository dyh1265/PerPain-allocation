from os.path import exists
import pandas as pd
import numpy as np
from models.GNN_TARnet_hyper_probabilistic import GNNTARnetProbHyper
from models.GAT_TARnet_hyper_probabilistic import GAT_TARnetHyper
from models.TARnet_hyper_probabilistic import TARnetHyperProb
from models.TLearner_hyper_probabilistic import TLearnerHyperProb
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
import os
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from utils.stats import find_means_and_std_bayesian
from utils.stats import compute_mse
from utils.plotting import print_outcomes
from utils.plotting import plot_proposed_treatments
from utils.stats import mean_confidence_interval
from utils.plotting import plot_outcomes    

def create_custom_gnn_graph(x_train, selected_cols):
    """
    Creates a custom graph structure for a Graph Neural Network (GNN) based on the input data 
    and selected columns.

    Args:
        x_train (pd.DataFrame): The input training data as a pandas DataFrame.
        selected_cols (dict): A dictionary where keys are group names (nodes influencing `y`) 
                              and values are lists of column names (features) associated with each group.

    Returns:
        tuple: A tuple containing:
            - edges (np.ndarray): A 2D numpy array representing the edges of the graph. 
                                  The first row contains source node indices, and the second row contains target node indices.
            - influence_y (list): A list of column indices corresponding to the group nodes influencing `y`.
            - x_train (pd.DataFrame): The modified training DataFrame with additional group nodes as columns.
    """
    quest_names = selected_cols.keys()
    cols = list(selected_cols.values())
    cols = [item for sublist in cols for item in sublist]
    quest_names = [name for name in quest_names]
    # id = x_train['ID']
    x_train = x_train[cols]

    # flatten the list of lists
    zeros_train = np.zeros((x_train.shape[0], len(quest_names)))

    grouped_features = pd.DataFrame(zeros_train, columns=quest_names)
    # create graph. Each group has an edge to a node influencing y]
    x_train = pd.concat([x_train, grouped_features], axis=1)

    nodes_from = []
    nodes_to = []
    grouped_features_names = quest_names
    for i in selected_cols.keys():
        for feature in selected_cols[i]:
            nodes_from.append(x_train.columns.get_loc(feature))
            nodes_to.append(x_train.columns.get_loc(i))

    edges = np.asarray([nodes_from, nodes_to])
    influence_y = [x_train.columns.get_loc(val) for val in grouped_features_names]
    # x_train['ID'] = id
    return edges, influence_y, x_train


def train_tarnet_prob(data_rct, t_new, y_eot_scaled, num):
    """
    Trains a TARNet model with hyperparameter tuning and returns the trained model.

    Args:
        data_rct (numpy.ndarray or pandas.DataFrame): The input dataset for training.
        t_new (numpy.ndarray or pandas.Series): The treatment assignment vector.
        y_eot_scaled (numpy.ndarray or pandas.Series): The scaled outcome variable.
        num (int): An identifier or parameter to be used in the TARNet model.

    Returns:
        object: The trained TARNet model instance.
    """
    params = {'dataset_name': "sum_new_latest_new", 'num': 15, 'lr': 1e-4, 'patience': 40, 'batch_size': 1,
              'num_train': data_rct.shape[0],
              'num_test': data_rct.shape[0], 'dropout_rate': 0.0,
              'reg_l2': .03, 'activation': 'softplus', 'epochs': 10, 'binary': False, 'verbose': 0,
              'tuner': kt.RandomSearch,
              'val_split': 0.0, 'kernel_init': 'GlorotNormal', 'max_trials': 10, 'defaults': False,
              'model_name': 'tarnet', 'alpha': 0.5,
              'n_fc': 4, 'hidden_phi': 240, 'n_fc_y0': 4, 'hidden_y0': 336, 'n_fc_y1': 3, 'n_fc_y2': 5,
              'tuner_name': 'random',
              'hidden_y1': 112, 'hidden_y2': 464, 'triplet': True}

    tarnet = TARnetHyperProb(params)
    tarnet.num = num
    tarnet.fit_tuner(data_rct, y_eot_scaled, t_new, 0)
    model = tarnet.fit_model(data_rct, y_eot_scaled, t_new, 0, 0)
    return model


def train_tlearner_prob(data_rct, t_new, y_eot_scaled, num=0):
    """
    Trains a T-Learner model with hyperparameter tuning and returns the trained model.

    Args:
        data_rct (pd.DataFrame): The input dataset containing the features for training.
        t_new (pd.Series or np.ndarray): The treatment assignment vector.
        y_eot_scaled (pd.Series or np.ndarray): The scaled outcome variable.
        num (int, optional): An identifier or index for the training instance. Defaults to 0.

    Returns:
        object: The trained T-Learner model instance.
    """
    params = {'dataset_name': "perpain_t_new_new_new_new", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 1,
              'defaults': False, 'alpha': 0.0,
              'dropout_rate': 0.0, 'normalize': False, 'reg_l2': .03,
              'activation': "softplus", 'hidden_y1': 60, 'hidden_y0': 124, 'eye': False,
              'json': True, 'tuner_name': 'random', 'tuner': kt.RandomSearch,
              'drop': None, 'model_name': 'tlearner', 'num_train': data_rct.shape[0],
              'num_test': 128, 'triplet': True, "n_hidden_2": 4, 'hidden_y2': 84,
              'epochs': 15, 'binary': False, 'n_hidden_1': 4, 'n_hidden_0': 6, 'notears': False,
              'verbose': 0, 'kernel_init': 'GlorotNormal', 'params': 'params_ihdp_a',
              'max_trials': 10}
    tlearner = TLearnerHyperProb(params)
    tlearner.num = num
    args = {'x': data_rct, 'y': y_eot_scaled, 't': t_new,
            'label': 'MPID_PS',
            'checkpoint_name': 'chkpt_tlearner', 'checkpoint': None}
    tlearner.fit_tuner(**args, seed=0)
    model = tlearner.fit_model(**args, seed=0)
    return model




def evaluate_models(label, seed=42, model_name='TARnet', test=True, num=0):
    """
    Evaluate different machine learning models for treatment effect estimation.
    Parameters:
    -----------
    label : str
        A label or identifier for the experiment.
    seed : int, optional, default=42
        Random seed for reproducibility.
    model_name : str, optional, default='TARnet'
        The name of the model to evaluate. Options include:
        - 'GNNTARnet'
        - 'GATTARnet'
        - 'TARnet'
        - 'TLearner'
        - Other models supported by the implementation.
    test : bool, optional, default=True
        If True, evaluates the model on a single test set. If False, performs
        evaluation across multiple datasets.
    num : int, optional, default=0
        Identifier for the dataset to use during evaluation.
    Returns:
    --------
    list
        If `test` is True, returns two lists:
        - mse_train: Mean squared error on the training set.
        - mse_test: Mean squared error on the test set.
        If `test` is False, returns two lists of MSE values for multiple datasets:
        - mse_trains: List of mean squared errors on training sets.
        - mse_tests: List of mean squared errors on test sets.
    Notes:
    ------
    - The function supports multiple models, including TARnet, TLearnern, GNNTARnet, and GATTARnet.
    - Data is preprocessed, including scaling and graph construction (for GNN-based models).
    - The function computes and prints MSE for both training and test datasets.
    - Additional plots are generated to visualize outcomes and proposed treatments.
    - Bayesian predictions are used to compute means and standard deviations for outcomes.
    """
    
    selected_cols = {
                    'mpid_PS': ["mpid_1_1", "mpid_1_7", "mpid_1_12"], 
                  }
    

    if test:
        # read data
        data_train = pd.read_csv(f'PERPAIN/PERPAIN_{num}/data_train_mpid.csv')
        data_test = pd.read_csv(f'PERPAIN/PERPAIN_{num}/data_test_mpid.csv')
        cols = list(selected_cols.values())
        cols = [item for sublist in cols for item in sublist]
        data_train = data_train[cols + ['y_baseline', 'y_eot', 't']]
        data_test = data_test[cols + ['y_baseline', 'y_eot', 't']]
        y_baseline_train = data_train['y_baseline'].values
        y_eot_train = data_train['y_eot'].values
        t_train = data_train['t'].values
        y_baseline_test = data_test['y_baseline'].values
        y_eot_test = data_test['y_eot'].values
        t_test = data_test['t'].values
        data_train.drop(['y_baseline', 'y_eot', 't'], axis=1, inplace=True)
        data_test.drop(['y_baseline', 'y_eot', 't'], axis=1, inplace=True)
        # expand dims
        y_eot_train = np.expand_dims(y_eot_train, axis=-1)
        t_train = np.expand_dims(t_train, axis=-1)


        if model_name == 'GNNTARnet':
            print(data_train.shape, data_test.shape)
            edges, influence_y, data_train = create_custom_gnn_graph(data_train, selected_cols)
            _, _, data_test = create_custom_gnn_graph(data_test, selected_cols)
            data_train = data_train.values
            data_test = data_test.values
            model = train_gnntarnet_prob(data_train, edges, influence_y, label, t_train, y_eot_train, 0)
            


            plot_outcomes(data_train, y_eot_train, y_baseline_train, t_train, model, model_name, mode='train')
            plot_outcomes(data_test, y_eot_test, y_baseline_test, t_test, model, model_name, mode='test')
            plot_proposed_treatments(data_train, y_eot_train, y_baseline_train, t_train, model, model_name, mode='train')
            plot_proposed_treatments(data_test, y_eot_test, y_baseline_test, t_test, model, model_name, mode='test')

        elif model_name == 'GATTARnet':
            print(data_train.shape, data_test.shape, '')
            edges, influence_y, data_train = create_custom_gnn_graph(data_train, selected_cols)
            _, _, data_test = create_custom_gnn_graph(data_test, selected_cols)
            data_train = data_train.values
            data_test = data_test.values
            model = train_gattarnet_prob(data_train, edges, influence_y, label, t_train, y_eot_train, 0)
            plot_outcomes(data_train, y_eot_train, y_baseline_train, t_train, model, model_name, mode='train')
            plot_outcomes(data_test, y_eot_test, y_baseline_test, t_test, model, model_name, mode='test')
            plot_proposed_treatments(data_train, y_eot_train, y_baseline_train, t_train, model, model_name,
                                     mode='train')
            plot_proposed_treatments(data_test, y_eot_test, y_baseline_test, t_test, model, model_name, mode='test')
        elif model_name == 'TARnet':
            data_train = data_train.values
            data_test = data_test.values
            scaler = StandardScaler()
            data_train = scaler.fit_transform(data_train)
            data_test = scaler.transform(data_test)
            model = train_tarnet_prob(data_train, t_train, y_eot_train, 0)
        else:
            scaler = StandardScaler()
            data_train = scaler.fit_transform(data_train)
            data_test = scaler.transform(data_test)
            y_scaler = StandardScaler()
            y_eot_scaled = y_scaler.fit_transform(y_eot_train)

            model = train_tlearner_prob(data_train, t_train, y_eot_scaled)
            plot_outcomes(data_train, y_eot_train, y_baseline_train, t_train, model, model_name, y_scaler, mode='train')
            plot_outcomes(data_test, y_eot_test, y_baseline_test, t_test, model, model_name, y_scaler, mode='test')
            plot_proposed_treatments(data_train, y_eot_train, y_baseline_train, t_train, model, model_name, y_scaler,
                                     mode='train')
            plot_proposed_treatments(data_test, y_eot_test, y_baseline_test, t_test, model, model_name, y_scaler, mode='test')



        (y0_pred_mean_train, y0_pred_std_train, y1_pred_mean_train,
        y1_pred_std_train, y2_pred_mean_train, y2_pred_std_train) = find_means_and_std_bayesian(data_train, model)

        (y0_pred_mean_test, y0_pred_std_test, y1_pred_mean_test,
         y1_pred_std_test, y2_pred_mean_test, y2_pred_std_test) = find_means_and_std_bayesian(data_test, model)

        mse_train = compute_mse(y0_pred_mean_train, y1_pred_mean_train, y2_pred_mean_train, y_baseline_train, y_eot_train, t_train, train=True)
        mse_test = compute_mse(y0_pred_mean_test, y1_pred_mean_test, y2_pred_mean_test, y_baseline_test, y_eot_test, t_test, train=False)
        print('mse train', mse_train)
        print('mse test', mse_test)

        y_effect_test = np.squeeze(y_eot_test)
        y_baseline_rct_test = np.squeeze(y_baseline_test)

        _ = print_outcomes(y0_pred_mean_test, y1_pred_mean_test, y2_pred_mean_test, y_effect_test,
                             y_baseline_rct_test, y0_pred_std_test, y1_pred_std_test, y2_pred_std_test,
                             t_test, model_name)


        return [mse_train], [mse_test]
    else:
        mse_trains = []
        mse_tests = []
        for i in range(1):
            data_train = pd.read_csv(f'PERPAIN/PERPAIN_{num}/data_train_' + str(i) + '_mpid.csv')
            data_test = pd.read_csv(f'PERPAIN/PERPAIN_{num}/data_test_' + str(i) + '_mpid.csv')
            cols = list(selected_cols.values())
            cols = [item for sublist in cols for item in sublist]
            data_train = data_train[cols + ['y_baseline', 'y_eot', 't']]
            data_test = data_test[cols + ['y_baseline', 'y_eot', 't']]
            y_baseline_train = data_train['y_baseline'].values
            y_eot_train = data_train['y_eot'].values
            t_train = data_train['t'].values
            y_baseline_test = data_test['y_baseline'].values
            y_eot_test = data_test['y_eot'].values
            t_test = data_test['t'].values
            data_train.drop(['y_baseline', 'y_eot', 't'], axis=1, inplace=True)
            data_test.drop(['y_baseline', 'y_eot', 't'], axis=1, inplace=True)
            # expand dims
            y_eot_train = np.expand_dims(y_eot_train, axis=-1)
            t_train = np.expand_dims(t_train, axis=-1)
            y_scaler = None
            if model_name == 'GNNTARnet':
                edges, influence_y, data_train = create_custom_gnn_graph(data_train, selected_cols)
                _, _, data_test = create_custom_gnn_graph(data_test, selected_cols)
                data_train = data_train.values
                data_test = data_test.values
                y_scaler = StandardScaler()
                y_eot_scaled = y_scaler.fit_transform(y_eot_train)
                model = train_gnntarnet_prob(data_train, edges, influence_y, label, t_train, y_eot_scaled, num=i)

            elif model_name == 'GATTARnet':
                edges, influence_y, data_train = create_custom_gnn_graph(data_train, selected_cols)
                _, _, data_test = create_custom_gnn_graph(data_test, selected_cols)
                data_train = data_train.values
                data_test = data_test.values
                y_scaler = StandardScaler()
                y_eot_scaled = y_scaler.fit_transform(y_eot_train)
                model = train_gattarnet_prob(data_train, edges, influence_y, label, t_train, y_eot_scaled, num=i)
            elif model_name == 'TARnet':
                # scale the data
                scaler = StandardScaler()
                data_train = scaler.fit_transform(data_train)
                data_test = scaler.transform(data_test)
                y_scaler = StandardScaler()
                y_eot_scaled = y_scaler.fit_transform(y_eot_train)
                model = train_tarnet_prob(data_train, t_train, y_eot_scaled, num=i)
            else:
                scaler = StandardScaler()
                data_train = scaler.fit_transform(data_train)
                data_test = scaler.transform(data_test)
                y_scaler = StandardScaler()
                y_eot_scaled = y_scaler.fit_transform(y_eot_train)
                model = train_tlearner_prob(data_train, t_train, y_eot_scaled, num=i)

            (y0_pred_mean_train, y0_pred_std_train, y1_pred_mean_train,
             y1_pred_std_train, y2_pred_mean_train, y2_pred_std_train) = find_means_and_std_bayesian(data_train, model, y_scaler)

            (y0_pred_mean_test, y0_pred_std_test, y1_pred_mean_test,
             y1_pred_std_test, y2_pred_mean_test, y2_pred_std_test) = find_means_and_std_bayesian(data_test, model, y_scaler)

            mse_train = compute_mse(y0_pred_mean_train, y1_pred_mean_train, y2_pred_mean_train, y_baseline_train, y_eot_train, t_train)
            mse_test = compute_mse(y0_pred_mean_test, y1_pred_mean_test, y2_pred_mean_test, y_baseline_test, y_eot_test, t_test)
            print(i, 'mse train', mse_train, 'mse test', mse_test)
            mse_trains.append(mse_train)
            mse_tests.append(mse_test)
        return mse_trains, mse_tests






def train_gnntarnet_prob(data_rct, edges, influence_y, label, t_new, y_effect_scaled, num):
    """
    Train a GNN TARNet model with probabilistic hyperparameter tuning.

    Args:
        data_rct (numpy.ndarray): The input data for the randomized controlled trial (RCT) with shape (num_samples, num_features).
        edges (tuple): A tuple representing the edges of the graph, where edges[1] contains the list of edges.
        influence_y (float): The influence parameter for the outcome variable.
        label (str): A label used for checkpoint naming and directory creation.
        t_new (numpy.ndarray): The treatment assignment vector.
        y_effect_scaled (numpy.ndarray): The scaled treatment effect outcomes.
        num (int): An identifier for the dataset or experiment.

    Returns:
        object: The trained GNN TARNet model.

    Notes:
        - The function initializes the GNN TARNet model with a set of predefined hyperparameters.
        - It uses a hyperparameter tuner (e.g., RandomSearch) to optimize the model.
        - Checkpoints are saved during training to allow for model recovery.
        - The function creates necessary directories for saving checkpoints if they do not exist.
    """
    params = {'dataset_name': f"perpain_select_{num}", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 1,
              'gnn_hidden_units': 40, 'gnn_n_fc': 5, 'aggregation_type': "sum", 'defaults': False,
              'combination_type': "add", 'dropout_rate': 0.0, 'normalize': False, 'reg_l2': .03,
              'activation': "softplus", 'hidden_y1': 68, 'hidden_y0': 60, 'eye': False,
              'json': True, 'tuner_name': 'random', 'tuner': kt.RandomSearch, 'alpha': 0.2,
              'drop': None, 'model_name': 'gnntarnet', 'num_train': data_rct.shape[0],
              'num_test': 128, 'triplet': True, "n_hidden_2": 4, 'hidden_y2': 64,
              'epochs': 100, 'binary': False, 'n_hidden_1': 5, 'n_hidden_0': 5, 'notears': False,
              'verbose': 1, 'kernel_init': 'GlorotNormal', 'params': 'params_ihdp_a',
              'max_trials': 10, 'influence_y': influence_y}


    gnntarnet = GNNTARnetProbHyper(params)
    gnntarnet.params['num_nodes'] = data_rct.shape[1]
    gnntarnet.params['num_edges'] = len(edges[1])
    gnntarnet.num = num
    weights = None
    # print(data_rct.shape)
    chekpoint_name = 'checkpoints_prob/' + params['dataset_name'] + label + '/checkpoint_effect_' + str(num) + '.h5'

    # Create directory if it doesn't exist
    if not os.path.exists('checkpoints_prob/' + label):
        os.makedirs('checkpoints_prob/' + label)
    #print("data rct shape", data_rct.shape)
    args = {'x': data_rct, 'y': y_effect_scaled, 't': t_new, 'edges': np.transpose(edges), 'weights': None,
            'label': label, 'checkpoint_name': chekpoint_name}
    gnntarnet.fit_tuner(**args, seed=0)
    # Create directory if it doesn't exist
    if not os.path.exists('checkpoints_prob/' + params['dataset_name'] + label):
        os.makedirs('checkpoints_prob/' + params['dataset_name'] + label)
    if exists(chekpoint_name):
        args['checkpoint'] = chekpoint_name
    else:
        args['checkpoint'] = None
    #args['checkpoint'] = None
    model = gnntarnet.fit_model(**args, count=0)
    return model

def train_gattarnet_prob(data_rct, edges, influence_y, label, t_new, y_effect_scaled, num):
    """
    Trains a GAT-TARnet model with hyperparameter tuning and returns the trained model.

    Args:
        data_rct (numpy.ndarray): Input data for the model, typically a 2D array where rows represent samples 
                                  and columns represent features.
        edges (tuple): A tuple representing the edges in the graph. Typically, edges[0] and edges[1] contain 
                       the source and destination nodes of the edges, respectively.
        influence_y (float): A parameter influencing the model's behavior, typically related to the target variable.
        label (str): A label or identifier for the training session, used for checkpoint naming and directory creation.
        t_new (numpy.ndarray): Treatment assignment data, typically a 1D array where each element represents 
                               the treatment for a corresponding sample.
        y_effect_scaled (numpy.ndarray): Scaled effect of the target variable, typically a 1D array.
        num (int): An identifier for the current training instance, used for checkpoint naming.

    Returns:
        object: The trained GAT-TARnet model instance.

    Notes:
        - The function initializes a GAT-TARnet model with specified hyperparameters.
        - It performs hyperparameter tuning using the Keras Tuner library.
        - Checkpoints are saved during training to allow resuming or evaluation of the model.
        - The function creates necessary directories for saving checkpoints if they do not exist.
    """
    # for test
    params = {'dataset_name': "perpain_select_t_new_new_95", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 1,
              'gnn_hidden_units': 40, 'gnn_n_fc': 5, 'aggregation_type': "sum", 'defaults': False,
              'combination_type': "add", 'dropout_rate': 0.0, 'normalize': False, 'reg_l2': .03,
              'activation': "softplus", 'hidden_y1': 68, 'hidden_y0': 60, 'eye': False,
              'json': True, 'tuner_name': 'random', 'tuner': kt.RandomSearch, 'alpha': 0.2,
              'drop': None, 'model_name': 'gattarnet', 'num_train': data_rct.shape[0],
              'num_test': 128, 'triplet': True, "n_hidden_2": 4, 'hidden_y2': 64,
              'epochs': 50, 'binary': False, 'n_hidden_1': 5, 'n_hidden_0': 5, 'notears': False,
              'verbose': 1, 'kernel_init': 'GlorotNormal', 'params': 'params_ihdp_a',
              'max_trials': 10, 'influence_y': influence_y}

    gattarnet = GAT_TARnetHyper(params)
    gattarnet.params['num_nodes'] = data_rct.shape[1]
    gattarnet.params['num_edges'] = len(edges[1])
    gattarnet.num = num
    weights = None
    # print(data_rct.shape)
    chekpoint_name = 'checkpoints_prob/' + params['dataset_name'] + label + '/checkpoint_effect_' + str(num) + '.h5'
    # Create directory if it doesn't exist
    if not os.path.exists('checkpoints_prob/' + label):
        os.makedirs('checkpoints_prob/' + label)
    #print("data rct shape", data_rct.shape)
    args = {'x': data_rct, 'y': y_effect_scaled, 't': t_new, 'edges': np.transpose(edges), 'weights': None,
             'label': label,
            'checkpoint_name': chekpoint_name}
    gattarnet.fit_tuner(**args, seed=0)
    # Create directory if it doesn't exist
    if not os.path.exists('checkpoints_prob/' + params['dataset_name'] + label):
        os.makedirs('checkpoints_prob/' + params['dataset_name'] + label)
    if exists(chekpoint_name):
        args['checkpoint'] = chekpoint_name
    else:
        args['checkpoint'] = None
    #args['checkpoint'] = None
    model = gattarnet.fit_model(**args, count=0)
    return model


def evaluate(num=0):
    """
    Evaluates the performance of specified models on a given dataset.

    Parameters:
    -----------
    num : int, optional
        An integer parameter that can be used to control the evaluation process. Default is 0.

    Description:
    ------------
    This function evaluates the performance of models specified in the `model_set` list. 
    It uses the `evaluate_models` function to compute the mean squared errors (MSEs) for 
    training and testing datasets. The results are printed along with their confidence intervals.

    The function operates in two modes:
    - Test mode (`test=True`): Evaluates the 'GNNTARnet' model and other models in the `model_set`.
    - Non-test mode (`test=False`): Evaluates the 'GNNTARnet' model and other models in the `model_set` 
      with a 200 splits configuration.

    The confidence intervals for the MSEs are calculated using the `mean_confidence_interval` function.

    Notes:
    ------
    - The `evaluate_models` and `mean_confidence_interval` functions are assumed to be defined elsewhere.
    - The `label` parameter is hardcoded to 'mpid_PS'.
    - The random seed is fixed at 42 for reproducibility.

    Returns:
    --------
    None
        This function does not return any value. It prints the evaluation results directly.
    """
    label = 'mpid_PS'
    model_set = ['TARnet', 'GNNTARnet', 'TLearner', 'GATTARnet']
    test = True
    if test:
        mses_train_gnn, mses_test_gnn = evaluate_models(label, seed=42, test=test, model_name=model_set[1], num=num)
        print('train GATTARnet', mean_confidence_interval(mses_train_gnn), 'pm'
              , 'test GATTARnet', mean_confidence_interval(mses_test_gnn))
    else:

        mses_train_tarnet, mses_test_tarnet = evaluate_models(label, seed=42, test=test, model_name=model_set[1], num=num)
        print('train gat', mean_confidence_interval(mses_train_tarnet), 'pm'
              , 'test gat', mean_confidence_interval(mses_test_tarnet))



    
if __name__ == "__main__":
    for i in range(1):
        evaluate(num=i)
