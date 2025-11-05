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
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
import os, sys
import tensorflow_probability as tfp
tf.get_logger().setLevel(logging.ERROR)
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy
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


class HyperGNNTarnetProb(kt.HyperModel, CausalModel):
    class HyperGNNTarnetProb:
        """
        A hypermodel class for building and training a probabilistic TARnet model with Graph Neural Networks (GNNs).
        This class integrates Keras Tuner's HyperModel and a custom CausalModel for hyperparameter tuning 
        and causal inference tasks.

        Attributes:
            params (dict): A dictionary of model parameters, including learning rate and batch size.
            name (str): The name of the model. Defaults to 'gnn_tarnet'.

        Methods:
            build(hp):
                Builds and compiles the GNN TARnet probabilistic model using the specified hyperparameters.
                
                Args:
                    hp (HyperParameters): A Keras Tuner HyperParameters object for managing hyperparameter tuning.

                Returns:
                    model (tf.keras.Model): A compiled TensorFlow Keras model.

            fit(hp, model, *args, **kwargs):
                Fits the model to the provided data using the specified hyperparameters and training arguments.
                
                Args:
                    hp (HyperParameters): A Keras Tuner HyperParameters object for managing hyperparameter tuning.
                    model (tf.keras.Model): The compiled model to be trained.
                    *args: Positional arguments for the `model.fit` method.
                    **kwargs: Keyword arguments for the `model.fit` method.

                Returns:
                    History: A Keras History object containing the training history.
        """
    def __init__(self, params, name='gnn_tarnet'):
        super().__init__()
        self.params = params
        self.name = name

    def build(self, hp):
        momentum = 0.9

        model = GNNTARnetProbModel(
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


class GraphConvLayer(layers.Layer):
    """
    GraphConvLayer is a custom TensorFlow layer that implements a graph convolutional layer. 
    It processes graph-structured data by performing three main steps: message preparation, 
    message aggregation, and node representation update.
    Attributes:
        params (dict): A dictionary of parameters for the layer, including aggregation type, 
            combination type, normalization flag, kernel initializer, and dropout rate.
        gnn_n_fc (int): Number of fully connected layers in the feedforward networks.
        gnn_hidden_units (int): Number of hidden units in the feedforward networks.
        ffn_prepare (FullyConnected): A feedforward network for preparing messages.
        update_fn (FullyConnected): A feedforward network for updating node representations.
    Methods:
        prepare(node_representations):
            Prepares messages for the nodes based on their representations.
            Args:
                node_representations (Tensor): Node representations of shape [num_edges, embedding_dim].
            Returns:
                Tensor: Prepared messages of shape [num_edges, embedding_dim].
        aggregate(node_indices, neighbour_messages, node_representations):
            Aggregates messages from neighboring nodes.
            Args:
                node_indices (Tensor): Indices of the nodes receiving messages, shape [num_edges].
                neighbour_messages (Tensor): Messages from neighboring nodes, shape [num_edges, representation_dim].
                node_representations (Tensor): Current node representations, shape [num_nodes, representation_dim].
            Returns:
                Tensor: Aggregated messages of shape [num_nodes, representation_dim].
        update(node_representations, aggregated_messages):
            Updates node representations based on aggregated messages.
            Args:
                node_representations (Tensor): Current node representations, shape [num_nodes, representation_dim].
                aggregated_messages (Tensor): Aggregated messages, shape [num_nodes, representation_dim].
            Returns:
                Tensor: Updated node embeddings of shape [num_nodes, representation_dim].
        call(inputs):
            Processes the inputs to produce updated node embeddings.
            Args:
                inputs (tuple): A tuple containing:
                    - node_representations (Tensor): Initial node representations, shape [num_nodes, representation_dim].
                    - edges (Tensor): Edge list, shape [num_edges, 2], where each row represents (source, target).
                    - edge_weights (Tensor): Weights of the edges, shape [num_edges].
            Returns:
                Tensor: Updated node embeddings of shape [num_nodes, representation_dim].
    """
    def __init__(
            self,
            params,
            gnn_n_fc,
            gnn_hidden_units,
            *args,
            **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.params = params
        self.aggregation_type = params['aggregation_type']
        self.combination_type = params['combination_type']
        self.normalize = params['normalize']
        self.gnn_n_fc = gnn_n_fc
        self.gnn_hidden_units = gnn_hidden_units

        self.ffn_prepare = FullyConnected(n_fc=self.gnn_n_fc, hidden_phi=self.gnn_hidden_units,
                                          final_activation='elu', out_size=self.gnn_hidden_units,
                                          kernel_init=self.params['kernel_init'], use_bias=False,
                                          kernel_reg=None, dropout=False, dropout_rate=self.params['dropout_rate'],
                                          name='ffn_prepare')

        self.update_fn = FullyConnected(n_fc=self.gnn_n_fc, hidden_phi=self.gnn_hidden_units,
                                        final_activation=None, out_size=self.gnn_hidden_units, use_bias=True,
                                        kernel_init=self.params['kernel_init'], batch_norm=False,
                                        kernel_reg=None, dropout=False, dropout_rate=self.params['dropout_rate'],
                                        name='update_fn')

    def prepare(self, node_representations):
        # node_representations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_representations)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_representations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_nodes = tf.shape(node_representations)[1]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(tf.transpose(neighbour_messages, [1, 0, 2]),
                                                              tf.cast(node_indices, tf.int32), num_segments=tf.cast(num_nodes, tf.int32))
            aggregated_message = tf.transpose(aggregated_message, [1, 0, 2])
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")
        return aggregated_message
    

    def update(self, node_representations, aggregated_messages):
        # node_representations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "concat":
            # Concatenate the node_representations and aggregated_messages.
            h = tf.concat([node_representations, aggregated_messages], axis=2)
        elif self.combination_type == "add":
            # Add node_representations and aggregated_messages.
            h = node_representations + aggregated_messages
        elif self.combination_type == "mlp":
            h = node_representations * aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.update_fn(h)
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_representations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """
        node_representations, edges, edge_weights = inputs
        # Get node_indices (source) and parent_indices (target) from edges.
        parent_indices, node_indices = edges[:, 0], edges[:, 1]
        parents_repesentations = tf.gather(node_representations, parent_indices, axis=1)
        # Prepare the messages of the parents.
        parent_messages = self.prepare(parents_repesentations)
        # Aggregate the parents messages.
        aggregated_messages = self.aggregate(node_indices, parent_messages, node_representations)
        return self.update(node_representations, aggregated_messages)


class Embedding(Model):
    """
    A TensorFlow Model subclass for creating an embedding layer with multiple 
    fully connected networks, one for each dimension of the input vector.

    Attributes:
        vector_size (int): The size of the input vector.
        num_neurons (int): The number of neurons in the output layer of each fully connected network.
        params (dict): A dictionary containing hyperparameters for the fully connected networks, 
                       such as 'dropout_rate' and 'kernel_init'.
        networks (list): A list of FullyConnected layers, one for each dimension of the input vector.

    Methods:
        call(inputs):
            Processes the input tensor through the embedding layer, applying each fully connected 
            network to the corresponding dimension of the input vector.

    Args:
        params (dict): A dictionary of hyperparameters for the fully connected networks.
            - 'dropout_rate' (float): The dropout rate to apply to the layers.
            - 'kernel_init' (str or tf.keras.initializers.Initializer): The kernel initializer for the layers.
        vector_size (int): The size of the input vector.
        num_neurons (int): The number of neurons in the output layer of each fully connected network.
    """
    def __init__(self, params, vector_size, num_neurons):
        super(Embedding, self).__init__()
        self.vector_size = vector_size
        self.num_neurons = num_neurons
        self.params = params
        self.networks = []
        for i in range(vector_size):
            x = FullyConnected(n_fc=1, hidden_phi=1,
                                      final_activation=None, out_size=self.num_neurons,
                                      dropout=False, dropout_rate=self.params['dropout_rate'],
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y'+str(i))
            self.networks.append(x)

    def call(self, inputs):
        outputs = []
        for i, layer in enumerate(self.networks):
            x_i = inputs[:, i]
            x = layer(x_i)
            outputs.append(x)
        outputs = tf.stack(outputs, axis=1)
        return outputs

class GNNTARnetProbModel(Model):
    """
    GNNTARnetProbModel is a probabilistic graph neural network model designed for TARnet-based architectures. 
    It leverages graph convolutional layers and fully connected layers to predict outcomes based on graph-structured data.

    Attributes:
        phi_0, phi_1, phi_2: Placeholder attributes for potential additional components.
        params (dict): Dictionary containing model parameters such as edges, weights, dropout rate, etc.
        model_name (str): Name of the model.
        edges: Graph edges used in the graph convolutional layers.
        gnn_weights: Weights for the graph neural network.
        gnn_n_fc (int): Number of fully connected layers in the graph convolutional layers.
        gnn_hidden_units (int): Number of hidden units in the graph convolutional layers.
        n_hidden_0, n_hidden_1, n_hidden_2 (int): Number of fully connected layers for the prediction heads.
        hidden_y0, hidden_y1, hidden_y2 (int): Number of hidden units in the prediction heads.
        phi: Placeholder for additional components.
        conv1, conv2: Graph convolutional layers.
        dropout (bool): Indicates whether dropout is applied.
        features_embedding: Embedding layer for node features.
        pred_y0, pred_y1, pred_y2: Fully connected layers for predicting outcomes.
        flatten: Flatten layer for reshaping the output.

    Methods:
        __init__(name, params, hp, *args, **kwargs):
            Initializes the GNNTARnetProbModel with the given parameters and hyperparameters.

        call(inputs, training=False):
            Forward pass of the model. Takes input data and returns predictions.
            Args:
                inputs: Input tensor containing node features.
                training (bool): Indicates whether the model is in training mode.
            Returns:
                Tensor containing concatenated predictions for y0, y1, and y2.
    """
    def __init__(
            self,
            name,
            params,
            hp,
            *args,
            **kwargs,
    ):
        super(GNNTARnetProbModel, self).__init__(*args, **kwargs)
        self.phi_0 = None
        self.phi_1 = None
        self.phi_2 = None
        self.params = params
        self.model_name = name
        self.edges = params['edges']
        self.gnn_weights = params['weights']
        self.gnn_n_fc = hp.Int('gnn_n_fc', min_value=2, max_value=10, step=1)
        self.gnn_hidden_units = hp.Int('gnn_hidden_units', min_value=2, max_value=128, step=4)
        self.n_hidden_0 = hp.Int('n_hidden_0', min_value=2, max_value=10, step=1)
        self.hidden_y0 = hp.Int('hidden_y0', min_value=4, max_value=128, step=4)
        self.n_hidden_1 = hp.Int('n_hidden_1', min_value=2, max_value=10, step=1)
        self.hidden_y1 = hp.Int('hidden_y1', min_value=4, max_value=128, step=4)
        if self.params['triplet']:
            self.n_hidden_2 = hp.Int('n_hidden_2', min_value=2, max_value=10, step=1)
            self.hidden_y2 = hp.Int('hidden_y2', min_value=4, max_value=128, step=4)
        self.phi = None
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            params=self.params,
            gnn_n_fc=self.gnn_n_fc,
            gnn_hidden_units=self.gnn_hidden_units,
            name="graph_conv1"
        )

        # # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            params=self.params,
            gnn_n_fc=self.gnn_n_fc,
            gnn_hidden_units=self.gnn_hidden_units,
            name="graph_conv2"
        )
        if self.params['dropout_rate'] > 0:
            self.dropout = True
        else:
            self.dropout = False
        self.features_embedding = tf.keras.layers.Embedding(input_dim=self.params['num_nodes'], output_dim=self.gnn_hidden_units)
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
        x = self.features_embedding(inputs)  # Shape: [num_samples, num_nodes, gnn_hidden_units]
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, None))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, None))
        x = x2 + x  # Shape: [num_samples, num_nodes, gnn_hidden_units]
        # Use info about nodes influencing the outcome
        x = tf.gather(x, self.params['influence_y'],
                      axis=1)  # Shape: [num_samples, num_influence_nodes, gnn_hidden_units]

        # Flatten
        x = self.flatten(x)  # Shape: [num_samples, num_influence_nodes * gnn_hidden_units]


        # Make a prediction
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        y2_pred = self.pred_y2(x)

        concat_pred = tf.concat([[y0_pred[:, 0], y0_pred[:, 1]],
                                 [y1_pred[:, 0], y1_pred[:, 1]],
                                 [y2_pred[:, 0], y2_pred[:, 1]]], axis=1)
        return concat_pred


class GNNTARnetProbHyper(CausalModel):
    """
    GNNTARnetProbHyper is a class that extends the CausalModel class and is designed for 
    hyperparameter tuning and training of a probabilistic graph neural network (GNN) model 
    for causal inference tasks. The class provides methods for hyperparameter tuning, model 
    training, graph loading, and evaluation.

    Attributes:
        params (dict): A dictionary containing model parameters and configurations.
        directory_name (str): Directory name for saving tuner results.
        project_name (str): Project name for the tuner.
        num (int): An optional attribute for tracking the model instance.

    Methods:
        __init__(params):
            Initializes the GNNTARnetProbHyper instance with the given parameters.

        fit_tuner(seed=0, **kwargs):
            Performs hyperparameter tuning using Keras Tuner. The method searches for the 
            best hyperparameters based on the specified objective function.

        fit_model(seed=0, count=0, **kwargs):
            Trains the model using the best hyperparameters obtained from the tuner. 
            Supports loading pre-trained weights from a checkpoint if provided.

        load_graph(path):
            Loads a graph from the specified path. Supports both JSON and CSV formats.

        get_graph_info(graph):
            Extracts graph information such as edges, edge weights, and influence values 
            from the loaded graph.

        load_graphs(x_train, count):
            Loads multiple graphs based on the dataset name and configuration. Supports 
            generating identity matrices for graphs if specified.

        evaluate(x_test, model):
            Evaluates the trained model on the test data and returns predictions.
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name = None
        self.num = None

    def fit_tuner(self, seed=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        edges = kwargs['edges']
        weights = kwargs['weights']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)

        directory_name = 'params_' + self.params['tuner_name'] + '/' + self.params['dataset_name']
        setSeed(seed)

        project_name = self.params["model_name"]

        self.params['edges'] = edges
        self.params['weights'] = weights
        self.params['num_edges'] = edges.shape[0]
        self.params['num_nodes'] = x.shape[1]

        hp = kt.HyperParameters()

        self.directory_name = directory_name
        self.project_name = project_name


        hypermodel = HyperGNNTarnetProb(params=self.params, name='gnn_tarnet_search')
        objective = kt.Objective("regression_probabilistic_loss", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='regression_probabilistic_loss', patience=5)]
        tuner.search(x, yt, epochs=30, validation_split=0.0, callbacks=[stop_early], verbose=self.params['verbose'])

        return

    def fit_model(self, seed=0, count=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        edges = kwargs['edges']
        weights = kwargs['weights']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)
        setSeed(seed)

        self.params['edges'] = edges
        self.params['weights'] = weights
        self.params['num_edges'] = edges.shape[0]

        tuner = self.params['tuner'](
            HyperGNNTarnetProb(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {'gnn_n_fc': self.params['gnn_n_fc'],
                                'gnn_hidden_units': self.params['gnn_hidden_units'],
                                'n_hidden_0': self.params['n_hidden_0'],
                               'hidden_y0': self.params['hidden_y0'],
                               'n_hidden_1': self.params['n_hidden_1'],
                               'hidden_y1': self.params['hidden_y1']}
            if self.params['triplet']:
                best_hps.values['n_hidden_2'] = self.params['n_hidden_2']
                best_hps.values['hidden_y2'] = self.params['hidden_y2']
        else:
            self.params['gnn_n_fc'] = best_hps.get('gnn_n_fc')
            self.params['gnn_hidden_units'] = best_hps.get('gnn_hidden_units')
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
            self.sparams = f""" gnn_n_fc = {best_hps.get('gnn_n_fc')} gnn_hidden_units = {best_hps.get('gnn_hidden_units')}
             n_hidden_0 = {best_hps.get('n_hidden_0')} n_hidden_1 = {best_hps.get('n_hidden_1')}
             hidden_y0 = {best_hps.get('hidden_y0')}  hidden_y1 = {best_hps.get('hidden_y1')}
         hidden_y2 = {best_hps.get('hidden_y2')}  n_hidden_y2 = {best_hps.get('n_hidden_2')}"""
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                              {self.sparams}""")
        return model

    def load_graph(self, path):
        if self.params['json']:
            with open(path) as f:
                graph = json.load(f)
        else:
            graph = pd.read_csv(path, header=None)
        return graph

    def get_graph_info(self, graph):
        if self.params['json']:
            edges = np.concatenate([np.asarray(graph['from']).reshape(-1, 1),
                                    np.asarray(graph['to']).reshape(-1, 1)], axis=1)
            influence_y = np.asarray(graph['influence_y'])
            edge_weights = np.asarray(graph['weights'])
            edge_weights = np.expand_dims(edge_weights / np.sum(edge_weights, axis=0), axis=-1)
            graph_info = {'edges': edges, 'edge_weights': edge_weights, 'influence_y': influence_y}
            return graph_info
        else:
            edges = np.asarray(graph)
            influence_y = []
            """Get non-zero elements from acyclic_W for edges and stack them to match the num of patients.
            Create an edges array (sparse adjacency matrix) of shape [num_samples, 2, num_edges]."""

            """Create an edge weights array of ones."""
            edge_weights = np.ones(shape=(edges.shape[0]))
            edge_weights = np.expand_dims(edge_weights / np.sum(edge_weights, axis=0), axis=-1)

            graph_info = {'edges': edges, 'edge_weights': edge_weights, 'influence_y': influence_y}
            return graph_info
    def load_graphs(self, x_train, count):

        if self.dataset_name == 'sum':
                path = 'graphs/sum_graph_' + str(self.params['num_layers'])
        else:
            path = 'graphs/' + self.params['dataset_name']

        if not self.params['json']:
            file_name = '/graph_' + str(count) + '.csv'
        else:
            file_name = '/graph_' + str(count) + '.json'

        graph = self.load_graph(path + file_name)
        graph_info = self.get_graph_info(graph)

        if self.params['eye']:
            acyclic_W = np.eye(x_train.shape[1])
            graph = np.asarray(np.nonzero(acyclic_W))
            edges = np.transpose(graph)
            graph_info['influence_y'] = np.arange(x_train.shape[1])
            graph_info['edges'] = edges

        return graph_info

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

