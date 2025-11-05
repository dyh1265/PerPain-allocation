import numpy as np
from tensorflow.keras import Model
from keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
from utils.layers import FullyConnected
from models.CausalModel import *
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping
import json
from models.CausalModel import *
import tensorflow as tf

os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def callbacks(rlr_monitor):
    """
    Generates a list of Keras callbacks for training a model.

    Args:
        rlr_monitor (str): The metric name to monitor for the ReduceLROnPlateau callback.

    Returns:
        list: A list of Keras callbacks, including:
            - TerminateOnNaN: Stops training if a NaN loss is encountered.
            - ReduceLROnPlateau: Reduces the learning rate when the monitored metric has stopped improving.
            - EarlyStopping: Stops training early if the monitored metric does not improve after a specified patience.

    Notes:
        - The ReduceLROnPlateau callback reduces the learning rate by a factor of 0.5 if the monitored metric does not improve for 5 epochs.
        - The EarlyStopping callback monitors 'val_regression_probabilistic_loss' and stops training if it does not improve for 40 epochs.
    """
    cbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(monitor=rlr_monitor, factor=0.5, patience=5, verbose=0, mode='auto',
                          min_delta=0., cooldown=0, min_lr=1e-8),
        EarlyStopping(monitor='val_regression_probabilistic_loss', patience=40, min_delta=0., restore_best_weights=False)
    ]
    return cbacks


class GATGNNTarnet(kt.HyperModel, CausalModel):
    """
    GATGNNTarnet is a class that combines the functionality of a KerasTuner HyperModel 
    and a CausalModel to build and train a graph attention network (GAT) based TARnet model 
    with probabilistic regression capabilities.

    Attributes:
        params (dict): A dictionary of parameters used for configuring the model, 
            including learning rate (`lr`) and batch size (`batch_size`).
        name (str): The name of the model.

    Methods:
        build(hp):
            Constructs and compiles the GAT_TARnetModel with the specified hyperparameters.
            Args:
                hp (kt.HyperParameters): A KerasTuner HyperParameters object for tuning.
            Returns:
                model (tf.keras.Model): A compiled instance of the GAT_TARnetModel.

        fit(hp, model, *args, **kwargs):
            Trains the compiled model using the provided data and parameters.
            Args:
                hp (kt.HyperParameters): A KerasTuner HyperParameters object for tuning.
                model (tf.keras.Model): The compiled model to be trained.
                *args: Positional arguments passed to the `fit` method of the model.
                **kwargs: Keyword arguments passed to the `fit` method of the model.
            Returns:
                History: A history object containing training details.
    """
    def __init__(self, params, name):
        super().__init__()
        self.params = params

    def build(self, hp):
        momentum = 0.9

        model = GAT_TARnetModel(
            name="gnn_model",
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


class GraphAttention(layers.Layer):
    """
    A custom TensorFlow layer implementing graph attention mechanism.

    This layer performs graph attention by computing pairwise attention scores
    between nodes in a graph, normalizing these scores, and aggregating the
    node states based on the attention scores.

    Attributes:
        units (int): The dimensionality of the output space.
        params (dict): A dictionary of parameters, including:
            - 'kernel_init': Initializer for the kernel weights.
            - 'bin_count': Number of bins for repeating attention scores.
        kernel_initializer: Initializer for the kernel weights.
        kernel_regularizer: Regularizer for the kernel weights.
        bin_count (int): Number of bins for repeating attention scores.

    Methods:
        build(input_shape):
            Creates the layer's weights based on the input shape.
        call(inputs):
            Performs the forward pass of the layer.

    Args:
        units (int): The dimensionality of the output space.
        params (dict): A dictionary of parameters for the layer.
        **kwargs: Additional keyword arguments for the base layer.

    Inputs:
        inputs (tuple):
            - node_states (Tensor): A tensor of shape [batch_size, num_nodes, hidden_dim]
              representing the states of the nodes in the graph.
            - edges (Tensor): A tensor of shape [num_edges, 2] representing the edges
              in the graph, where each edge is defined by a pair of node indices.

    Outputs:
        Tensor: A tensor of shape [batch_size, num_nodes, hidden_dim] representing
        the updated node states after applying the graph attention mechanism.
    """
    def __init__(
            self,
            units,
            params,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.params = params
        self.kernel_initializer = self.params['kernel_init']
        self.kernel_regularizer = regularizers.l2(.01)
        self.bin_count = params['bin_count']

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.units, self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs
        # Linearly transform node states
        # node_states_transformed = [bs, num_nodes, hidden_dim]

        node_states_transformed = tf.matmul(node_states, self.kernel)
        # (1) Compute pair-wise attention scores
        # node_states_expanded = [bs, num_edges, 2, hidden_dim]
        node_states_expanded = tf.gather(node_states_transformed, edges, axis=1)
        # node_states_expanded = [bs, num_edges, 2 * hidden_dim]
        node_states_expanded = tf.reshape(
            node_states_expanded,
            (-1, tf.shape(edges)[0], node_states_expanded.shape[-2] * node_states_expanded.shape[-1])
        )
        # attention_scores = [bs, num_edges, 1]
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        # attention_score = [bs, num_edges]
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        # attention_score = [bs, num_edges]
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=tf.transpose(attention_scores, [1, 0]),
            segment_ids=edges[:, 1],
            num_segments=tf.reduce_max(edges[:, 1]) + 1,
        )

        # attention_scores_sum = [bs, num_nodes]
        attention_scores_sum = tf.transpose(attention_scores_sum, [1, 0])

        bin_count = self.bin_count
        attention_scores_sum = tf.map_fn(lambda i: tf.repeat(
            i, bin_count), attention_scores_sum)

        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of parents, apply attention scores and aggregate
        # node_states_parents = [bs, num_edges, hidden_dim]
        node_states_parents = tf.gather(node_states_transformed, edges[:, 0], axis=1)
        # node_states_parents = tf.gather(node_states_transformed, edges[:, 0], axis=1)

        node_times_attention = node_states_parents * attention_scores_norm[:, :, tf.newaxis]
        out = tf.math.unsorted_segment_sum(
            data=tf.transpose(node_times_attention, [1, 0, 2]),
            segment_ids=edges[:, 1],
            num_segments=tf.shape(node_states)[1],
        )

        out = tf.transpose(out, [1, 0, 2])
        return out


class MultiHeadGraphAttention(layers.Layer):
    class MultiHeadGraphAttention:
        """
        A multi-head graph attention layer for processing graph-structured data.

        This layer applies multiple graph attention mechanisms in parallel, combining
        their outputs either by concatenation or averaging. It is designed to handle
        graph data where nodes have features and edges are represented by pair indices.

        Attributes:
            num_heads (int): The number of attention heads to use.
            merge_type (str): The method to merge outputs from multiple heads. 
                              Options are "concat" (concatenation) or "mean" (averaging).
            attention_layers (list): A list of `GraphAttention` layers, one for each head.

        Args:
            units (int): The dimensionality of the output space for each attention head.
            params (dict): Additional parameters to configure the `GraphAttention` layers.
            num_heads (int, optional): The number of attention heads. Defaults to 8.
            merge_type (str, optional): The method to merge outputs from multiple heads. 
                                        Defaults to "mean".
            **kwargs: Additional keyword arguments for the parent `Layer` class.

        Methods:
            call(inputs):
                Computes the output of the multi-head graph attention layer.

                Args:
                    inputs (tuple): A tuple containing:
                        - atom_features (tf.Tensor): Node feature matrix of shape 
                          (num_nodes, feature_dim).
                        - pair_indices (tf.Tensor): Edge indices of shape (num_edges, 2).

                Returns:
                    tf.Tensor: The updated node feature matrix after applying multi-head 
                    graph attention, with shape depending on the `merge_type`:
                    - If "concat": (num_nodes, num_heads * units)
                    - If "mean": (num_nodes, units)
        """
    def __init__(self, units, params, num_heads=8, merge_type="mean", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units, params) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)


class Embedding(Model):
    """
    A TensorFlow Keras model that creates an embedding layer using a collection of 
    fully connected neural networks, one for each dimension of the input vector.

    Attributes:
        vector_size (int): The size of the input vector.
        num_neurons (int): The number of neurons in the output layer of each fully connected network.
        params (dict): A dictionary of parameters for configuring the fully connected networks.
        networks (list): A list of fully connected network layers, one for each dimension of the input vector.

    Methods:
        call(inputs):
            Processes the input tensor through the collection of fully connected networks 
            and returns the stacked outputs.

    Args:
        params (dict): A dictionary containing configuration parameters for the fully connected networks.
            Expected keys:
                - 'kernel_init': Kernel initializer for the layers.
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
                               kernel_init=self.params['kernel_init'],
                               kernel_reg=regularizers.l2(.01), name='pred_y' + str(i))
            self.networks.append(x)

    def call(self, inputs):
        outputs = []
        for i, layer in enumerate(self.networks):
            x_i = inputs[:, i]
            x = layer(x_i)
            outputs.append(x)
        outputs = tf.stack(outputs, axis=1)
        return outputs


class GAT_TARnetModel(Model):
    """
    GAT_TARnetModel is a graph attention network (GAT) model designed for TARnet-based hyperparameter 
    optimization with probabilistic modeling. It leverages multi-head graph attention layers and fully 
    connected layers to make predictions based on graph-structured data.

    Attributes:
        params (dict): A dictionary containing model parameters such as edges, dropout rate, activation 
            functions, kernel initializers, and other configurations.
        model_name (str): The name of the model.
        edges (list): The edges of the graph used in the attention layers.
        gnn_n_fc (int): Number of fully connected layers in the GNN.
        gnn_hidden_units (int): Number of hidden units in the GNN layers.
        hp_n_hidden_0 (int): Number of hidden layers for the first prediction head (y0).
        hp_hidden_y0 (int): Number of hidden units in each layer for the first prediction head (y0).
        hp_n_hidden_1 (int): Number of hidden layers for the second prediction head (y1).
        hp_hidden_y1 (int): Number of hidden units in each layer for the second prediction head (y1).
        n_hidden_2 (int, optional): Number of hidden layers for the third prediction head (y2), if triplet 
            mode is enabled.
        hidden_y2 (int, optional): Number of hidden units in each layer for the third prediction head (y2), 
            if triplet mode is enabled.
        num_heads (int): Number of attention heads in the multi-head graph attention layers.
        num_layers (int): Number of graph attention layers.
        dropout (bool): Whether to apply dropout based on the dropout rate.
        attention_layers (list): A list of multi-head graph attention layers.
        pred_y0 (FullyConnected): Fully connected layer for the first prediction head (y0).
        pred_y1 (FullyConnected): Fully connected layer for the second prediction head (y1).
        pred_y2 (FullyConnected, optional): Fully connected layer for the third prediction head (y2), if 
            triplet mode is enabled.
        flatten (tf.keras.layers.Flatten): A flattening layer to reshape the output.
        features_embedding (tf.keras.layers.Embedding): An embedding layer for node features.

    Methods:
        __init__(name, params, hp, *args, **kwargs):
            Initializes the GAT_TARnetModel with the given parameters and hyperparameters.

        call(inputs, training=False):
            Performs a forward pass through the model. Embeds the input features, applies graph attention 
            layers, and makes predictions using the fully connected layers for y0, y1, and optionally y2.

            Args:
                inputs (tf.Tensor): Input tensor representing node indices.
                training (bool): Whether the model is in training mode.

            Returns:
                tf.Tensor: Concatenated predictions for y0, y1, and y2.
    """
    def __init__(
            self,
            name,
            params,
            hp,
            *args,
            **kwargs,
    ):
        super(GAT_TARnetModel, self).__init__(*args, **kwargs)
        self.params = params
        self.model_name = name
        self.edges = params['edges']
        # self.gnn_weights = params['weights']

        self.gnn_n_fc = hp.Int('gnn_n_fc', min_value=2, max_value=10, step=1)
        self.gnn_hidden_units = hp.Int('gnn_hidden_units', min_value=2, max_value=128, step=4)
        self.hp_n_hidden_0 = hp.Int('n_hidden_0', min_value=2, max_value=10, step=1)
        self.hp_hidden_y0 = hp.Int('hidden_y0', min_value=4, max_value=128, step=4)
        self.hp_n_hidden_1 = hp.Int('n_hidden_1', min_value=2, max_value=10, step=1)
        self.hp_hidden_y1 = hp.Int('hidden_y1', min_value=4, max_value=128, step=4)
        if self.params['triplet']:
            self.n_hidden_2 = hp.Int('n_hidden_2', min_value=2, max_value=10, step=1)
            self.hidden_y2 = hp.Int('hidden_y2', min_value=4, max_value=128, step=4)

        self.params['gnn_hidden_units'] = self.gnn_hidden_units
        self.params['gnn_n_fc'] = self.gnn_n_fc
        self.num_heads = 2  # 1
        self.num_layers = 2  # 2

        if self.params['dropout_rate'] > 0:
            self.dropout = True
        else:
            self.dropout = False

        self.attention_layers = [
            MultiHeadGraphAttention(units=self.gnn_hidden_units, num_heads=self.num_heads, params=self.params) for _ in
            range(self.num_layers)
        ]
        self.pred_y0 = FullyConnected(n_fc=self.hp_n_hidden_0, hidden_phi=self.hp_hidden_y0,
                                      final_activation=self.params['activation'], out_size=2,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='y0')

        self.pred_y1 = FullyConnected(n_fc=self.hp_n_hidden_1, hidden_phi=self.hp_hidden_y1,
                                      final_activation=self.params['activation'], out_size=2,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='y1')

        if self.params['triplet']:
            self.pred_y2 = FullyConnected(n_fc=self.n_hidden_2, hidden_phi=self.hidden_y2,
                                          final_activation=self.params['activation'], out_size=2,
                                          dropout=self.dropout, dropout_rate=self.params['dropout_rate'],
                                          kernel_init=self.params['kernel_init'],
                                          kernel_reg=regularizers.l2(.01), name='pred_y2')


        self.flatten = layers.Flatten()
        self.features_embedding = tf.keras.layers.Embedding(input_dim=self.params['num_nodes'], output_dim=self.gnn_hidden_units)


    def call(self, inputs, training=False):
        x = self.features_embedding(inputs)
        # Apply the first graph conv layer.
        for attention_layer in self.attention_layers:
            x = attention_layer([x, self.edges]) + x
        # Flatten
        x = tf.gather(x, self.params['influence_y'], axis=1)
        x = self.flatten(x)
        # Make a prediction
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        y2_pred = self.pred_y2(x)

        concat_pred = tf.concat([[y0_pred[:, 0], y0_pred[:, 1]],
                                 [y1_pred[:, 0], y1_pred[:, 1]],
                                 [y2_pred[:, 0], y2_pred[:, 1]]], axis=1)
        return concat_pred


class GAT_TARnetHyper(CausalModel):
    """
    GAT_TARnetHyper is a class that implements a causal model using Graph Attention Networks (GAT) 
    and TARnet architecture with hyperparameter tuning and probabilistic loss.

    Attributes:
        params (dict): A dictionary containing model parameters.

    Methods:
        __init__(params):
            Initializes the GAT_TARnetHyper instance with the given parameters.

        fit_tuner(seed=0, **kwargs):
            Performs hyperparameter tuning using Keras Tuner.
            Args:
                seed (int): Random seed for reproducibility.
                **kwargs: Additional arguments including 'x', 'y', 't', 'edges', and 'weights'.

        fit_model(seed=0, count=0, **kwargs):
            Fits the model using the best hyperparameters obtained from the tuner.
            Args:
                seed (int): Random seed for reproducibility.
                count (int): Counter for tracking the number of fits.
                **kwargs: Additional arguments including 'x', 'y', 't', 'edges', 'weights', and 'checkpoint'.

        load_graph(path):
            Loads a graph from a specified path.
            Args:
                path (str): Path to the graph file.
            Returns:
                dict or DataFrame: The loaded graph.

        get_graph_info(graph):
            Extracts graph information such as edges, edge weights, and influence values.
            Args:
                graph (dict or DataFrame): The graph data.
            Returns:
                dict: A dictionary containing edges, edge weights, and influence values.

        load_graphs(x_train, count):
            Loads multiple graphs based on the dataset and configuration.
            Args:
                x_train (ndarray): Training data.
                count (int): Graph index.
            Returns:
                dict: A dictionary containing graph information.

        evaluate(x_test, model):
            Evaluates the model on test data.
            Args:
                x_test (ndarray): Test data.
                model (tf.keras.Model): The trained model.
            Returns:
                ndarray: Model predictions.
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params

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
        bin_count = np.bincount(tf.cast(edges[:, 1], "int32"))
        # bin_count = np.bincount(tf.cast(edges[:, 0], "int32"))

        self.params['bin_count'] = bin_count

        hypermodel = GATGNNTarnet(params=self.params, name='gat_tarnet_search')
        objective = kt.Objective("regression_probabilistic_loss", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='regression_probabilistic_loss', patience=5)]
        tuner.search(x, yt, epochs=30, validation_split=0.0, callbacks=[stop_early], verbose=self.params['verbose'])
        return

    def fit_model(self, seed=0, count=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        count=0
        edges = kwargs['edges']
        weights = kwargs['weights']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)
        setSeed(seed)
        bin_count = np.bincount(tf.cast(edges[:, 1], "int32"))
        self.params['bin_count'] = bin_count
        self.params['edges'] = edges
        self.params['weights'] = weights
        self.params['num_edges'] = edges.shape[0]

        tuner = self.params['tuner'](
            GATGNNTarnet(params=self.params, name='gat_tarnet_search'),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = tuner.get_best_hyperparameters()[0]
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
            dummy_data = tf.zeros((x.shape[0], x.shape[1]))  # replace `input_shape` with the shape of your input data
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

        if count == 0:
            print(model.summary())
            self.sparams = f""" 
                            n_hidden_0 = {best_hps.get('n_hidden_0')} 
                            n_hidden_1 = {best_hps.get('n_hidden_1')} 
                            n_hidden_2 = {best_hps.get('n_hidden_2')}
                            hidden_y0 = {best_hps.get('hidden_y0')}  
                            hidden_y1 = {best_hps.get('hidden_y1')}
                            hidden_y2 = {best_hps.get('hidden_y2')} """
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



