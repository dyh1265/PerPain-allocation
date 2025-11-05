# PERPAIN Project

## Overview
The PERPAIN project is designed to analyze and predict potential treatment outcomes using cutting-edge machine learning models. These models include TARnet, GNN-TARnet, GAT-TARnet, and T-Learner, which utilize probabilistic loss functions and hyperparameter tuning to deliver accurate predictions.

## Features
- Implementation of advanced models:
  - **TARnet**: Transformation-based Adversarial Representation Network for causal inference.
  - **GNN-TARnet**: Graph Neural Network extension of TARnet. Pretrained model weights are saved in the checkpoints_prob directory.
  - **GAT-TARnet**: Graph Attention Network extension of TARnet.
  - **T-Learner**: A two-model approach for estimating treatment effects, with separate models for treated and control groups.
- Probabilistic regression loss for robust and reliable predictions.
- Hyperparameter tuning with Keras Tuner for optimal model performance.
- Graph-based modeling for structured and relational data.

## Prerequisites
- Python 3.8 or higher
- TensorFlow 2.11
- Keras Tuner
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Additional dependencies listed in `requirements.txt`.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dyh1265/perpain.git
   cd perpain
   ```

2. Use docker for running the project:
   ```bash
   ./docker_commands.sh
   ```

## Usage
1. Prepare your dataset and place it in the `perpain_data/` directory.
2. Inside the Docker container run the main script to train and evaluate models:
   ```bash
   python main_perpain.py
   ```

3. Modify the `evaluate()` function in `main_perpain.py` to specify the model and evaluation type.

## Results
- The results, including Mean Squared Error (MSE) and confidence intervals, are printed to the console.
- Plots of the outcomes and proposed treatments are saved in the project directory.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
