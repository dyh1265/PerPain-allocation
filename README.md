# PerPain Allocation

Patient allocation system for the PerPain project, providing multiple treatment assignment strategies based on patient characteristics and advanced machine learning models.

## Overview

This repository contains two complementary approaches for allocating patients to different treatments during the PerPain study:

1. **Clustering Allocation**: An R Shiny application that groups patients by similarity
2. **Individual Allocation**: Python-based machine learning models for personalized treatment recommendations

Choose the allocation strategy appropriate for your study design, sample size, and available patient features.

## Project Structure

```
PerPain-allocation/
├── Clustering Allocation/    # R Shiny app for clustering-based allocation
│   ├── app.R                 # Main Shiny application
│   ├── model.rds             # Pretrained XGBoost model
│   └── README.md             # Detailed instructions for R app
│
└── Individual Allocation/    # Python ML models for individual allocation
    ├── main_perpain.py       # Main training/evaluation script
    ├── models/               # Model implementations (TARnet, GNN-TARnet, etc.)
    ├── utils/                # Utility functions
    ├── checkpoints_prob/     # Pretrained model weights
    └── README.md             # Detailed instructions for Python models
```

## Allocation Methods

### 1. Clustering Allocation

**Use when**: You need a straightforward approach for grouping similar patients and want an interactive interface.

**Features**:
- Interactive R Shiny web application
- Uses pretrained XGBoost model for predictions
- Visual display of patient profiles
- Automatic package installation on first run
- Groups patients by similarity using clustering algorithms

**Quick Start**:
```bash
# Navigate to Clustering Allocation directory
cd "Clustering Allocation"

# Run the Shiny app
Rscript run.R
```

See [`Clustering Allocation/README.md`](./Clustering%20Allocation/README.md) for detailed instructions.

### 2. Individual Allocation

**Use when**: You need personalized treatment recommendations based on patient-level covariates and relational data.

**Features**:
- Advanced machine learning models:
  - **TARnet**: Treatment-Aware Representation Network for causal inference
  - **GNN-TARnet**: Graph Neural Network extension of TARnet
  - **GAT-TARnet**: Graph Attention Network extension of TARnet
  - **T-Learner**: Two-model approach for treatment effect estimation
- Probabilistic regression loss for robust predictions
- Hyperparameter tuning with Keras Tuner
- Graph-based modeling for structured relational data
- Pretrained model weights included

**Quick Start**:
```bash
# Navigate to Individual Allocation directory
cd "Individual Allocation"

# Run using Docker (recommended)
./docker_commands.sh

# Or run directly with Python 3.8+
python main_perpain.py
```

See [`Individual Allocation/README.md`](./Individual%20Allocation/README.md) for detailed instructions.

## Prerequisites

### For Clustering Allocation (R)
- R 3.6 or higher
- Required packages (auto-installed on first run):
  - Shiny
  - XGBoost
  - Additional dependencies

### For Individual Allocation (Python)
- Python 3.8 or higher
- TensorFlow 2.11
- Keras Tuner
- Docker (recommended)
- See `Individual Allocation/README.md` for full dependency list

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dyh1265/PerPain-allocation.git
   cd PerPain-allocation
   ```

2. Navigate to the specific allocation method directory and follow its README instructions.

## Usage

### Clustering Allocation Workflow
1. Launch the R Shiny application
2. Input patient characteristics
3. View clustering results and treatment recommendations
4. Explore patient profiles visually

### Individual Allocation Workflow
1. Prepare your dataset and place it in the appropriate directory
2. Run the main script to train/evaluate models
3. Modify evaluation parameters as needed
4. Review results, MSE metrics, and generated plots

## Data

**Note**: Patient data is not included in this repository due to privacy considerations. Users must provide their own datasets following the format expected by each allocation method.

## Results

Both allocation methods produce:
- Treatment assignment recommendations
- Evaluation metrics (accuracy, MSE, confidence intervals)
- Visual plots and reports saved to the project directory

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request with a clear description

## License

This project is licensed under the MIT License. See the `LICENSE` file in the Clustering Allocation directory for details.

## Citation

If you use this code in your research, please cite the PerPain project accordingly.

## Support

For questions or issues:
- Check the method-specific README files for detailed documentation
- Open an issue on the GitHub repository
- Contact the project maintainers

## Acknowledgments

This project was developed as part of the PerPain study for analyzing and predicting treatment outcomes using machine learning approaches.
