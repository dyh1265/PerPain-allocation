# Allocation Types

This project supports multiple patient-allocation strategies. Choose the type appropriate for your study design, sample size, and available patient features.

## 1. Clustering Allocation
- Groups patients by similarity using clustering algorithm.

## 2. Individual Allocation (GNN‑TARnet and others)
- Personalized allocation using Graph Neural Networks (GNN) combined with TARnet-style treatment-effect estimation.
- Produces individualized treatment recommendations by modeling patient relationships and heterogeneous treatment effects.
- Use when patient-level covariates and graph structure are informative and individualized assignments are desired.
- Data is not included.

For implementation details and examples, consult the documentation pages for each method in the repo.
