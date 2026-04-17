# Models with a Cause

This repository contains the code necessary to reproduce the experiments from:

> **Rushing, B., & Gomez-Lavin, J. (2026).** *Models with a Cause: Causal Discovery with Language Models on Temporally Ordered Text Data.* Transactions on Machine Learning Research (TMLR). [OpenReview](https://openreview.net/forum?id=YJddclPGuY)

The paper investigates whether language models possess the inductive biases necessary to identify causal structures in token generation processes. It evaluates four architectures — NADE, Encoder-Decoder Transformer, Decoder-only Transformer, and Switch Transformer — on synthetic mixtures of Markov chains to test whether they learn the conditional independencies and Markov exchangeability properties required for causal discovery (see Sections 4–5 of the paper).

## Repository Structure

The repository has been reorganized to follow a more standard Python project layout. The previous `csr/` folder has been replaced by a top-level `src/` package, with training scripts, experiments, figures, and notebooks promoted to siblings of `src/`.

```
.
├── src/                              # Core library code
│   ├── __init__.py
│   ├── data/                         # Dataset generation utilities
│   │   ├── __init__.py
│   │   ├── datasets.py               # PyTorch Dataset wrappers
│   │   ├── generate_perms.py         # Markov-exchangeable permutation search (Sec. 5.3)
│   │   ├── higher_markov_chain.py    # Second-order Markov generators (Sec. 5.4)
│   │   ├── intervention.py           # Interventional data generation
│   │   └── markov_chain.py           # First-order Markov generators (Sec. 5.1)
│   ├── nets/                         # Model architectures
│   │   ├── __init__.py
│   │   ├── models.py                 # NADE, Encoder-Decoder, Decoder-only transformers
│   │   ├── moe.py                    # Switch (mixture-of-experts) transformer
│   │   └── utils.py
│   ├── search/                       # Hyperparameter search
│   │   ├── __init__.py
│   │   └── gridsearch.py             # Grid search with k-fold CV (App. C)
│   └── tests/                        # Statistical test implementations
│       ├── __init__.py
│       └── independence_tests.py     # χ² conditional-independence tests (Sec. 5.2)
├── train/                            # Training entry points
│   ├── train_nade.py
│   ├── train_transformer.py
│   ├── train_decodertransformer.py
│   ├── train_moetransformer.py
│   ├── train_higher_transformer.py
│   ├── train_increasing_transformer.py
│   ├── train_increasing_decodertransformer.py
│   └── train_increasing_moetransformer.py
├── experiments/                      # Evaluation scripts
│   ├── baseline_experiments.py
│   ├── exchangeability_experiments.py
│   ├── exchangeability_experiments_higher.py
│   ├── exchangeability_experiments_notrain.py
│   ├── exchangeability_experiments_short.py
│   ├── gridsearch_experiments.py
│   ├── independence_experiments.py
│   ├── intervention_experiments_outcomes.py
│   ├── intervention_experiments_probabilities.py
│   ├── statistical_tests.py
│   ├── statistical_tests_higher.py
│   └── statistical_tests_intervention.py
├── figures/                          # Generated figures (Figs. 1–2 of the paper)
└── notebooks/                        # Jupyter notebooks for dataset construction
```

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/brushing-git/ModelsWithCause.git
    cd ModelsWithCause
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Core dependencies include `torch`, `numpy`, `pandas`, `scipy`, and `scikit-learn`. Jupyter is required to run the dataset construction notebooks.

## Usage

The pipeline follows three stages: (1) generate synthetic datasets, (2) train models, and (3) run the evaluation experiments from the paper.

### 1. Generating Datasets

Each dataset is a mixture of Markov chains (see Figure 1 and Section 5.1 of the paper). A sequence is produced by first sampling a starting token from a prior distribution, which selects a stochastic transition matrix, and then sampling successive tokens from that matrix. Dataset sizes are chosen to match Chinchilla-optimal token counts for each model's parameter count (Hoffmann et al., 2022).

Generate the datasets via the notebooks in `notebooks/`:

```bash
cd notebooks/
jupyter lab
```

The notebooks construct first-order and second-order Markov chain mixtures at sequence lengths of 6, 100, and 500 tokens, as well as the densely-sampled length-6 dataset used in Section 5.5.

### 2. Training Models

Training entry points live in `train/`, one per architecture. The paper evaluates:

- **NADE** — Neural Autoregressive Distribution Estimator (Uria et al., 2016)
- **Encoder-Decoder Transformer** (Vaswani, 2017)
- **Decoder-only Transformer** (Radford, 2018)
- **Switch Transformer** — sparsely-gated mixture-of-experts (Fedus et al., 2022)

For example, to train the decoder-only transformer:

```bash
python train/train_decodertransformer.py
```

The `train_increasing_*` scripts train each architecture across varying dataset sizes for the data-scaling experiment in Figure 2f of the paper. `train_higher_transformer.py` trains on second-order Markov mixtures (Section 5.4, Appendix E).

All models were trained with Adam — cosine annealing for transformers and a step scheduler for NADE — and evaluated on an 80/20 train/validation split. Selected hyperparameters are listed in Table 6 of the paper; to reproduce the search itself, see the next section.

### 3. Running Experiments

The `experiments/` directory contains the scripts behind each table and figure in the paper.

**Conditional independence (Section 5.2):**
```bash
python experiments/baseline_experiments.py          # χ² baseline on raw sequences (Table 1)
python experiments/independence_experiments.py      # JSD tests on trained models (Table 2a)
```

**Markov exchangeability and symmetry (Section 5.3):**
```bash
python experiments/exchangeability_experiments.py           # Main results (Fig. 2a–b)
python experiments/exchangeability_experiments_notrain.py   # Untrained baselines (Fig. 2c–d)
python experiments/exchangeability_experiments_short.py     # Length-6 sequences
python experiments/exchangeability_experiments_higher.py    # Second-order Markov (App. E)
```

**Distribution approximation and qualitative rankings (Section 5.5):**
```bash
python experiments/statistical_tests.py                 # Log-probability gap (Fig. 2e, Table 2b)
python experiments/statistical_tests_higher.py          # Higher-order variant
python experiments/statistical_tests_intervention.py    # Interventional comparisons
python experiments/intervention_experiments_outcomes.py
python experiments/intervention_experiments_probabilities.py
```

**Hyperparameter search (Appendix C):**
```bash
python experiments/gridsearch_experiments.py
```

Most scripts expose command-line arguments for model paths, dataset locations, and output directories; run any script with `--help` for details. Because results in the paper are averaged across five random seeds, reproducing the exact confidence intervals requires running each experiment over multiple seeds.

## Citation

If you use this code, please cite:

```bibtex
@article{rushing2026models,
  title   = {Models with a Cause: Causal Discovery with Language Models on Temporally Ordered Text Data},
  author  = {Rushing, Bruce and Gomez-Lavin, Javier},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=YJddclPGuY}
}
```

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Commit with clear, descriptive messages.
5. Push your branch and open a pull request.
