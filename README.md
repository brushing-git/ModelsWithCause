# Reproducing Experiments for AI and Statistics

This repository contains the code necessary to reproduce experiments related to AI and Statistics. It includes scripts for running various experiments, training models, and generating datasets.

## Repository Structure

The repository is organized into the following main directories:

* `csr/`: Contains all core code for executing model operations and loading data.
    * `experiments/`: Scripts for running different experimental setups.
    * `models/`: Code for training various models (e.g., `decodertransformer_train.py`, `moetransformer_train.py`).
    * `notebooks/`: Jupyter notebooks for generating and processing datasets (e.g., `build_dataset.ipynb`).

Here is a visual representation of the file structure:

`

## Setup and Installation

To get started with this repository, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    You will need to install the necessary Python packages. A `requirements.txt` file is typically included in such projects. If one is not present, you may need to infer dependencies from the import statements in the scripts or create one based on your environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing, you'll need to create it or install packages like `torch`, `numpy`, `pandas`, `scikit-learn`, etc., as needed by the scripts.)*

## Usage

This section outlines how to use the different parts of the codebase to reproduce the experiments.

### 1. Generating Datasets

The datasets used in the experiments can be generated using the Jupyter notebooks in the `csr/notebooks/` directory.

* Navigate to the notebooks directory:
    ```bash
    cd csr/notebooks/
    ```
* Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
* Open and run the relevant notebooks (e.g., `build_dataset.ipynb`, `build_dataset_increasing.ipynb`) to generate the required data files.

### 2. Training Models

The `csr/models/` directory contains scripts for training various models.

* Navigate to the models directory:
    ```bash
    cd csr/models/
    ```
* Run the training scripts for the models you are interested in. For example, to train a decoder transformer:
    ```bash
    python decodertransformer_train.py --config_path ../configs/decodertransformer_config.json # Example config path
    ```
    *(Note: You may need to create or locate configuration files that specify model parameters, dataset paths, etc. These are typically found in a `configs` directory at the project root or within `csr`.)*

### 3. Running Experiments

The core experiments are located in the `csr/experiments/` directory. These scripts utilize the trained models and generated datasets to run specific experimental setups.

* Navigate to the experiments directory:
    ```bash
    cd csr/experiments/
    ```
* Execute the desired experiment script. For example:
    ```bash
    python baseline_experiments.py --model_path ../models/trained_model.pth --data_path ../data/dataset.pkl
    # or
    python exchangeability_experiments.py
    ```
    Each experiment script may have its own command-line arguments for specifying model paths, dataset locations, output directories, and experiment-specific parameters. Consult the individual script files for details on their arguments.

## Contributing

If you wish to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your features or bug fixes.
3.  Make your changes.
4.  Commit your changes with clear and descriptive messages.
5.  Push your branch and open a pull request.

## License

*(This section is a placeholder. Please choose and specify an appropriate open-source license, e.g., MIT, Apache 2.0, GPL.)*

This project is licensed under the \[MIT] - see the [LICENSE.md](LICENSE.md) file for details.
