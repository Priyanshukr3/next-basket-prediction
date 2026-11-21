# Next Basket Prediction (NBP)

This project implements a hybrid deep learning pipeline to predict a customer's next itemset based on their transaction history, capturing both spatial item co-occurrences and sequential temporal patterns.

## Dataset
We are using the **Ta Feng Grocery Dataset** (4 months of transaction logs).

## Environment Setup
### Conda
1. Create a Conda environment:
   `conda create --name vitrualenv python=3.12`
2. Activate the environment:
   `conda activate vitrualenv`
3. Install dependencies:
   `pip install -r requirements.txt`
   - If (NVIDIA) GPU support is required:
      `pip install tensorflow[and-cuda]`

### Python
1. Create a virtual environment:
   `python venv .venv`
2. Activate the environment:
   **Linux/MacOS bash**
   `source .venv/bin/activate`
   **Windows PowerShell**
   `.venv\bin\Activate.ps1`
3. Install dependencies:
   `pip install -r requirements.txt`
   - If (NVIDIA) GPU support is required:
      `pip install tensorflow[and-cuda]`


## Notebook details
The notebooks form the part of two separate pipelines.
The notebook for each pipeline is in the format `0X_<NAME>.ipynb`, where `X `is the
step number for the pipeline, and `<NAME>` is a descriptive name for the step that the
notebook represents.

### Shared
- `notebooks/01_data_eda.ipynb`: Loads the Ta Feng data, maps the top 1,000 most frequent items, groups rare items into a single token, and constructs chronological sequence arrays for each user.

### RBM pipeline
- `notebooks/02_RBM.ipynb`: Loads the preprocessed data, trains the RBM, extracts dense hidden-layer features for each basket, and saves them as X_rbm_dense.npy for the LSTM.
- `src/RBM.py`: The RBM class and its training, feature extraction, and save/load functions.

### LSTM pipeline
- `notebooks/02_Data_loading.ipynb`: Builds the training, validation and test examples from each user’s sequence, checks that validation and test targets are not used for training, and saves the splits.
- `notebooks/03_LSTM_Architecture_Example.ipynb`: Trains the RBM-blended LSTM over a small hyperparameter grid, selects the best model on validation, saves it, and writes its test scores.
- `notebooks/03_raw-history-LSTM.ipynb`: The raw-history variant. It feeds the multi-hot history without the RARE column, with the same training setup, and compares it against the RBM-blended model.
- `src/LSTM.py`: The LSTM model class featuring one LSTM layer with dropout, a sigmoid output over the items, and binary cross-entropy loss.

### Baselines and evaluation
- `notebooks/03_Validation_Baseline.ipynb`: Checks the RBM input data, builds the Most Popular and FP-Growth baselines, and saves their scores.
- `notebooks/04_Error_Analysis.ipynb`: Compares the models by history length, checks how often RARE ranks first, and tests score cutoffs.

## Pytest details
Pytest tests for RBM metrics is provided under 2 files as below:
- `src/metrics.py`: Computes Hit-Ratio and Recall@K, ignores the RARE token, and compares models against a baseline.
- `tests/metrics.py`: Tests for the metrics functions.

