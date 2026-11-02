# Next Basket Prediction (NBP)

This project implements a hybrid deep learning pipeline to predict a customer's next itemset based on their transaction history, capturing both spatial item co-occurrences and sequential temporal patterns.

## Dataset
We are using the **Ta Feng Grocery Dataset** (4 months of transaction logs).

## Environment Setup
1. Create a Conda environment:
   `conda create --name vitrualenv python=3.12`
2. Activate the environment:
   `conda activate vitrualenv`
3. Install dependencies:
   `pip install -r requirements.txt`
   - If (NVIDIA) GPU support is required:
      `pip install tensorflow[and-cuda]`

## Notebook details
- `notebooks/01_data_eda.ipynb`: Loads the Ta Feng data, maps the top 1,000 most frequent items, groups rare items into a single token, and constructs chronological sequence arrays for each user.
- `notebooks/02_RBM.ipynb`: Contains RBM class, loads the preprocessed data, trains a RBM model, extracts dense feature representations and packs them in a suitable format for comparison with LSTM model.
