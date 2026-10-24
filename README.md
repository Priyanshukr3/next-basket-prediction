# Next Basket Prediction (NBP)

This project implements a hybrid deep learning pipeline to predict a customer's next itemset based on their transaction history, capturing both spatial item co-occurrences and sequential temporal patterns.

## Dataset
We are using the **Ta Feng Grocery Dataset** (4 months of transaction logs).
* Download the dataset from Kaggle.
* Place the merged CSV file at: `data/raw/ta_feng_all_months_merged.csv`.

## Environment Setup
1. Create a Conda environment:
   `conda create --name vitrualenv python=3.12`
2. Activate the environment:
   `conda activate vitrualenv`
3. Install dependencies:
   `pip install -r requirements.txt`


* `notebooks/01_data_eda.ipynb`: Loads the Ta Feng data, maps the top 1,000 most frequent items, groups rare items into a single token, and constructs chronological sequence arrays for each user.