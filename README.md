# Next Basket Prediction (NBP)

This project implements a hybrid deep learning pipeline to predict a customer's next itemset based on their transaction history, capturing both intra-basket item co-occurrences and sequential temporal patterns.

## Dataset
This project uses the **Ta Feng Grocery Dataset** (4 months of transaction logs: 119,578 transactions, 32,266 users, and 23,812 unique items). The dataset is downloaded automatically via `kagglehub` when running `notebooks/01_data_eda.ipynb`.

## Quick Environment Setup (All-in-One)

### Option 1: Conda 
Run the following command from the project root to create the environment, install all dependencies, and register the Jupyter kernel at once:

```bash
conda create -n nbp-env python=3.12 -y && conda activate nbp-env && pip install -r requirements.txt && python -m ipykernel install --user --name=nbp-env --display-name "Python 3.12 (nbp-env)"
```

*(For NVIDIA GPU support on Linux/WSL2, replace `pip install -r requirements.txt` with `pip install -r requirements.txt && pip install "tensorflow[and-cuda]"`.)*

### Option 2: Python `venv`

**Linux / macOS / WSL2 (Bash):**
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python -m ipykernel install --user --name=nbp-venv --display-name "Python 3.12 (nbp-venv)"
```

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; python -m ipykernel install --user --name=nbp-venv --display-name "Python 3.12 (nbp-venv)"
```

---

## Running the Pipeline

Execute the notebooks in the following order:

1. `notebooks/01_data_eda.ipynb` — Downloads Ta Feng via `kagglehub`, maps the top 1,000 items deterministically, groups rare items into index `0`, and saves leak-free `X_raw.npy`, `X_padded.npy`, and `y_target.npy`.
2. `notebooks/02_Data_loading.ipynb` — Constructs chronological train/validation/test sliding-window splits and saves `splits.npz`.
3. `notebooks/02_RBM.ipynb` — Trains the RBM (`1001 -> 500`) on training baskets only and exports `X_rbm_dense.npy` and `rbm.weights.h5`.
4. `notebooks/03_Validation_Baseline.ipynb` — Runs data integrity diagnostics, builds the Most Popular and FP-Growth baselines, and saves baseline scores and reports.
5. `notebooks/03_LSTM_Architecture_Example.ipynb` — Runs hyperparameter search for the RBM + Masked LSTM (`WeightedBinaryCrossentropy` + `RepeatGate`), saves `lstm_best.keras`, and writes test scores.
6. `notebooks/03_raw-history-LSTM.ipynb` — Trains the fair Raw-History LSTM ablation (without RBM compression) and compares it against RBM + LSTM and baselines.
7. `notebooks/04_Error_Analysis.ipynb` — Evaluates overall metrics, history-length breakdowns, top-1 item diversity, and threshold curves.

---

## Source Modules & Tests

- `src/RBM.py`: Single-layer RBM class along with training, feature extraction, and weight save/load utilities.
- `src/LSTM.py`: Masked unidirectional LSTM with Layer Normalization, `WeightedBinaryCrossentropy`, and the `RepeatGate` output-layer residual branch.
- `src/metrics.py`: Computes HitRatio@K, Recall@K, and lift over baseline while excluding the `<RARE_ITEM>` token and rare-only targets.
- `tests/metrics.py`: Unit tests for the evaluation metrics.

### Running Unit Tests
```bash
python -m pytest tests/metrics.py
```

*(If Windows throws a `pytest11` plugin error, run `set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && python -m pytest tests/metrics.py`.)*

### Unified CLI Comparison
After generating all model score matrices in `data/predictions/`, print a combined comparison table with:

```bash
python -m src.metrics \
  --targets data/processed/y_target.npy \
  --scores MostPop=data/predictions/most_popular_scores.npy \
           FPGrowth=data/predictions/fpgrowth_scores.npy \
           RawLSTM=data/predictions/raw_lstm_test_scores.npy \
           RBMLSTM=data/predictions/lstm_test_scores.npy \
  --k 1 3 5 9 \
  --baseline MostPop \
  --output data/reports/final_comparison.json
```# Next Basket Prediction (NBP)

This project implements a hybrid deep learning pipeline to predict a customer's next itemset based on their transaction history, capturing both intra-basket item co-occurrences and sequential temporal patterns.

## Dataset
This project uses the **Ta Feng Grocery Dataset** (4 months of transaction logs: 119,578 transactions, 32,266 users, and 23,812 unique items). The dataset is downloaded automatically via `kagglehub` when running `notebooks/01_data_eda.ipynb`.

## Quick Environment Setup (All-in-One)

> **Requirements:** Python **3.12** is required for TensorFlow compatibility.  
> **NVIDIA GPU Note:** TensorFlow >= 2.11 requires **Linux or WSL2** for CUDA GPU acceleration (native Windows runs on CPU).

### Option 1: Conda (Recommended)
Run the following command from the project root to create the environment, install all dependencies, and register the Jupyter kernel at once:

```bash
conda create -n nbp-env python=3.12 -y && conda activate nbp-env && pip install -r requirements.txt && python -m ipykernel install --user --name=nbp-env --display-name "Python 3.12 (nbp-env)"
```

### Option 2: Python `venv`

**Linux / macOS / WSL2 (Bash):**
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python -m ipykernel install --user --name=nbp-venv --display-name "Python 3.12 (nbp-venv)"
```

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; python -m ipykernel install --user --name=nbp-venv --display-name "Python 3.12 (nbp-venv)"
```

**Windows (Command Prompt):**
```cmd
py -3.12 -m venv .venv && .venv\Scripts\activate.bat && pip install -r requirements.txt && python -m ipykernel install --user --name=nbp-venv --display-name "Python 3.12 (nbp-venv)"
```

When opening any notebook in VS Code or Jupyter, select the **Python 3.12 (nbp-env)** or **Python 3.12 (nbp-venv)** kernel.

---

## Running the Pipeline

Execute the notebooks in the following order:

1. `notebooks/01_data_eda.ipynb` — Downloads Ta Feng via `kagglehub`, maps the top 1,000 items deterministically, groups rare items into index `0`, and saves leak-free `X_raw.npy`, `X_padded.npy`, and `y_target.npy`.
2. `notebooks/02_Data_loading.ipynb` — Constructs chronological train/validation/test sliding-window splits and saves `splits.npz`.
3. `notebooks/02_RBM.ipynb` — Trains the RBM (`1001 -> 500`) on training baskets only and exports `X_rbm_dense.npy` and `rbm.weights.h5`.
4. `notebooks/03_Validation_Baseline.ipynb` — Runs data integrity diagnostics, builds the Most Popular and FP-Growth baselines, and saves baseline scores and reports.
5. `notebooks/03_LSTM_Architecture_Example.ipynb` — Runs hyperparameter search for the RBM + Masked LSTM (`WeightedBinaryCrossentropy` + `RepeatGate`), saves `lstm_best.keras`, and writes test scores.
6. `notebooks/03_raw-history-LSTM.ipynb` — Trains the fair Raw-History LSTM ablation (without RBM compression) and compares it against RBM + LSTM and baselines.
7. `notebooks/04_Error_Analysis.ipynb` — Evaluates overall metrics, history-length breakdowns, top-1 item diversity, and threshold curves.

---

## Source Modules & Tests

- `src/RBM.py`: Single-layer RBM class along with training, feature extraction, and weight save/load utilities.
- `src/LSTM.py`: Masked unidirectional LSTM with Layer Normalization, `WeightedBinaryCrossentropy`, and the `RepeatGate` output-layer residual branch.
- `src/metrics.py`: Computes HitRatio@K, Recall@K, and lift over baseline while excluding the `<RARE_ITEM>` token and rare-only targets.
- `tests/metrics.py`: Unit tests for the evaluation metrics.

### Running Unit Tests
```bash
python -m pytest tests/metrics.py
```

### Unified CLI Comparison
After generating all model score matrices in `data/predictions/`, print a combined comparison table with:

```bash
python -m src.metrics \
  --targets data/processed/y_target.npy \
  --scores MostPop=data/predictions/most_popular_scores.npy \
           FPGrowth=data/predictions/fpgrowth_scores.npy \
           RawLSTM=data/predictions/raw_lstm_test_scores.npy \
           RBMLSTM=data/predictions/lstm_test_scores.npy \
  --k 1 3 5 9 \
  --baseline MostPop \
  --output data/reports/final_comparison.json
```