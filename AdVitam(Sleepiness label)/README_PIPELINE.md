# KSS Prediction Pipeline - AdVitam Dataset

This repository contains a complete pipeline for predicting Karolinska Sleepiness Scale (KSS) scores from physiological data using LSTM models. The pipeline includes preprocessing, feature extraction, model training, evaluation, and visualization.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Data Preprocessing](#data-preprocessing)
   - [Feature Extraction](#feature-extraction)
   - [Target Extraction](#target-extraction)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Configuration](#configuration)
6. [Examples](#examples)
7. [Output Structure](#output-structure)
8. [Project Structure](#project-structure)

## Quick Start

### Prerequisites

This project uses standard Python package management with `pip` and virtual environments.
If you prefer the faster [uv](https://github.com/astral-sh/uv) toolchain, see the
alternative commands below.

```bash
# Clone and setup
git clone https://github.com/strivn/tnivs-neuroadaptive-ml
cd tnivs-neuroadaptive-ml

# Create a virtual environment (pip way)
python -m venv .venv
pip install -r requirements.txt
source .venv/bin/activate

# --- Alternative: using uv (recommended) ---
# Install uv only once (if not available)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync      # installs deps and creates .venv automatically
source .venv/bin/activate
```

### Setup Data

```bash
# Setup data directories (honours DATA_ROOT from `.env`)
python -m scripts.setup_data_folder        # Create data directories

# Download dataset (~1.5 GB, will be stored under ${DATA_ROOT}/AdVitam)
python -m scripts.download_advitam         # Download AdVitam dataset
```

### Run Preprocessing

```bash
# Ensure you have configured a `.env` (or rely on the built-in defaults).
# Run the two preprocessing steps via the module interface so that
# `src` is correctly on the Python import path.

python -m src.preprocess.feature_extraction   # Extract features from BioPac files
python -m src.preprocess.target_extraction    # Generate KSS targets from questionnaire data
```

### Run the Pipeline

```bash
# Complete training pipeline with plots
python main.py --mode train,evaluate --config configs/baseline.yaml --plot
```

## Environment Setup

```bash
# Create a virtual environment (pip way)
python -m venv .venv
pip install -r requirements.txt
source .venv/bin/activate

# --- Alternative: using uv (recommended) ---
# Install uv only once (if not available)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync      # installs deps and creates .venv automatically
source .venv/bin/activate
```

## Data Preprocessing

### Feature Extraction

The feature extraction process converts raw BioPac `.acq` files into windowed feature matrices suitable for LSTM models.

#### Input Data Structure

```
data/AdVitam/Raw/Physio/BioPac/
├── 01_AC16.acq
├── 02_DU16.acq
├── 03_AC16.acq
└── ...
```

#### Running Feature Extraction

```bash
# Run feature extraction on all BioPac files
python src/preprocess/feature_extraction.py

# Run with custom parameters (not implemented yet)
python src/preprocess/feature_extraction.py --window_duration 30 --slide 10 --n_chunks 6
```

#### Parameters

- `window_duration`: Duration of each window in seconds (default: 30)
- `slide`: Sliding window step in seconds (default: 10)
- `n_chunks`: Number of chunks per participant (default: 6)

#### Output Structure

```
data/AdVitam/Preprocessed2/Feature/
├── 01_AC16_block_info.json
├── 01_AC16_feature_names.txt
├── 01_AC16_windowed_features_1.npy
├── 01_AC16_windowed_features_2.npy
└── ...
```

#### Extracted Features

- **EDA Features**: Filtered EDA, tonic EDA, SCR peaks frequency, amplitude
- **ECG Features**: Heart rate, HRV metrics (SDNN, RMSSD, pNN50, etc.)
- **Respiration Features**: Breathing rate, amplitude, variability

### Target Extraction

The target extraction process creates interpolated KSS scores for each chunk based on questionnaire data.

#### Input Data

```
data/AdVitam/Raw/Questionnaire/
└── export_data_french_ok.csv  # Contains KSS_B_1, KSS_1, KSS_B_2, KSS_2
```

#### Running Target Extraction

```bash
# Run target extraction
python src/preprocess/target_extraction.py
```

#### Output Structure

```
data/AdVitam/Preprocessed2/Target/
├── all_kss_targets.json
└── target_extraction_log.txt
```

#### Target Generation Process

1. **Load questionnaire data** with KSS baseline and end values
2. **Separate scenarios**: Scenario 1 (KSS_B_1 → KSS_1) and Scenario 2 (KSS_B_2 → KSS_2)
3. **Linear interpolation** between baseline and end KSS values
4. **Generate chunk targets** for each participant-scenario combination

## Model Training and Evaluation

### Basic Usage

```bash
# Complete pipeline (train + evaluate + predict)
python main.py --mode all --config configs/baseline.yaml

# Training and evaluation with plots
python main.py --mode train,evaluate --config configs/baseline.yaml --plot

# Prediction only (need future real-world data)
python main.py --mode predict --input data/test_features.npy
```

### Command Line Arguments

| Argument    | Description                                                     | Default                              |
| ----------- | --------------------------------------------------------------- | ------------------------------------ |
| `--mode`    | Pipeline modes: `train`, `evaluate`, `predict`, or combinations | `all`                                |
| `--config`  | Path to configuration file                                      | `configs/baseline.yaml`              |
| `--input`   | Input file for prediction mode                                  | None                                 |
| `--output`  | Output directory for results                                    | `results/experiment_YYYYMMDD_HHMMSS` |
| `--verbose` | Enable verbose logging                                          | False                                |
| `--plot`    | Generate plots after training/evaluation                        | False                                |
| `--seed`    | Random seed for reproducibility                                 | 42                                   |

## Configuration

Configuration files are in YAML format and control model parameters, data paths, and training settings.

### Example Configuration (`configs/baseline.yaml`)

```yaml
# Data configuration
data:
  features_dir: "data/AdVitam/Preprocessed2/Feature"
  labels_dir: "data/AdVitam/Preprocessed2/Target"
  labels_file: "all_kss_targets.json"
  scenarios: ["scenario1", "scenario2"]

# Model configuration
model:
  use_improved_model: false # Use basic LSTM model
  hidden_size: 128
  num_layers: 3
  dropout: 0.3
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 200
  patience: 10
```

## Examples

### Example 1: Complete Training Pipeline with Plots

```bash
python main.py --mode train,evaluate --config configs/baseline.yaml --plot
```

This command will:

1. Load preprocessed features and targets
2. Train an LSTM model with the specified configuration
3. Evaluate the model on the test set
4. Generate comprehensive plots including:
   - Training history (loss, MAE, R²)
   - Test set predictions vs actual values
   - Test data residual analysis
   - KSS distribution by dataset
   - Model performance metrics comparison

## Output Structure

After running the pipeline, you'll find the following structure in your output directory:

```
results/experiment_YYYYMMDD_HHMMSS/
├── model.pth                    # Trained model weights
├── scaler.pkl                   # Feature scaler for preprocessing
├── history.json                 # Training history
├── metrics.json                 # Final evaluation metrics
├── plots/                       # Generated plots
│   ├── training_history.png     # Training curves
│   ├── predictions_vs_actual.png # Test set predictions
│   ├── residual_analysis.png    # Test data residuals
│   ├── data_distribution.png    # KSS distribution
│   └── metrics_comparison.png   # Performance metrics
└── evaluation/                  # Detailed evaluation results
    └── evaluation_metrics.json  # Test set metrics
```

### Plot Descriptions

1. **Training History**: Shows training and validation loss, MAE, and R² over epochs
2. **Predictions vs Actual**: Scatter plot of predicted vs true KSS values (test set only)
3. **Residual Analysis**: Comprehensive residual analysis including:
   - Residuals vs predicted values
   - Residuals distribution histogram
   - Q-Q plot for normality
   - Residuals over time
4. **Data Distribution**: Histograms and box plots of KSS values for train/val/test sets
5. **Metrics Comparison**: Bar chart comparing MSE, MAE, RMSE, and R² across train/val/test sets

## Project Structure

```
tnivs-neuroadaptive-ml/
├── configs/                        # Configuration files
│   ├── baseline.yaml               # Baseline model configuration
│   └── arch_ablation.yaml          # Architecture ablation config
├── data/                           # Dataset files (gitignored)
│   └── AdVitam/                    # AdVitam dataset
│       ├── Raw/                    # Raw data files
│       └── Preprocessed2/          # Preprocessed data
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── initial/                    # Historical EDA notebooks
│   ├── model-initial-advitam.ipynb # Baseline model (preprocessed data)
│   ├── model-test-spectogram-advitam.ipynb # Spectrogram exploration
│   └── output/                     # Analysis results
├── references/                     # Research papers and documentation
├── research/                       # Research documentation and notes
├── results/                        # Experiment results (gitignored)
├── scripts/                        # Utility scripts
│   ├── setup_data_folder.py        # Setup data directory structure
│   └── download_advitam.py         # Download and organize AdVitam dataset
├── src/                            # Core source code
│   ├── data/                       # Data loading and preprocessing
│   │   ├── advitam_loader.py       # AdVitam dataset loader
│   │   └── feature_processor.py    # Feature processing utilities
│   ├── models/                     # Model implementations
│   │   ├── architectures.py        # LSTM model architectures
│   │   └── lstm.py                 # LSTM training pipeline
│   ├── preprocess/                 # Preprocessing scripts
│   │   ├── feature_extraction.py   # Feature extraction from BioPac files
│   │   └── target_extraction.py    # KSS target interpolation
│   └── utils/                      # Utility functions
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging utilities
│       └── plotting.py             # Plotting utilities
├── main.py                         # Main training pipeline
├── pyproject.toml                  # Project dependencies
├── README.md                       # Project overview
├── README_PIPELINE.md              # This file - detailed pipeline documentation
└── requirements.txt                # Python dependencies
```

## Data Flow

```
Raw BioPac Files (.acq)
         ↓
Feature Extraction (feature_extraction.py)
         ↓
Windowed Feature Matrices (.npy)
         ↓
Questionnaire Data (.csv)
         ↓
Target Extraction (target_extraction.py)
         ↓
Interpolated KSS Targets (.json)
         ↓
Main Pipeline (main.py)
         ↓
Trained Model + Results + Plots
```

## Naming Conventions

For files, put them into their respective folders as briefly described below.

For filenames:

```
Notebooks : [eda/model]-[dataset]-[task]
References: [dataset id/name]-[paper about]
```

Examples:

- `eda-advitam-sleep.ipynb`
- `advitam-meteier-classification.pdf`

### Data

⚠️ Data files shouldn't be committed to this repository due to size issues.

## Notes

- The pipeline uses **participant-based splitting** to avoid data leakage
- **Test set predictions** are used for all evaluation plots
- **Early stopping** is implemented to prevent overfitting
- **Missing value handling** uses rolling mean approach
- **Feature scaling** is applied using StandardScaler fitted on training data only
- **Reproducibility** is ensured with configurable random seeds
