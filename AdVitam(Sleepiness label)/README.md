# NIVS Capstone Project

Repository for the capstone project. 
See section Results 

## Quick Start (TL;DR)

I'm using `uv` for package manager. If you are not familiar of it, it's simply a replacement/alternative for `pip` but much faster.
See more: https://docs.astral.sh/uv/

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/strivn/tnivs-neuroadaptive-ml
cd tnivs
uv sync

# activate environment
source .venv/bin/activate

# set up files (paths follow DATA_ROOT in `.env`)
python -m scripts.setup_data_folder       # Create data directories
python -m scripts.download_advitam        # Download AdVitam dataset
```

## Project Summary

### Sleepiness Detection (AdVitam Dataset)

We developed three different approaches for sleepiness detection using the AdVitam dataset:

#### 1. Baseline Model (Preprocessed Data)
- **Key File**: `notebooks/model-initial-advitam.ipynb`
- **Approach**: Uses preprocessed dataset with baseline LSTM model
- **Description**: Initial implementation using readily available preprocessed features
- **How to Run**: Open and execute the Jupyter notebook directly

#### 2. Final Model (Custom Temporal Segmentation)
- **Key Files**: `main.py` and `README_PIPELINE.md`
- **Approach**: Custom temporal segmentation from raw dataset with advanced LSTM architecture
- **Description**: Production-ready pipeline with comprehensive preprocessing, training, and evaluation
- **How to Run**: `python main.py --mode train,evaluate --config configs/baseline.yaml --plot`
- **Documentation**: See `README_PIPELINE.md` for detailed usage instructions

#### 3. Spectrogram Exploration (Experimental)
- **Key File**: `notebooks/model-test-spectogram-advitam.ipynb`
- **Approach**: Exploratory dataset modification using spectrograms
- **Status**: Limited exploration due to initial results not being encouraging
- **How to Run**: Open and execute the Jupyter notebook directly

#### Historical Notebooks
Additional exploratory notebooks have been moved to `notebooks/initial/`. These may require slight adjustments to file paths when running.


## Codebase Standards 

### Specifically for Notebooks and Git

Additionally, I'm testing out using `nb-clean` and git filters to filter out any incoming commit due to notebooks that aren't substantive.

Run `nb-clean add-filter` in the environment.
Also run `nb-clean add-filter --preserve-cell-outputs` so outputs are saved.
See more: [https://github.com/srstevenson/nb-clean](https://github.com/srstevenson/nb-clean)

#### For Traditional Python Users

If you're used to `pip` and `venv`, here's the comparison:

| Traditional                       | uv equivalent           |
| --------------------------------- | ----------------------- |
| `pip install -r requirements.txt` | `uv sync`               |
| `pip install package`             | `uv add package`        |
| `pip freeze > requirements.txt`   | automatic via `uv.lock` |


### Naming Conventions

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

⚠️ Data files shouldn't be committed to this repository due to the size issues. Instead, put explanation in DATA_README.md on how we can download the dataset. See more in [DATA_README](DATA_README.md)

### Project Structure

```
tnivs-neuroadaptive-ml/
├── configs/                        # Configuration files
│   ├── baseline.yaml               # Baseline model configuration  
│   └── arch_ablation.yaml          # Architecture ablation config
├── data/                           # Dataset files (gitignored)
│   └── AdVitam/                    # AdVitam dataset structure
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── initial/                    # Historical EDA notebooks (may need path adjustments)
│   ├── model-initial-advitam.ipynb # Baseline model (preprocessed data)
│   ├── model-test-spectogram-advitam.ipynb # Spectrogram exploration
│   └── output/                     # Analysis results and exports
├── references/                     # Research papers and documentation
├── research/                       # Research documentation and notes
├── results/                        # Experiment results 
├── scripts/                        # Utility scripts
│   ├── setup_data_folder.py        # Setup data directory structure
│   └── download_advitam.py         # Download and organize AdVitam dataset
├── src/                            # Core source code
│   ├── data/                       # Data loading and preprocessing
│   ├── models/                     # LSTM model architectures
│   ├── preprocess/                 # Feature and target extraction
│   └── utils/                      # Utility functions
├── main.py                         # Main training pipeline (final model)
├── README.md                       # This file
└── README_PIPELINE.md              # Detailed pipeline documentation for AdVitam Final Modelling
```
