# Stress Detection using DriveDB Dataset

This folder contains the end-to-end pipeline for stress detection using the [DriveDB dataset](https://physionet.org/content/drivedb/1.0.0/) from PhysioNet. The project is part of the Neuroadaptive In-Vehicle Safety (NIVS) Capstone, which aims to detect driver stress in real-world driving conditions using physiological signals such as ECG, GSR, EMG, and respiration.

---

##  Project Summary

**Goal:** Detect driver stress using real physiological signals collected in naturalistic driving scenarios.  
**Dataset:** DriveDB – includes 18 driving sessions (25–93 minutes) across 8 subjects.  
**Sensors Used:**
- ECG (Heart activity)
- GSR (Skin conductance, hand & foot)
- EMG (Muscle tension)
- RESP (Breathing rate)
- Marker Events (Manual stress events)

---

##  Approaches

### 1. Baseline MLP Model (Feature-Based)
- Extracts handcrafted features (mean, std, RMS, peak-to-peak, zero-crossings)
- Applies subject-wise heuristic labeling using 75th percentile of marker signal
- Trained a Multilayer Perceptron (MLP) with ~8000 windows
- Achieved **~85% accuracy** in validation

**How to run:**  
```bash
# Run the MLP training script
python drivedb_baseline_mlp.py
```

### 2. Advanced LSTM Model
- Two-layer LSTM (64 units) trained on overlapping 10s windows
- Features standardized across sessions; dropout = 0.3
- Achieved **92% accuracy**, **0.93 ROC-AUC**, **F1 = 0.90**
- Inference latency = **~15ms** per window on NVIDIA Jetson TX2

**How to run:**  
```bash
python neuroadaptive_infotainment_system_baseline.py
```

---

##  Directory Structure

```
drivedb/

├── drivedb_baseline_mlp.py         # Baseline MLP classifier
├── neuroadaptive_infotainment_system_baseline.py  # Final LSTM pipeline
├── README.md                       # This file
```

---

## Setup Instructions

We recommend using [uv](https://docs.astral.sh/uv/) for fast package management.

### Quick Start
```bash
# Install uv (optional, faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo and enter the folder
git clone https://github.com/strivn/tnivs-neuroadaptive-ml
cd tnivs
uv sync
source .venv/bin/activate

# Download and prepare DriveDB (manual download from PhysioNet)
mkdir -p drivedb/data/
# Place all .hea and .dat files into drivedb/data/

# Run preprocessing and modeling
python drivedb/drivedb_baseline_mlp.py
```

---

##  Results Summary

| Metric        | Value    |
|---------------|----------|
| Accuracy      | 92%      |
| ROC-AUC       | 0.93     |
| Precision     | 0.91     |
| Recall        | 0.90     |
| Inference Time| ~15ms    |

**Top Contributing Features (via SHAP):**
- `GSR_mean` (hand)
- `ECG_std` (heart rate variability)
- `EMG_rms`
- `RESP_mean`
- `HRV LF/HF ratio`

---

##  Stress Labeling Logic

Since DriveDB does not include labeled stress intervals:
- We use `marker_mean` (manual event count signal)
- Sessions with top 25% marker_mean are labeled as "High Stress"
- Each 10s window inherits session-level label

---

##  References

- Healey, J. A., & Picard, R. W. (2005). [Detecting stress during real-world driving tasks using physiological sensors](https://doi.org/10.1109/TITS.2005.848368). *IEEE Transactions on Intelligent Transportation Systems*.
- PhysioNet DriveDB: https://physionet.org/content/drivedb/1.0.0/
- Final Report: See `Final Report (1).pdf` in repository root

---

##  Notes

- Data files (`.dat`, `.hea`) are not committed due to size. Please manually download from PhysioNet.
- Stress detection pipeline built with real-time deployment in mind (≤15 ms latency).
- Visualization notebooks available in `/notebooks/`.

---

##  Naming Convention

- `drivedb_baseline_mlp.py`: MLP stress detector
- `neuroadaptive_infotainment_system_baseline.py`: Final production-ready pipeline (LSTM)
- `Driver_Stress_Recognition_EDA.ipynb`: Visual EDA of signal stats and labeling

---

##  Future Work

- Real-time model deployment on edge devices (Jetson Nano, Pi)
- Signal fusion with context (GPS, traffic, time of day)
- Adaptive infotainment UI for stress mitigation
