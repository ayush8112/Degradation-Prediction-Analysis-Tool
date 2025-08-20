# Degradation Prediction & Analysis Tool

A flexible Python toolkit to analyze multiple user-defined datasets, fit a selection of mathematical models, compare their performance, visualize results, and extrapolate lifetime estimates (for example, days to 50% degradation).

## Table of contents
- Features
- Quick start
- Prerequisites
- Installation
- Data format
- Usage
- Models & metrics
- Output structure
- Examples
- Notes & caveats
- Troubleshooting
- Contributing
- License
- Contact

## Features
- Automatic preprocessing and cleaning of Excel time-series degradation data.
- Handles 1 to 10 user-defined datasets in parallel, allowing for flexible comparison across various conditions.
- Interactive Model Selection: Choose to run a single model, a specific subset, or all available models.
- Fits four robust model classes: Linear, Logarithmic, Power-law, and Exponential.
- Computes R², MSE, and RMSE for each fit to evaluate performance.
- Extrapolates degradation and estimates the number of days to reach a 50% degradation threshold.
- Generates publication-ready, per-batch plots and metric comparison heatmaps.
- Saves all processed data, metric summaries, and a comprehensive, human-readable report.

## Quick start
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

# create and activate venv (recommended)
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt

# run analysis
python Prediction.py
```

## Prerequisites
- Python 3.6+
- Recommended: use a virtual environment.
- Entries for `requirements.txt`:
  ```
  pandas
  numpy
  matplotlib
  scikit-learn
  openpyxl
  scipy
  ```

## Installation
1. Clone the repo.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Prepare your Excel input files.

## Data format
Provide **one to ten Excel files**, each representing a different experimental condition or dataset. The tool interactively prompts for the number of datasets and their respective paths.

**Requirements for each Excel file:**
- A **time column**. Accepted names: `Day`, `Days`, `Time`, `Hours` (case-insensitive). Values must be numeric.
- One or more **batch columns**. Each column contains degradation percent values (0–100). Column headers should be descriptive (e.g., `Standard Batch (0%)`, `Neem, 25%`).
- Missing values (NaN) are tolerated and are handled via linear interpolation during preprocessing.

**Example sheet layout**

| Day | Standard Batch (0%) | Neem, 25% |
|-----|---------------------|-----------|
| 0   | 0.0                 | 0.0       |
| 5   | 2.3                 | 1.8       |
| 15  | 6.2                 | 5.0       |

## Usage
Run the script from your terminal:
```bash
python Prediction.py
```

The script is fully interactive and will guide you through a three-step setup process:

1. **Dataset Configuration**
   - Enter the number of datasets (1–10).
   - Provide dataset name and Excel file path.

2. **Model Selection**
   - Choose one, multiple, or all models.

3. **Analysis Execution**
   - Runs complete analysis and saves outputs automatically.

## Models & metrics

**Models fitted**
- Linear: y = mx + c
- Logarithmic: y = a * ln(x + b) + c
- Power law: y = a * x^b + c
- Exponential: y = a * e^(bx) + c

**Metrics**
- R² — Coefficient of Determination
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error

The script identifies the best model for each batch based on the highest R² value.

## Output structure
The script creates `graphs/` and `results/` folders in the same directory as the first input file.

```
<input-dir>/
  graphs/
    <BatchName>_<DatasetName>_<Model>_model.png
    heatmap_<DatasetName>_R2.png
    heatmap_<DatasetName>_RMSE.png
  results/
    <input-file>_processed.csv
    summary_<DatasetName>_<Model>.csv
    degradation_analysis_report.txt
```

`degradation_analysis_report.txt` includes:
- Dataset summary (time range, number of points).
- Performance metrics and days-to-50% estimates for each batch and model.
- Results separated per dataset.

## Examples
Example flow with two datasets and two models:
```
Enter datasets: 2

Dataset 1 → Room Temperature → C:\data\room_study.xlsx
Dataset 2 → 4C Refrigerator → C:\data\fridge_study.xlsx

Select models: 1, 3 (Linear, Power Law)
```

Outputs saved under `C:\data\graphs` and `C:\data\results`.

## Notes & caveats
- Logarithmic and Power Law require strictly positive time values. A small value is added automatically.
- Extrapolation depends on model and data quality. Treat days-to-50% as estimates, not exact predictions.
- Sparse or noisy datasets may favor simpler models.
- If 50% threshold is not reached, report flags as "Not Reached".

## Troubleshooting
- **Time column missing:** Ensure column named Day, Days, Time, or Hours with numeric values.
- **Poor fits (low R²):** Check duplicates, negatives, or outliers.
- **Unrealistic plots:** Check data scaling and integrity.

## Contributing
- Contributions welcome via issues and PRs.

**Future improvements:**
- CLI arguments for non-interactive use.
- Bootstrap confidence intervals.
- Advanced model selection (AIC/BIC).
- Web UI (Streamlit) or Jupyter interface.

**Git workflow:**
```bash
git checkout -b feature/your-feature
# implement
git commit -m "Add feature"
git push origin feature/your-feature
# open PR
```
