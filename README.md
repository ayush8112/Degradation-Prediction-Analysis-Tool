# Degradation Prediction & Analysis Tool

A focused Python toolkit to analyze time-series material degradation, fit multiple mathematical models, compare their performance, visualize results, and extrapolate lifetime estimates (for example, days to 50% degradation).

---

## Table of contents
- [Features](#features)
- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data format](#data-format)
- [Usage](#usage)
- [Models & metrics](#models--metrics)
- [Output structure](#output-structure)
- [Examples](#examples)
- [Notes & caveats](#notes--caveats)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features
- Automatic preprocessing and cleaning of Excel time-series degradation data.
- Fits four model classes: Linear, Logarithmic, Power-law, Exponential.
- Computes R², MSE, RMSE for each fit.
- Extrapolates degradation and estimates days to reach 50% degradation.
- Generates per-batch plots and metric heatmaps.
- Saves processed data, metric summaries, and a human-readable report.

---

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

---

## Prerequisites
- Python 3.6+
- Recommended: use a virtual environment.

Recommended entries for `requirements.txt`:
```
pandas
numpy
matplotlib
scikit-learn
openpyxl
scipy
```

---

## Installation
1. Clone the repo.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Place your input Excel files where you want or note their full paths for the script prompt.

---

## Data format
Provide **two Excel files**: one for room temperature data and one for refrigerator data. Requirements:

- A time column. Accepted names: `Day`, `Days`, `Time`, `Hours` (case-insensitive). Values must be numeric and strictly increasing rows represent later times.
- One or more batch columns. Each column contains degradation percent values (0–100). Column headers can be descriptive, e.g. `Standard Batch (0%)`, `Neem 25%`.
- Missing values (NaN) are tolerated and handled in preprocessing.

**Example sheet layout**
| Day | Standard Batch (0%) | Neem, 25% |
|-----|---------------------|-----------|
| 0   | 0.0                 | 0.0       |
| 5   | 2.3                 | 1.8       |
| 15  | 6.2                 | 5.0       |

---

## Usage
Run:
```bash
python Prediction.py
```
The script will prompt:
- `Enter the path to the ROOM temperature data Excel file:`
- `Enter the path to the REFRIGERATOR temperature data Excel file:`  

Processing steps:
1. Load and clean each Excel file.
2. For each batch column:
   - Fit Linear, Logarithmic, Power-law, Exponential models.
   - Compute metrics: R², MSE, RMSE.
   - Extrapolate forward (configurable in the script).
   - Find earliest day where predicted degradation ≥ 50% (if model crosses 50%).
3. Save plots, CSV summaries, and a detailed `.txt` report.

---

## Models & metrics
**Models fitted**
- Linear: `y = a*x + b`
- Logarithmic: `y = a * log(x) + b`
- Power law: `y = a * x^b`
- Exponential: `y = a * exp(b*x)`

**Metrics**
- R² — coefficient of determination
- MSE — mean squared error
- RMSE — root mean squared error

The script ranks models using R² and RMSE and reports a per-batch "best model".

---

## Output structure
For each input file the script creates `graphs/` and `results/` folders next to the input file:

```
<input-dir>/
  graphs/
    <batch>_<model>_fit.png
    <batch>_prediction_summary.png
    metrics_heatmap_R2.png
    metrics_heatmap_MSE.png
    metrics_heatmap_RMSE.png
  results/
    processed_<input-file>.csv
    metrics_summary_<input-file>.csv
    report_<input-file>.txt
```

**report_*.txt** includes:
- Summary of preprocessing actions.
- Per-batch model metrics and chosen best model.
- Days-to-50% estimates per model.
- Flags and caveats where extrapolation is unreliable.

---

## Examples
Example CLI flow:
```
Enter the path to the ROOM temperature data Excel file: /home/ayush/data/room_temp.xlsx
Enter the path to the REFRIGERATOR temperature data Excel file: /home/ayush/data/fridge_temp.xlsx
```
Outputs appear under `/home/ayush/data/graphs` and `/home/ayush/data/results`.

---

## Notes & caveats
- Log and power models require strictly positive time values. The script will shift time values if zeros are present or skip models that are invalid.
- Extrapolation is model-dependent and sensitive to noise and sampling range. Treat days-to-50% as estimates not absolute facts.
- Sparse or highly noisy datasets will often favor simpler models. Validate model choice with domain knowledge.
- If none of the fitted models cross 50% within the extrapolation horizon, the report will flag this.

---

## Troubleshooting
- **Script fails to find time column**: rename your time column to `Day` or `Time` or ensure it is numeric.
- **Models produce NaN or inf**: check for duplicate time rows, negative or zero time values for log/power, or infinite values in input.
- **Plots look unrealistic**: check data scaling and outliers. Consider trimming erroneous data or increasing sampling.

---

## Contributing
Contributions accepted by issues and pull requests.
Suggested improvements:
- Add CLI arguments and config file.
- Add bootstrap confidence intervals for predictions.
- Add automated model-selection logic.
- Add a simple web UI or Jupyter notebook interface.

Suggested Git workflow:
```bash
git checkout -b feature/your-feature
# implement
git commit -m "Add feature"
git push origin feature/your-feature
# open PR
```

---

## License
This project is available under the MIT License. Add a `LICENSE` file with the standard MIT text.

**Short notice**: you must add a `LICENSE` file containing the full MIT license to apply the license.

---
