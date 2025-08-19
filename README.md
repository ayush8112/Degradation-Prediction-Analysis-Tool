Degradation Analysis and Predictive Modeling Tool
1. Overview
This Python-based tool offers a comprehensive, automated solution for analyzing and predicting material degradation over time. By processing time-series data from two different environmental conditions (e.g., room temperature and refrigerator), it fits four distinct mathematical models to accurately forecast material lifespan.
The script is designed for researchers, engineers, and scientists who need to extrapolate material stability, compare different formulations (batches), and rigorously evaluate the performance of predictive models. It automates the entire workflow from data cleaning to generating a full suite of publication-ready outputs, including detailed graphs, performance tables, and comparative heatmaps.
2. Key Features
Dual-Condition Analysis: Simultaneously processes and compares data from two separate Excel files (e.g., Room vs. Refrigerator temperature).
Automated Data Preprocessing: Intelligently finds and cleans time-series data from input files, handling common column names like 'Day', 'Days', 'Time', or 'Hours'.
Multi-Model Fitting: Implements four robust models to capture different degradation behaviors:
Linear: For constant-rate degradation.
Logarithmic: For degradation that slows over time.
Power Law: For flexible, non-linear degradation (both accelerating and decelerating).
Exponential: For degradation that accelerates over time.
Lifespan Prediction: Calculates and reports the predicted time (in days) for each batch to reach a 50% degradation threshold.
Publication-Ready Visualizations:
Generates individual plots for every batch, condition, and model.
Features intelligent axis scaling that automatically creates scientifically logical, evenly-spaced intervals for optimal data presentation.
Clearly distinguishes between interpolated data and extrapolated predictions using solid and dashed lines.
In-Depth Performance Metrics: For each model, it calculates and displays the R² (Coefficient of Determination) and RMSE (Root Mean Squared Error).
Comparative Heatmaps: Creates intuitive heatmaps to visually compare the performance (R², RMSE) of all models across all batches for a given condition.
Comprehensive Reporting:
Generates clean, easy-to-read summary tables in the console.
Saves summary tables to CSV files for further analysis.
Produces a detailed text-based report summarizing all findings.
Organized Output: Automatically creates graphs/ and results/ directories to save all generated files neatly.
3. Models Used
The tool leverages four different models to provide a comprehensive analysis of degradation trends:
Model	Equation	Description
Linear	$y = mx + c$	Represents a constant rate of degradation. Simple, but effective as a baseline.
Logarithmic	$y = a \cdot \ln(x + b) + c$	Models a process that is initially rapid but slows down and levels off over time.
Power Law	$y = a \cdot x^b + c$	A flexible non-linear model. If b > 1, it accelerates; if 0 < b < 1, it decelerates. A robust replacement for polynomial models as it never peaks.
Exponential	$y = a \cdot e^{bx} + c$	Describes a process that starts slowly and then accelerates rapidly, often seen in autocatalytic reactions.
4. Prerequisites
Before running the script, you need to have Python installed on your system. You also need to install the following Python libraries:
pandas
matplotlib
scikit-learn
openpyxl
scipy
You can install them all by running the following command in your terminal or command prompt:
code
Bash
pip install pandas matplotlib scikit-learn openpyxl scipy
5. Usage
Save the script to a file (e.g., analysis.py).
Place your two Excel data files in a convenient location (preferably the same directory).
Open a terminal or command prompt and navigate to the directory where you saved the script.
Run the script using the following command:
code
Bash
python analysis.py
The script will prompt you to enter the file paths for the room temperature data and the refrigerator temperature data. Paste each path and press Enter.
The analysis will run automatically. Once completed, check the graphs and results sub-folders that have been created in the same directory as your input files.
6. Input File Format
The script expects the input data to be in an Excel (.xlsx or .xls) file. The sheet should contain:
A time column, which can be named Day, Days, Time, or Hours. If 'Hours' is used, the values will be automatically converted to days.
Separate columns for each batch. The column names must exactly match the ones defined in the script (Standard Batch (0%), Neem, 25%, etc.).
Example Data Structure:
Day	Standard Batch (0%)	Neem, 25%	Eucalyptus, 25%	Moringa, 25%
0	0.5	0.4	0.6	0.5
7	5.2	4.8	5.5	5.1
14	10.1	9.5	10.8	9.9
21	15.6	14.2	16.1	15.0
...	...	...	...	...
7. Output Files
After a successful run, the following files will be generated:
graphs/
Individual Model Plots: [Batch_Name]_[Condition]_[Model]_model.png - A detailed plot for each combination of batch, condition, and model.
Performance Heatmaps: heatmap_[Condition]_[Metric].png - Visual summaries of model performance (R², MSE, RMSE) for each condition.
results/
Processed Data: [input_file_name]_processed.csv - The cleaned, interpolated data used for modeling.
Summary Tables: summary_[Model].csv - CSV files summarizing the key results (R², RMSE, Days to 50%) for each model type.
Comprehensive Report: degradation_analysis_report.txt - A detailed text file containing all metrics and predictions for every batch and model, providing a complete overview of the analysis.