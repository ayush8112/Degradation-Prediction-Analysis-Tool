# Required Packages:
# To run this script, you need to install the following libraries.
# Open your terminal or command prompt and run these commands:
# pip install pandas
# pip install matplotlib
# pip install scikit-learn
# pip install openpyxl
# pip install scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

class DegradationAnalysis:
    """
    A comprehensive and flexible class for analyzing and predicting material degradation
    from multiple user-defined datasets using a user-selected set of models.
    """

    def __init__(self, datasets_config, selected_models):
        """
        Initializes the analysis with dataset configurations and a list of selected models.
        """
        if not datasets_config: raise ValueError("Dataset configuration cannot be empty.")
        if not selected_models: raise ValueError("A list of models to run must be provided.")
            
        self.datasets_config = datasets_config
        self.selected_models = selected_models
        
        self.output_dir = os.path.dirname(self.datasets_config[0]['path']) or '.'
        self.graphs_dir = os.path.join(self.output_dir, 'graphs')
        self.results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"\n✅ All outputs will be saved to '{self.graphs_dir}' and '{self.results_dir}'")

        self.batches = ['Standard Batch (0%)', 'Neem, 25%', 'Eucalyptus, 25%', 'Moringa, 25%']
        
        self.dataframes = {}
        self.all_models = {}

    # --- MODEL FUNCTIONS ---
    def _logarithmic_func(self, x, a, b, c): return a * np.log(np.maximum(x + b, 1e-6)) + c
    def _linear_func(self, x, m, c): return m * x + c
    def _power_law_func(self, x, a, b, c): return a * np.power(x, b) + c
    def _exponential_func(self, x, a, b, c): return a * np.exp(np.clip(b * x, -np.inf, 100)) + c

    def _preprocess_data(self, file_path):
        print(f"\nPreprocessing data from '{os.path.basename(file_path)}'...")
        try:
            df = pd.read_excel(file_path, sheet_name='t-DR')
        except ValueError:
            try:
                xl_file = pd.ExcelFile(file_path)
                df = pd.read_excel(file_path, sheet_name=xl_file.sheet_names[0])
            except Exception as e:
                raise IOError(f"Could not read the Excel file '{file_path}'. Error: {e}")
        
        df.columns = df.columns.str.strip()
        time_data = None
        for col in ['Day', 'Days', 'Time', 'Hours', 'day', 'days', 'time', 'hours']:
            if col in df.columns:
                df = df.dropna(subset=[col])
                time_data = (pd.to_numeric(df[col], errors='coerce').values / 24) if 'hour' in col.lower() else pd.to_numeric(df[col], errors='coerce').values
                break
        if time_data is None: raise KeyError(f"Could not find a valid time column (e.g., 'Day') in {os.path.basename(file_path)}.")
        
        processed_data = {'Day': time_data}
        for batch in self.batches:
            if batch in df.columns:
                processed_data[f"{batch}_degradation"] = np.maximum(pd.to_numeric(df[batch], errors='coerce').interpolate(method='linear').values, 0)
        
        processed_df = pd.DataFrame(processed_data).dropna()
        if len(processed_df) < 3: raise ValueError(f"Insufficient data points (<3) in {os.path.basename(file_path)}.")
        
        output_path = os.path.join(self.results_dir, f'{os.path.basename(file_path).split(".")[0].lower()}_processed.csv')
        processed_df.to_csv(output_path, index=False)
        print(f"Successfully processed {len(processed_data)-1} batches. Saved to {output_path}")
        return processed_df

    def _fit_model_with_fallbacks(self, x_data, y_data, model_func, model_name):
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean, y_clean = x_data[finite_mask], y_data[finite_mask]
        if len(x_clean) < 3: return None, None
        
        param_strategies, slope_est = [], (y_clean[-1] - y_clean[0]) / (x_clean[-1] - x_clean[0]) if x_clean[-1] != x_clean[0] else 0

        if model_name.lower() == 'linear': param_strategies = [[slope_est, y_clean[0]], [0.1, 0]]
        elif model_name.lower() == 'logarithmic': param_strategies = [[y_clean.max() - y_clean.min(), 1, y_clean.min()], [10, 1, 0]]
        elif model_name.lower() == 'power law': param_strategies = [[slope_est, 1, y_clean[0]], [1, 0.5, 0]]
        elif model_name.lower() == 'exponential': param_strategies = [[1.0, 0.01, y_clean[0]], [0.1, 0.01, 0]]

        best_params, best_metrics = None, {"r2": -np.inf}
        for params in param_strategies:
            try:
                popt, _ = curve_fit(model_func, x_clean + 1e-9, y_clean, p0=params, maxfev=10000)
                y_pred = model_func(x_clean + 1e-9, *popt)
                if not np.all(np.isfinite(y_pred)): continue
                r2 = r2_score(y_clean, y_pred)
                if r2 > best_metrics["r2"] and r2 >= 0:
                    best_params, mse = popt, mean_squared_error(y_clean, y_pred)
                    best_metrics = {"r2": r2, "mse": mse, "rmse": np.sqrt(mse)}
            except Exception: continue
        return best_params, best_metrics if best_params is not None else None

    def _create_models_for_dataset(self, dataset_info):
        dataset_name, file_path = dataset_info['name'], dataset_info['path']
        df = self._preprocess_data(file_path)
        self.dataframes[dataset_name] = df
        
        models_for_dataset = self.all_models.setdefault(dataset_name, {})

        all_available_models = {
            "Linear": self._linear_func, "Logarithmic": self._logarithmic_func,
            "Power Law": self._power_law_func, "Exponential": self._exponential_func
        }
        models_to_fit = {name: func for name, func in all_available_models.items() if name in self.selected_models}

        days_for_prediction = np.linspace(0, 2000, 4000) 
        
        print(f"\n{'='*60}\nCreating Models for '{dataset_name}'\n{'='*60}")
        for batch in self.batches:
            col = f"{batch}_degradation"
            if col not in df.columns: continue
            print(f"\n--- Processing {batch} ---")
            days_actual, degradation_actual = df["Day"].values, df[col].values
            
            if len(days_actual) < 3 or np.all(degradation_actual == degradation_actual[0]):
                print("No variation in data or insufficient points."); continue
            
            models_for_dataset[batch] = {}
            best_model_name, best_r2_val = None, -np.inf
            for name, func in models_to_fit.items():
                params, metrics = self._fit_model_with_fallbacks(days_actual, degradation_actual, func, name)
                if params is None: print(f"    Failed to fit {name} model."); continue
                
                print(f"    {name}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
                pred = func(days_for_prediction + 1e-9, *params)
                days_to_50 = self._find_days_to_threshold(days_for_prediction, pred, 50)
                models_for_dataset[batch][name] = {"params": params, "metrics": metrics, "days_to_50": days_to_50}
                if metrics['r2'] > best_r2_val: best_r2_val, best_model_name = metrics['r2'], name
                self._create_individual_plot(days_actual, degradation_actual, days_for_prediction, pred, batch, dataset_name, name, metrics, days_to_50)
            if best_model_name: print(f"  Best model for {batch}: {best_model_name} (R² = {best_r2_val:.4f})")

    def _find_days_to_threshold(self, days, degradation, threshold):
        indices = np.where(degradation >= threshold)[0]
        if len(indices) == 0: return "Not Reached"
        idx = indices[0]
        if idx == 0: return days[0]
        x1, y1, x2, y2 = days[idx-1], degradation[idx-1], days[idx], degradation[idx]
        return x1 + (x2 - x1) * (threshold - y1) / (y2 - y1) if y2 != y1 else days[idx]

    def _get_equation_text(self, model_name, metrics):
        eq_map = {
            'linear': r'$y = mx + c$', 'logarithmic': r'$y = a \cdot \ln(x + b) + c$',
            'power law': r'$y = a \cdot x^b + c$', 'exponential': r'$y = a \cdot e^{bx} + c$'
        }
        eq = eq_map.get(model_name.lower(), 'Unknown Model')
        return f"Model: {eq}\n$R^2$   = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}"

    def _create_individual_plot(self, days_actual, degradation_actual, days_for_prediction, degradation_predicted, batch, dataset_name, model_name, metrics, days_to_50):
        plt.figure(figsize=(12, 8)); ax = plt.gca()
        if isinstance(days_to_50, float): x_axis_limit = np.ceil(days_to_50 / 5) * 5
        else: x_axis_limit = np.ceil((days_actual.max() * 1.25) / 5) * 5
        
        limit_idx = np.searchsorted(days_for_prediction, x_axis_limit)
        days_plot, pred_plot = days_for_prediction[:limit_idx+1], degradation_predicted[:limit_idx+1]
        
        ax.scatter(days_actual, degradation_actual, color="red", s=50, label="Actual Data", zorder=5)
        color_map = {'Linear': 'blue', 'Logarithmic': 'green', 'Power Law': 'purple', 'Exponential': 'brown'}
        color = color_map.get(model_name, 'black')
        
        max_actual_day = days_actual.max()
        split_idx = np.searchsorted(days_plot, max_actual_day)
        
        ax.plot(days_plot[:split_idx+1], pred_plot[:split_idx+1], color=color, linewidth=2.5, label=f"{model_name} Model ($R^2$ = {metrics['r2']:.4f})")
        ax.plot(days_plot[split_idx:], pred_plot[split_idx:], color=color, linewidth=2.5, linestyle='--')
        ax.axvline(x=max_actual_day, color='gray', linestyle=':', linewidth=2, label=f'End of Actual Data ({max_actual_day:.0f} days)')
        ax.axhline(y=50, color="black", linestyle="--", alpha=0.7, label="50% Degradation Threshold")
        
        if isinstance(days_to_50, float) and days_to_50 <= x_axis_limit:
            ax.axvline(x=days_to_50, color="orange", linestyle=":", alpha=0.8)
            ax.text(days_to_50 * 1.02, 45, f"{days_to_50:.1f} days", color="orange", fontweight='bold', rotation=90)
        
        ax.set(title=f"{batch} - {dataset_name}\n{model_name} Model Extrapolation", xlabel="Time (Days)", ylabel="Degradation (%)", ylim=(0, 105), xlim=(0, x_axis_limit))
        ax.legend(fontsize=10)
        
        final_ticks = None
        for num_intervals in range(8, 3, -1):
            if x_axis_limit > 0 and x_axis_limit % num_intervals == 0:
                interval = x_axis_limit / num_intervals
                if interval >= 1: final_ticks = np.arange(0, x_axis_limit + 1, interval); break
        if final_ticks is None: final_ticks = np.round(np.linspace(0, x_axis_limit, num=6))
        ax.set_xticks(np.unique(final_ticks.astype(int)))

        ax.text(0.02, 0.98, self._get_equation_text(model_name, metrics), transform=ax.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        safe_batch_name = ''.join(c for c in batch if c.isalnum() or c in ' _').rstrip()
        safe_dataset_name = ''.join(c for c in dataset_name if c.isalnum()).rstrip()
        filepath = os.path.join(self.graphs_dir, f"{safe_batch_name}_{safe_dataset_name}_{model_name}_model.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"    Plot saved: {os.path.basename(filepath)}")

    def create_summary_tables(self):
        print(f"\n{'='*80}\nDEGRADATION ANALYSIS SUMMARY\n{'='*80}")
        for dataset_name, models_data in self.all_models.items():
            print(f"\nSUMMARY FOR: '{dataset_name}'\n{'-'*60}")
            
            for model_type in self.selected_models:
                print(f"\n--- {model_type.upper()} MODEL RESULTS ---")
                results_data = []
                for batch in self.batches:
                    model_info = models_data.get(batch, {}).get(model_type, {})
                    metrics = model_info.get('metrics', {})
                    row = {"Batch": batch, "Days to 50%": model_info.get('days_to_50', 'N/A'),
                           "R²": metrics.get('r2', 'N/A'), "RMSE": metrics.get('rmse', 'N/A')}
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                for col in ['R²', 'RMSE']: results_df[col] = pd.to_numeric(results_df[col], errors='coerce').map('{:.4f}'.format)
                print(results_df.to_string(index=False))
                safe_name = ''.join(c for c in dataset_name if c.isalnum()).rstrip()
                results_df.to_csv(os.path.join(self.results_dir, f'summary_{safe_name}_{model_type}.csv'), index=False)
            print(f"Results for '{dataset_name}' saved to CSV files.")

    def create_performance_heatmaps(self):
        print(f"\n{'='*60}\nCREATING PERFORMANCE HEATMAPS\n{'='*60}")
        for dataset_name, models_data in self.all_models.items():
            metrics_data = {
                'R²': np.full((len(self.selected_models), len(self.batches)), np.nan),
                'RMSE': np.full((len(self.selected_models), len(self.batches)), np.nan)
            }
            for i, model_name in enumerate(self.selected_models):
                for j, batch_name in enumerate(self.batches):
                    metrics = models_data.get(batch_name, {}).get(model_name, {}).get('metrics')
                    if metrics:
                        metrics_data['R²'][i, j] = metrics.get('r2')
                        metrics_data['RMSE'][i, j] = metrics.get('rmse')
            
            safe_name = ''.join(c for c in dataset_name if c.isalnum()).rstrip()
            self._plot_heatmap(metrics_data['R²'], f'{dataset_name}\n$R^2$ Performance', f'heatmap_{safe_name}_R2.png')
            self._plot_heatmap(metrics_data['RMSE'], f'{dataset_name}\nRMSE Performance', f'heatmap_{safe_name}_RMSE.png', cmap=plt.cm.RdYlGn_r)

    def _plot_heatmap(self, data, title, filename, cmap=plt.cm.RdYlGn):
        fig, ax = plt.subplots(figsize=(10, 5))
        if np.isnan(data).all():
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([])
            plt.savefig(os.path.join(self.graphs_dir, filename), dpi=300); plt.close()
            return

        ax.imshow(data, cmap=cmap, aspect='auto')
        ax.set(xticks=np.arange(len(self.batches)), yticks=np.arange(len(self.selected_models)),
               xticklabels=self.batches, yticklabels=self.selected_models, title=title)
        ax.set_xlabel('Batches', fontsize=14, fontweight='bold', labelpad=15)
        ax.set_ylabel('Models', fontsize=14, fontweight='bold', labelpad=15)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

        for i in range(len(self.selected_models)):
            for j in range(len(self.batches)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f'{data[i, j]:.3f}', ha="center", va="center", color="black", fontsize=16)

        # --- RESTORED GRIDLINES ---
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color='black', linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        # --- END OF FIX ---

        fig.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, filename), dpi=300); plt.close()
        print(f"Heatmap saved: {filename}")

    def generate_detailed_report(self):
        report_path = os.path.join(self.results_dir, "degradation_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("DEGRADATION ANALYSIS COMPREHENSIVE REPORT\n" + "="*60 + "\n")
            for dataset_name, models in self.all_models.items():
                df_ = self.dataframes[dataset_name]
                f.write(f"\n\n{'='*60}\nANALYSIS FOR DATASET: '{dataset_name}'\n{'='*60}\n")
                f.write(f"  Data Summary:\n  Data points: {len(df_)}\n  Time range: {df_['Day'].min():.1f} - {df_['Day'].max():.1f} days\n\n")
                f.write("DETAILED RESULTS BY BATCH:\n" + "="*40 + "\n")
                for batch in self.batches:
                    f.write(f"\n{batch}:\n" + "-"*len(batch) + "-\n")
                    batch_models = models.get(batch, {})
                    if batch_models:
                        for name in self.selected_models:
                            if name in batch_models:
                                data = batch_models[name]
                                days_str = f"{data['days_to_50']:.2f}" if isinstance(data['days_to_50'], float) else data['days_to_50']
                                m = data['metrics']
                                f.write(f"    {name:<12}: R²={m['r2']:.4f}, RMSE={m['rmse']:.4f}, Days to 50% = {days_str}\n")
                    else: f.write("    No valid models created for this batch.\n")
        print(f"Detailed report saved to: {report_path}")

    def run_complete_analysis(self):
        print("\n🚀 Starting complete degradation analysis workflow...")
        try:
            for dataset_info in self.datasets_config:
                self._create_models_for_dataset(dataset_info)
            self.create_summary_tables()
            self.create_performance_heatmaps()
            self.generate_detailed_report()
            print(f"\n{'='*60}\n✅ ANALYSIS COMPLETED SUCCESSFULLY!\n{'='*60}")
        except Exception as e:
            print(f"\n❌ ERROR DURING ANALYSIS: {e}"); import traceback; traceback.print_exc()

def main():
    print("="*60 + "\nDEGRADATION ANALYSIS & EXTRAPOLATION TOOL\n" + "="*60)
    datasets_config = []
    
    while True:
        try:
            num_datasets = int(input("Enter the number of datasets to analyze (1-10): "))
            if 1 <= num_datasets <= 10: break
            else: print("Error: Please enter a number between 1 and 10.")
        except ValueError: print("Error: Invalid input. Please enter a whole number.")

    for i in range(num_datasets):
        print(f"\n--- Configuring Dataset {i+1} ---")
        name = input(f"Enter a descriptive name for this dataset: ").strip()
        while True:
            path = input(f"Enter the full path to the Excel file for '{name}': ").strip().strip('"')
            if os.path.exists(path): datasets_config.append({'name': name, 'path': path}); break
            else: print(f"❌ File not found at '{path}'. Please try again.")

    available_models = ['Linear', 'Logarithmic', 'Power Law', 'Exponential']
    selected_model_names = []
    print("\n" + "="*60 + "\nMODEL SELECTION\n" + "="*60)
    print("Please select the models you want to run:")
    for i, model in enumerate(available_models):
        print(f"  {i+1}: {model}")
    print("  5: All")
    
    while True:
        choice = input("Enter your choice(s) (e.g., '1', '2,4', or 'all'): ").lower().strip()
        if choice == 'all' or choice == '5':
            selected_model_names = available_models
            break
        try:
            indices = [int(i.strip()) - 1 for i in choice.split(',')]
            if all(0 <= index < len(available_models) for index in indices):
                selected_model_names = [available_models[i] for i in sorted(list(set(indices)))]
                break
            else: print("Error: One or more numbers are out of the valid range (1-4).")
        except ValueError:
            print("Error: Invalid input. Please enter numbers separated by commas, or 'all'.")

    print(f"\nModels selected for analysis: {', '.join(selected_model_names)}")
    
    try:
        analysis = DegradationAnalysis(datasets_config, selected_model_names)
        analysis.run_complete_analysis()
        print(f"\n🎉 Analysis complete! Check the 'graphs' and 'results' folders.")
    except (FileNotFoundError, ValueError, KeyError, IOError) as e:
        print(f"\n❌ A configuration or data error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
