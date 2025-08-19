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
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

class DegradationAnalysis:
    """
    A comprehensive class for analyzing and predicting material degradation
    with fully automatic, data-driven graph axis scaling and detailed error metrics,
    including performance summary heatmaps.
    """

    def __init__(self, room_excel_file_path, ref_excel_file_path):
        """
        Initializes the DegradationAnalysis class with two separate files.
        """
        if not os.path.exists(room_excel_file_path):
            raise FileNotFoundError(f"Error: The room temperature file '{room_excel_file_path}' was not found.")
        if not os.path.exists(ref_excel_file_path):
            raise FileNotFoundError(f"Error: The refrigerator temperature file '{ref_excel_file_path}' was not found.")

        self.room_excel_file_path = room_excel_file_path
        self.ref_excel_file_path = ref_excel_file_path
        
        self.output_dir = os.path.dirname(room_excel_file_path) or '.'
        self.graphs_dir = os.path.join(self.output_dir, 'graphs')
        self.results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Output will be saved to '{self.graphs_dir}' and '{self.results_dir}'")

        self.batches = ['Standard Batch (0%)', 'Neem, 25%', 'Eucalyptus, 25%', 'Moringa, 25%']
        # --- REPLACED 'Polynomial' with 'Power Law' ---
        self.model_names = ['Linear', 'Logarithmic', 'Power Law', 'Exponential']
        self.room_df, self.ref_df = None, None
        self.room_models, self.ref_models = {}, {}

    def _logarithmic_func(self, x, a, b, c):
        return a * np.log(np.maximum(x + b, 1e-6)) + c

    def _linear_func(self, x, m, c):
        return m * x + c
    
    # --- NEW MODEL FUNCTION: POWER LAW ---
    def _power_law_func(self, x, a, b, c):
        # Use np.power for safe handling of exponents
        return a * np.power(x, b) + c
    
    def _exponential_func(self, x, a, b, c):
        return a * np.exp(np.clip(b * x, -np.inf, 100)) + c

    def _preprocess_data(self, file_path, sheet_name='t-DR'):
        print(f"\nPreprocessing data from '{os.path.basename(file_path)}'...")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError:
            xl_file = pd.ExcelFile(file_path)
            sheet_name = xl_file.sheet_names[0]
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        df.columns = df.columns.str.strip()
        time_data = None
        for col in ['Day', 'Days', 'Time', 'Hours', 'day', 'days', 'time', 'hours']:
            if col in df.columns:
                time_col = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=[col])
                time_data = (pd.to_numeric(df[col], errors='coerce').values / 24) if 'hour' in col.lower() else pd.to_numeric(df[col], errors='coerce').values
                break
        if time_data is None: raise KeyError("Could not find a time column.")
        
        processed_data = {'Day': time_data}
        for batch in self.batches:
            if batch in df.columns:
                processed_data[f"{batch}_degradation"] = np.maximum(pd.to_numeric(df[batch], errors='coerce').interpolate(method='linear').values, 0)
        
        processed_df = pd.DataFrame(processed_data).dropna()
        if len(processed_df) < 3: raise ValueError("Insufficient data points for analysis.")
        
        output_path = os.path.join(self.results_dir, f'{os.path.basename(file_path).split(".")[0].lower()}_processed.csv')
        processed_df.to_csv(output_path, index=False)
        print(f"Successfully processed {len(processed_data)-1} batches. Processed data saved to {output_path}")
        return processed_df

    def _fit_model_with_fallbacks(self, x_data, y_data, model_func, model_name):
        finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean, y_clean = x_data[finite_mask], y_data[finite_mask]
        # 3 points are required to reliably fit a 3-parameter model like Power Law
        if len(x_clean) < 3: return None, None
        
        param_strategies = []
        slope_est = (y_clean[-1] - y_clean[0]) / (x_clean[-1] - x_clean[0]) if x_clean[-1] != x_clean[0] else 0

        if model_name.lower() == 'linear':
            param_strategies = [[slope_est, y_clean[0]], [0.1, 0]]
        elif model_name.lower() == 'logarithmic':
            param_strategies = [[y_clean.max() - y_clean.min(), 1, y_clean.min()], [10, 1, 0]]
        # --- ADDED INITIAL GUESSES FOR POWER LAW MODEL ---
        elif model_name.lower() == 'power law':
            # Guess b=1 (linear-like), a=slope, c=intercept
            param_strategies = [[slope_est, 1, y_clean[0]], [1, 0.5, 0]]
        elif model_name.lower() == 'exponential':
            param_strategies = [[1.0, 0.01, y_clean[0]], [0.1, 0.01, 0]]

        best_params, best_metrics = None, {"r2": -np.inf}
        for params in param_strategies:
            try:
                # Add a small epsilon to x_clean to avoid zero division/domain errors in power/log functions
                popt, _ = curve_fit(model_func, x_clean + 1e-9, y_clean, p0=params, maxfev=10000)
                y_pred = model_func(x_clean + 1e-9, *popt)
                if not np.all(np.isfinite(y_pred)): continue
                r2 = r2_score(y_clean, y_pred)
                if r2 > best_metrics["r2"] and r2 >= 0:
                    best_params = popt
                    mse = mean_squared_error(y_clean, y_pred)
                    best_metrics = {"r2": r2, "mse": mse, "rmse": np.sqrt(mse)}
            except Exception: continue

        return best_params, best_metrics if best_params is not None else None

    def _create_models_for_condition(self, condition):
        df, temp_str, models = (self.room_df, "Room Temperature", self.room_models) if condition == 'room' else (self.ref_df, "Refrigerator Temperature", self.ref_models)
        if df is None:
            df = self._preprocess_data(self.room_excel_file_path if condition == 'room' else self.ref_excel_file_path)
            if condition == 'room': self.room_df = df
            else: self.ref_df = df

        # --- UPDATED THE FITTING DICTIONARY WITH POWER LAW ---
        models_to_fit = {
            "Linear": self._linear_func, 
            "Logarithmic": self._logarithmic_func,
            "Power Law": self._power_law_func,
            "Exponential": self._exponential_func
        }
        days_for_prediction = np.linspace(0, 2000, 4000) 
        
        print(f"\n{'='*60}\nCreating Models for {temp_str}\n{'='*60}")
        for batch in self.batches:
            col = f"{batch}_degradation"
            if col not in df.columns: continue
            print(f"\n--- Processing {batch} ---")
            days_actual, degradation_actual = df["Day"].values, df[col].values
            
            if len(days_actual) < 3 or np.all(degradation_actual == degradation_actual[0]):
                print("No variation in data or insufficient points."); continue
            
            models[batch] = {}
            best_model_name, best_r2_val = None, -np.inf
            for name, func in models_to_fit.items():
                params, metrics = self._fit_model_with_fallbacks(days_actual, degradation_actual, func, name)
                if params is None:
                    print(f"    Failed to fit {name} model."); continue
                
                print(f"    {name}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
                pred = func(days_for_prediction + 1e-9, *params)
                days_to_50 = self._find_days_to_threshold(days_for_prediction, pred, 50)
                models[batch][name] = {"params": params, "metrics": metrics, "days_to_50": days_to_50}
                if metrics['r2'] > best_r2_val:
                    best_r2_val, best_model_name = metrics['r2'], name
                self._create_individual_plot(days_actual, degradation_actual, days_for_prediction, pred, batch, temp_str, name, metrics, days_to_50, condition)
            if best_model_name: print(f"  Best model for {batch}: {best_model_name} (R² = {best_r2_val:.4f})")

    def _find_days_to_threshold(self, days, degradation, threshold):
        indices = np.where(degradation >= threshold)[0]
        if len(indices) == 0: return "Not Reached"
        idx = indices[0]
        if idx == 0: return days[0]
        x1, y1, x2, y2 = days[idx-1], degradation[idx-1], days[idx], degradation[idx]
        return x1 + (x2 - x1) * (threshold - y1) / (y2 - y1) if y2 != y1 else days[idx]

    def _create_individual_plot(self, days_actual, degradation_actual, days_for_prediction, degradation_predicted, batch, temp_str, model_name, metrics, days_to_50, condition):
        plt.figure(figsize=(12, 8)); ax = plt.gca()
        
        if isinstance(days_to_50, float):
            x_axis_limit = np.ceil(days_to_50 / 5) * 5
        else:
            max_actual_day = days_actual.max()
            x_axis_limit = np.ceil((max_actual_day * 1.25) / 5) * 5
        
        limit_idx = np.searchsorted(days_for_prediction, x_axis_limit)
        days_plot, pred_plot = days_for_prediction[:limit_idx+1], degradation_predicted[:limit_idx+1]
        
        ax.scatter(days_actual, degradation_actual, color="red", s=50, label="Actual Data", zorder=5)
        
        # --- UPDATED COLOR MAPPING ---
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
        
        ax.set(title=f"{batch} - {temp_str}\n{model_name} Model Extrapolation", xlabel="Time (Days)", ylabel="Degradation (%)", ylim=(0, 105), xlim=(0, x_axis_limit))
        ax.legend(fontsize=10)
        
        final_ticks = None
        for num_intervals in range(8, 3, -1):
            if x_axis_limit > 0 and x_axis_limit % num_intervals == 0:
                interval = x_axis_limit / num_intervals
                if interval >= 1:
                    final_ticks = np.arange(0, x_axis_limit + 1, interval)
                    break
        if final_ticks is None:
            ticks = np.linspace(0, x_axis_limit, num=6)
            final_ticks = np.round(ticks)

        ax.set_xticks(np.unique(final_ticks.astype(int)))

        equation_text = self._get_equation_text(model_name, metrics)
        ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        safe_batch_name = ''.join(c for c in batch if c.isalnum() or c in ' _').rstrip()
        filepath = os.path.join(self.graphs_dir, f"{safe_batch_name}_{condition.split()[0]}_{model_name}_model.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"    Plot saved: {os.path.basename(filepath)}")

    def _get_equation_text(self, model_name, metrics):
        # --- UPDATED EQUATION TEXT FOR POWER LAW ---
        model_name_lower = model_name.lower()
        if model_name_lower == 'linear':
            eq = r'$y = mx + c$'
        elif model_name_lower == 'logarithmic':
            eq = r'$y = a \cdot \ln(x + b) + c$'
        elif model_name_lower == 'power law':
            eq = r'$y = a \cdot x^b + c$'
        elif model_name_lower == 'exponential':
            eq = r'$y = a \cdot e^{bx} + c$'
        else:
            eq = 'Unknown Model'
            
        return f"Model: {eq}\n$R^2$   = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}"

    def create_summary_tables(self):
        print(f"\n{'='*80}\nDEGRADATION ANALYSIS SUMMARY\n{'='*80}")
        for model_type in self.model_names:
            print(f"\n{model_type.upper()} MODEL RESULTS:\n{'-'*60}")
            results_data = []
            for batch in self.batches:
                row = {"Batch": batch}
                for cond, models in [("Room", self.room_models), ("Ref", self.ref_models)]:
                    metrics = models.get(batch, {}).get(model_type, {}).get('metrics', {})
                    row[f"{cond} Days to 50%"] = models.get(batch, {}).get(model_type, {}).get('days_to_50', 'N/A')
                    row[f"{cond} R²"] = metrics.get('r2', 'N/A')
                    row[f"{cond} MSE"] = metrics.get('mse', 'N/A')
                    row[f"{cond} RMSE"] = metrics.get('rmse', 'N/A')
                results_data.append(row)
            
            results_df = pd.DataFrame(results_data)
            for col in results_df.columns:
                first_valid = results_df[col].dropna().iloc[0] if not results_df[col].dropna().empty else None
                if isinstance(first_valid, (int, float)):
                    results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

            print(results_df.to_string(index=False))
            results_df.to_csv(os.path.join(self.results_dir, f'summary_{model_type}.csv'), index=False)
            print(f"Results saved to 'summary_{model_type}.csv'")

    def create_performance_heatmaps(self):
        print(f"\n{'='*60}\nCREATING PERFORMANCE HEATMAPS\n{'='*60}")
        for condition, models_data, temp_name in [("Room", self.room_models, "Room Temperature"), ("Ref", self.ref_models, "Refrigerator Temperature")]:
            metrics_data = {'R²': np.full((len(self.model_names), len(self.batches)), np.nan), 'MSE': np.full((len(self.model_names), len(self.batches)), np.nan), 'RMSE': np.full((len(self.model_names), len(self.batches)), np.nan)}
            for i, model_name in enumerate(self.model_names):
                for j, batch_name in enumerate(self.batches):
                    metrics = models_data.get(batch_name, {}).get(model_name, {}).get('metrics')
                    if metrics:
                        metrics_data['R²'][i, j] = metrics.get('r2')
                        metrics_data['MSE'][i, j] = metrics.get('mse')
                        metrics_data['RMSE'][i, j] = metrics.get('rmse')
            
            self._plot_heatmap(metrics_data['R²'], 'Coefficient of Determination ($R^2$)', f'{temp_name}\n$R^2$ Performance', plt.cm.RdYlGn, f'heatmap_{condition}_R2.png')
            self._plot_heatmap(metrics_data['MSE'], 'Mean Squared Error (MSE)', f'{temp_name}\nMSE Performance', plt.cm.RdYlGn_r, f'heatmap_{condition}_MSE.png')
            self._plot_heatmap(metrics_data['RMSE'], 'Root Mean Squared Error (RMSE)', f'{temp_name}\nRMSE Performance', plt.cm.RdYlGn_r, f'heatmap_{condition}_RMSE.png')
        
    def _plot_heatmap(self, data, cbar_label, title, cmap, filename):
        fig, ax = plt.subplots(figsize=(10, 5))
        if np.isnan(data).all():
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([])
            plt.savefig(os.path.join(self.graphs_dir, filename), dpi=300); plt.close()
            return

        im = ax.imshow(data, cmap=cmap, aspect='auto')
        ax.set_xticks(np.arange(len(self.batches))); ax.set_yticks(np.arange(len(self.model_names)))
        ax.set_xticklabels(self.batches, fontsize=12, fontweight='bold')
        ax.set_yticklabels(self.model_names, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Batches', fontsize=14, fontweight='bold', labelpad=15)
        ax.set_ylabel('Models', fontsize=14, fontweight='bold', labelpad=15)

        for i in range(len(self.model_names)):
            for j in range(len(self.batches)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f'{data[i, j]:.3f}', ha="center", va="center", color="black", fontsize=16)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True); ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color='black', linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, filename), dpi=300); plt.close()
        print(f"Heatmap saved: {filename}")
    
    def generate_detailed_report(self):
        report_path = os.path.join(self.results_dir, "degradation_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("DEGRADATION ANALYSIS COMPREHENSIVE REPORT\n" + "="*60 + "\n\n")
            for cond, df_ in [("Room Temperature", self.room_df), ("Refrigerator Temperature", self.ref_df)]:
                if df_ is not None:
                    f.write(f"{cond} Data Summary:\n  Data points: {len(df_)}\n  Time range: {df_['Day'].min():.1f} - {df_['Day'].max():.1f} days\n\n")
            f.write("DETAILED RESULTS BY BATCH:\n" + "="*40 + "\n")
            for batch in self.batches:
                f.write(f"\n{batch}:\n" + "-"*len(batch) + "-\n")
                for cond, models in [("Room Temp", self.room_models), ("Ref. Temp", self.ref_models)]:
                    f.write(f"  {cond}:\n")
                    batch_models = models.get(batch, {})
                    if batch_models:
                        for name, data in batch_models.items():
                            days_str = f"{data['days_to_50']:.2f}" if isinstance(data['days_to_50'], float) else data['days_to_50']
                            m = data['metrics']
                            f.write(f"    {name:<12}: R²={m['r2']:.4f}, MSE={m['mse']:.4f}, RMSE={m['rmse']:.4f}, Days to 50% = {days_str}\n")
                    else: f.write("    No valid models\n")
        print(f"Detailed report saved to: {report_path}")

    def run_complete_analysis(self):
        print("🚀 Starting complete degradation analysis workflow...")
        try:
            self._create_models_for_condition('room')
            self._create_models_for_condition('ref')
            self.create_summary_tables()
            self.create_performance_heatmaps()
            self.generate_detailed_report()
            
            print(f"\n{'='*60}\n✅ ANALYSIS COMPLETED SUCCESSFULLY!\n{'='*60}")
            print(f"All outputs saved to the 'graphs' and 'results' sub-folders in: {self.output_dir}")
        except Exception as e:
            print(f"\n❌ ERROR DURING ANALYSIS: {e}")
            import traceback
            traceback.print_exc()

def main():
    print("="*60 + "\nDEGRADATION ANALYSIS & EXTRAPOLATION TOOL\n" + "="*60)
    print("This tool automatically generates predictive models, graphs, and performance heatmaps.")
    room_file = input("Enter the path to the ROOM temperature data Excel file: ").strip('"')
    ref_file = input("Enter the path to the REFRIGERATOR temperature data Excel file: ").strip('"')
    
    try:
        analysis = DegradationAnalysis(room_file, ref_file)
        analysis.run_complete_analysis()
        print(f"\n🎉 Analysis complete! Check the 'graphs' and 'results' folders.")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n❌ An error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()