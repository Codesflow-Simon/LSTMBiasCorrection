import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cdfs(data, numeric_data, save_path):
    """Create both regular and double logarithmic CDF plots"""
    # Filter for validation data
    val_data = numeric_data[data['set'] == 'validation']
    
    # Create regular CDF plot
    plt.figure(figsize=(10, 6))
    for col in ['target', 'pred', 'input']:
        # Sort data in ascending order
        sorted_data = np.sort(val_data[col])
        # Calculate CDF (not exceedance)
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        plt.plot(sorted_data, p, label=col.title())
    
    plt.xlabel('Precipitation (mm/day)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF Plot')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/cdf.png')
    plt.close()
    
    # Create double logarithmic CDF plot
    plt.figure(figsize=(10, 6))
    for col in ['target', 'pred', 'input']:
        # Sort data in ascending order
        sorted_data = np.sort(val_data[col])
        nonzero_mask = sorted_data > 0
        sorted_data = sorted_data[nonzero_mask]
        # Calculate CDF (not exceedance)
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        plt.loglog(sorted_data, p, label=col.title())
    
    plt.xlabel('Precipitation (mm/day)')
    plt.ylabel('Cumulative Probability')
    plt.title('Double Logarithmic CDF Plot')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/double_log_cdf.png')
    plt.close()

def plot_zoomed_timeseries(data, numeric_data, save_path):
    """Plot full time series with zoomed section"""
    # Filter for validation data
    val_data = numeric_data[data['set'] == 'validation']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[1, 1.5])
    
    # Plot full time series
    for col in ['target', 'pred', 'input']:
        ax1.plot(val_data.index, val_data[col], label=col.title(), alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Precipitation (mm/day)')
    ax1.set_title('Full Time Series')
    ax1.legend()
    
    # Get middle 3 months of validation data for zoom
    mid_point = len(val_data) // 2
    window = 45  # ~3 months
    zoom_data = val_data.iloc[mid_point-window:mid_point+window]
    
    # Plot zoomed section
    for col in ['target', 'pred', 'input']:
        ax2.plot(zoom_data.index, zoom_data[col], label=col.title(), 
                marker='o', markersize=3, alpha=0.7)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Precipitation (mm/day)')
    ax2.set_title('Zoomed View (3 Months)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/timeseries_zoom.png')
    plt.close()

def plot_model_boxplots(all_results, save_dir):
    """Create boxplots comparing models across datasets"""
    plot_data = []
    for model_name, model_results in all_results.items():
        for dataset, metrics in model_results.items():
            if 'failed' not in metrics:
                plot_data.append({
                    'Model': model_name,
                    'Dataset': dataset,
                    'Area Between CDFs': metrics['validation']['area_between_cdfs'],
                    'Max CDF Distance': metrics['validation']['max_cdf_distance'],
                    'Correlation': metrics['validation']['correlation']
                })
    
    df = pd.DataFrame(plot_data)
    
    # Updated metrics list (removed RMSE, added Area Between CDFs)
    metrics = ['Area Between CDFs', 'Max CDF Distance', 'Correlation']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Model', y=metric)
        plt.xticks(rotation=45)
        plt.title(f'{metric} by Model')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/boxplot_{metric.lower().replace(" ", "_")}.png')
        plt.close()

def plot_dataset_comparison(all_results, save_dir):
    """Create heatmap of model performance across datasets"""
    # Updated metrics list (removed rmse, added area_between_cdfs)
    metrics = ['area_between_cdfs', 'max_cdf_distance', 'correlation']
    for metric in metrics:
        data = {}
        for model_name, model_results in all_results.items():
            data[model_name] = []
            for dataset, results in model_results.items():
                if 'failed' not in results:
                    data[model_name].append(results['validation'][metric])
                else:
                    data[model_name].append(np.nan)
        
        df = pd.DataFrame(data, index=list(all_results[list(all_results.keys())[0]].keys()))
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0)
        plt.title(f'{metric.upper()} across Models and Datasets')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/heatmap_{metric}.png')
        plt.close()

def main():
    # Load overall results
    with open('data/processed/overall_results.json', 'r') as f:
        all_results = json.load(f)
    
    # Create output directory for comparison plots
    os.makedirs('results/figures/comparisons', exist_ok=True)
    
    # Generate comparison plots
    plot_model_boxplots(all_results, 'results/figures/comparisons')
    plot_dataset_comparison(all_results, 'results/figures/comparisons')
    
    # Process individual experiments
    processed_dir = 'data/processed'
    for filename in os.listdir(processed_dir):
        if filename.endswith('_data.csv'):
            experiment_name = filename.replace('_data.csv', '')
            print(f"Processing {experiment_name}")
            
            # Create output directory
            save_path = f'results/figures/{experiment_name}'
            os.makedirs(save_path, exist_ok=True)
            
            # Load data with proper date parsing
            data = pd.read_csv(os.path.join(processed_dir, filename), parse_dates=['date'])
            data.set_index('date', inplace=True)
            
            # Remove 'set' column before resampling
            numeric_columns = ['input', 'target', 'pred']
            data_numeric = data[numeric_columns]
            
            # Generate plots with validation data only
            plot_cdfs(data, data_numeric, save_path)
            plot_zoomed_timeseries(data, data_numeric, save_path)
            
            # Monthly averages
            val_data = data_numeric[data['set'] == 'validation']
            monthly_data = val_data.resample('ME').mean()
            plt.figure(figsize=(12, 6))
            for col in numeric_columns:
                plt.plot(monthly_data.index, monthly_data[col], label=col.title())
            plt.xlabel('Date')
            plt.ylabel('Precipitation (mm/day)')
            plt.title('Monthly Averages (Validation Set)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{save_path}/monthly_averages.png')
            plt.close()
            
            # Scatter plot (validation only)
            plt.figure(figsize=(12, 6))
            plt.scatter(val_data['input'], 
                       val_data['pred'], 
                       alpha=0.5, 
                       label='validation')
            plt.xlabel('Input')
            plt.ylabel('Prediction')
            max_val = max(val_data['input'].max(), val_data['pred'].max())
            plt.plot([0, max_val], [0, max_val], 'r--')
            plt.title('Prediction vs Input (Validation Set)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_path}/scatter.png')
            plt.close()

if __name__ == "__main__":
    main() 