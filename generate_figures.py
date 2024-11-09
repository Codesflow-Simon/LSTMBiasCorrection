import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_double_log_cdf(data, save_path):
    """Create double logarithmic CDF plot"""
    plt.figure(figsize=(10, 6))
    for col in ['target', 'pred', 'input']:
        # Sort data and calculate empirical CDF
        sorted_data = np.sort(data[col])
        # Remove zeros for log scale
        nonzero_mask = sorted_data > 0
        sorted_data = sorted_data[nonzero_mask]
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        plt.loglog(sorted_data, 1 - p, label=col.title())
    
    plt.xlabel('Precipitation (mm/day)')
    plt.ylabel('Exceedance Probability')
    plt.title('Double Logarithmic CDF Plot')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/double_log_cdf.png')
    plt.close()

def plot_model_boxplots(all_results, save_dir):
    """Create boxplots comparing models across datasets"""
    # Prepare data for plotting
    plot_data = []
    for model_name, model_results in all_results.items():
        for dataset, metrics in model_results.items():
            if 'failed' not in metrics:
                plot_data.append({
                    'Model': model_name,
                    'Dataset': dataset,
                    'RMSE': metrics['validation']['rmse'],
                    'Max CDF Distance': metrics['validation']['max_cdf_distance'],
                    'Correlation': metrics['validation']['correlation']
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create boxplots for each metric
    metrics = ['RMSE', 'Max CDF Distance', 'Correlation']
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
    # Prepare data for plotting
    metrics = ['rmse', 'max_cdf_distance', 'correlation']
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
        
        # Create heatmap
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
            
            # Load data
            data = pd.read_csv(os.path.join(processed_dir, filename))
            
            # Generate double log CDF plot
            plot_double_log_cdf(data, save_path)
            
            # Basic plots
            # Monthly averages
            monthly_data = data.set_index('date').resample('ME').mean()
            plt.figure(figsize=(12, 6))
            for col in ['target', 'pred', 'input']:
                plt.plot(monthly_data.index, monthly_data[col], label=col.title())
            plt.xlabel('Date')
            plt.ylabel('Precipitation (mm/day)')
            plt.title('Monthly Averages')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{save_path}/monthly_averages.png')
            plt.close()
            
            # Scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(data['input'], data['pred'], alpha=0.5)
            plt.xlabel('Input')
            plt.ylabel('Prediction')
            max_val = max(data['input'].max(), data['pred'].max())
            plt.plot([0, max_val], [0, max_val], 'r--')
            plt.title('Prediction vs Input')
            plt.tight_layout()
            plt.savefig(f'{save_path}/scatter.png')
            plt.close()

if __name__ == "__main__":
    main() 