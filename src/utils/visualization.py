import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_results(experiment_name):
    """Generate plots from processed data"""
    # Load data
    daily_data = pd.read_csv(f'data/processed/{experiment_name}_daily.csv', index_col='date', parse_dates=True)
    monthly_data = pd.read_csv(f'data/processed/{experiment_name}_monthly.csv', index_col='date', parse_dates=True)
    cdfs = pd.read_csv(f'data/processed/{experiment_name}_cdfs.csv')
    
    # Create output directory
    save_path = f'results/figures/{experiment_name}'
    os.makedirs(save_path, exist_ok=True)
    
    # Monthly averages plot
    plot_monthly_averages(monthly_data, save_path)
    
    # Daily data plot
    plot_daily_data(daily_data, save_path)
    
    # CDFs plot
    plot_cdfs(cdfs, save_path)
    
    # Scatter plot
    plot_scatter(daily_data, save_path)

def plot_monthly_averages(monthly_data, save_path):
    plt.figure(figsize=(10, 6))
    plt.title('Monthly averages')
    for col in ['total_target', 'pred', 'total_input']:
        plt.plot(monthly_data.index, monthly_data[col], label=col.replace('total_', '').title())
    plt.xlabel('Date')
    plt.ylabel('Precipitation')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/monthly_averages.png')
    plt.close()

def plot_daily_data(daily_data, save_path):
    plt.figure(figsize=(10, 6))
    plt.title('Model output')
    for col in ['total_target', 'pred', 'total_input']:
        plt.plot(daily_data.index, daily_data[col], label=col.replace('total_', '').title())
    plt.xlabel('Date')
    plt.ylabel('Precipitation')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/data.png')
    plt.close()

def plot_cdfs(cdfs, save_path):
    plt.figure(figsize=(10, 6))
    plt.title('Cumulative Distribution Functions')
    for name in ['target', 'pred', 'input']:
        plt.plot(cdfs[f'total_{name}_x'], cdfs[f'total_{name}_y'], 
                label=name.title())
    plt.xlabel('Precipitation')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.xlim(0, 15)
    plt.tight_layout()
    plt.savefig(f'{save_path}/cdfs.png')
    plt.close()

def plot_scatter(data, save_path):
    plt.figure(figsize=(10, 6))
    correlation = data['pred'].corr(data['total_input'])
    plt.title(f'Scatter plot of Corrected vs Input (r = {correlation:.2f})')
    plt.scatter(data['total_input'], data['pred'], alpha=0.5)
    plt.xlabel('Input')
    plt.ylabel('Corrected')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}/scatter.png')
    plt.close() 