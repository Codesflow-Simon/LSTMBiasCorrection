import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import os
import torch.nn as nn
from torch_bias_corrector import LSTM_BiasCorrector
from base import QuantileMapping
from metric import DryDayLoss, MaximumPrecipLoss, AverageRainfallLoss, RainfallVariance, MonthlyMaxLoss, MonthlyAverageLoss

def calculate_cdf(data):
    """Calculate the Cumulative Distribution Function (CDF) for given data."""
    if len(data) == 0:
        raise ValueError("Input data is empty")
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if np.isnan(data).any():
        print("Warning: Input data contains NaN values. These will be ignored.")
        data = data[~np.isnan(data)]
    
    if len(data) == 0:
        raise ValueError("All input data was NaN")
    
    if np.isinf(data).any():
        print("Warning: Input data contains infinite values. These may affect the CDF calculation.")
    
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Handle zeros separately
    zero_prob = np.sum(sorted_data == 0) / len(sorted_data)
    if zero_prob > 0:
        sorted_data = np.insert(sorted_data, 0, 0)
        y = np.insert(y, 0, 0)
    
    return sorted_data, y

def plot_and_save(x, y, title, xlabel, ylabel, labels, filename, xlim=None):
    """Generic function to create and save plots."""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    for xi, yi, label in zip(x, y, labels):
        plt.plot(xi, yi, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def preprocess_data(df):
    """Preprocess the input dataframe."""
    # Convert 'dayno' to datetime
    df['date'] = pd.to_datetime('1989-12-31') + pd.to_timedelta(df['dayno'], unit='D')
    
    # Calculate the day of the year and its cyclical features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Split data into train and validation sets
    train_df = df[:int(len(df) * 0.8)]
    validation_df = df[int(len(df) * 0.8):]
    
    # Calculate months since the start of the dataset
    min_year = df['date'].dt.year.min()
    for dataset in [train_df, validation_df]:
        dataset['months'] = (dataset['date'].dt.year - min_year) * 12 + dataset['date'].dt.month - 1
    
    return train_df, validation_df

def prepare_model_input(df, column, include_cyclical=False):
    """Prepare input data for the model."""
    normaliser = StandardScaler(with_mean=False)
    data = normaliser.fit_transform(df[column].values.reshape(-1, 1))
    
    if include_cyclical:
        cyclical_features = df[['sin_day', 'cos_day']].values
        data = np.concatenate([data, cyclical_features], axis=1)
    
    return data, normaliser

def set_model_on_data(train_df, validation_df, train_column, gt_column, losses, weights, use_QM=False):
    """Set up and return the model based on the input data."""
    train_input, normaliser = prepare_model_input(train_df, train_column)
    train_target = normaliser.transform(train_df[gt_column].values.reshape(-1, 1))
    
    valid_input = normaliser.transform(validation_df[train_column].values.reshape(-1, 1))
    valid_target = normaliser.transform(validation_df[gt_column].values.reshape(-1, 1))
    
    min_rain = 0.1
    scaled_rain = normaliser.transform(np.array([[min_rain]])).item()
    
    for loss in losses:
        if hasattr(loss, 'set_scaled_rain'):
            loss.set_scaled_rain(scaled_rain)
    
    if use_QM:
        model = QuantileMapping(train_input, train_target)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTM_BiasCorrector(train_input, train_target,
                                   losses,
                                   weights,
                                   input_size=train_input.shape[1],
                                   lr=1e-3)
        model.set_device(device)
    
    return model, valid_input, valid_target

def make_figures(model, train_column, experiment_name, valid_input, valid_target):
    # Preprocessing
    
    pred = model.predict(valid_input)
    
    data = pd.DataFrame({
        'total_input': valid_input.flatten(),
        'total_target': valid_target.flatten(),
        'pred': pred.flatten()
    })
    
    # Create a date range for the validation period
    data['date'] = pd.date_range(start='1/1/2000', periods=len(data), freq='D')
    data.set_index('date', inplace=True)
    
    # Calculate monthly averages
    monthly_data = data.resample('M').mean()

    save_path = f'figures/{experiment_name}'
    os.makedirs(save_path, exist_ok=True)

    # Monthly averages plot
    plot_and_save(
        [monthly_data.index] * 3,
        [monthly_data['total_target'], monthly_data['pred'], monthly_data['total_input']],
        'Monthly averages',
        'Date',
        'Precipitation',
        ['Observed', 'Corrected', 'Input'],
        f'{save_path}/monthly_averages.png'
    )

    # Model output plot
    plot_and_save(
        [data.index] * 3,
        [data['total_target'], data['pred'], data['total_input']],
        'Model output',
        'Date',
        'Precipitation',
        ['Observed', 'Corrected', 'Input'],
        f'{save_path}/data.png'
    )

    # CDFs plot
    cdfs = [calculate_cdf(data[col]) for col in ['total_target', 'pred', 'total_input']]
    plot_and_save(
        [cdf[0] for cdf in cdfs],
        [cdf[1] for cdf in cdfs],
        'Cumulative Distribution Functions',
        'Precipitation',
        'Cumulative Probability',
        ['Observed', 'Corrected', 'Input'],
        f'{save_path}/cdfs.png',
        xlim=(0, 15)
    )

    # Calculate maximum distance between CDFs
    observed_cdf_x, observed_cdf_y = cdfs[0]
    corrected_cdf_x, corrected_cdf_y = cdfs[1]

    # Interpolate CDFs to a common x-axis
    common_x = np.unique(np.concatenate((observed_cdf_x, corrected_cdf_x)))
    observed_interp = np.interp(common_x, observed_cdf_x, observed_cdf_y)
    corrected_interp = np.interp(common_x, corrected_cdf_x, corrected_cdf_y)

    # Calculate distances
    distances = np.abs(observed_interp - corrected_interp)
    max_distance = np.max(distances)
    argmax_distance = np.argmax(distances)

    # Calculate area between CDFs
    area_between_cdfs = np.trapz(np.abs(observed_interp - corrected_interp), common_x)
    
    with open(f'{save_path}/cdf_distance.txt', 'w') as f:
        f.write(f'Maximum distance between target and prediction CDFs: {max_distance}\n')
        f.write(f'Argmax distance: {argmax_distance}\n')
        f.write(f'Area between CDFs: {area_between_cdfs}\n')
        f.write(f'Zero probability (observed): {observed_cdf_y[0]}\n')
        f.write(f'Zero probability (corrected): {corrected_cdf_y[0]}\n')

    # Scatter plot
    correlation = data['pred'].corr(data['total_input'])
    plt.figure(figsize=(10, 6))
    plt.title(f'Scatter plot of Corrected vs Input (r = {correlation:.2f})')
    plt.scatter(data['total_input'], data['pred'], alpha=0.5)
    plt.xlabel('Input')
    plt.ylabel('Corrected')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}/scatter.png')
    plt.close()

    # Validation losses
    if hasattr(model, 'losses'):
        losses = [loss(torch.Tensor(valid_input).to(model.device), 
                       torch.Tensor(valid_target).to(model.device)) 
                  for loss in model.losses]
        with open(f'{save_path}/validation_losses.txt', 'w') as f:
            for loss_name, value in zip([loss.__class__.__name__ for loss in model.losses], losses):
                f.write(f'{loss_name}: {value.item()}\n')
            
            if isinstance(model, LSTM_BiasCorrector):
                f.write(f"Loss coefficients: {model.loss_coeff}\n")
                f.write(f"Loss data: {np.array(model.loss_data)[:,0]}\n")

    # Loss plots
    if isinstance(model, LSTM_BiasCorrector):
        for i, loss in enumerate(model.losses):
            plt.figure(figsize=(10, 6))
            plt.title(f'{loss.__class__.__name__} over time')
            plt.plot(np.array(model.loss_data)[:, i])
            plt.ylabel(loss.__class__.__name__)
            plt.tight_layout()
            plt.savefig(f'{save_path}/losses_{loss.__class__.__name__}.png')
            plt.close()

    return max_distance, area_between_cdfs

def evaluate_model(model_instance, valid_input, valid_target, train_column, experiment_name):
    max_dist, area_between_cdfs = make_figures(model_instance, train_column, experiment_name, valid_input, valid_target)
    
    pred = model_instance.predict(valid_input)
    
    # Simple checks for very bad models
    zero_pred_ratio = np.mean(pred == 0)
    constant_pred = np.allclose(pred, pred[0], atol=1e-5)
    correlation = np.corrcoef(valid_target.flatten(), pred.flatten())[0, 1]
    
    # Model is rejected if it:
    # 1. Predicts all zeros (or very close to it)
    # 2. Predicts a constant value
    # 3. Has a negative correlation with the target
    # 4. Has a very large max CDF distance
    model_rejected = (
        zero_pred_ratio > 0.99 or
        constant_pred or
        correlation <= 0 or
        max_dist > 0.9
    )
    
    if model_rejected:
        print(f"Model rejected. Metrics: max_dist={max_dist:.4f}, correlation={correlation:.4f}, zero_pred_ratio={zero_pred_ratio:.4f}")
    
    return not model_rejected, {
        'max_dist': max_dist,
        'area_between_cdfs': area_between_cdfs,
        # 'correlation': correlation,
        'zero_pred_ratio': zero_pred_ratio
    }

def main():
    # Load data
    df = pd.read_csv('221212.csv')
    train_df, validation_df = preprocess_data(df)
    
    # Define the 6 models
    models = [
        
        {
            'name': 'LSTM_AllLosses',
            'type': 'LSTM',
            'losses': [DryDayLoss(), MaximumPrecipLoss(), AverageRainfallLoss(), RainfallVariance()],
            'weights': [0.5*1/0.36914465, 0.3*1/2.80799103, 1./0.4435252, 1/1.39767456]
        },
        {
            'name': 'LSTM_MonthlyLosses',
            'type': 'LSTM',
            'losses': [DryDayLoss(), MonthlyMaxLoss(train_df['months'].to_numpy()), MonthlyAverageLoss(train_df['months'].to_numpy()), RainfallVariance()],
            'weights': [0.5*1/0.36914465, 0.3*1/32.22470093, 1./0.26838824, 1/1.39767456]
        },
        {
            'name': 'LSTM_BasicLosses',
            'type': 'LSTM',
            'losses': [AverageRainfallLoss(), MaximumPrecipLoss()],
            'weights': [1/0.4435252, 0.3*1/2.80799103]
        },
        {
            'name': 'LSTM_MonthlyBasic',
            'type': 'LSTM',
            'losses': [MonthlyAverageLoss(train_df['months'].to_numpy()), MonthlyMaxLoss(train_df['months'].to_numpy())],
            'weights': [1/0.26838824, 0.3*1/32.22470093]
        },
        {
            'name': 'LSTM_MSE',
            'type': 'LSTM',
            'losses': [nn.MSELoss()],
            'weights': [1]
        },
        {
            'name': 'QuantileMapping',
            'type': 'QM',
            'losses': None,
            'weights': None
        }
    ]

    # Set the columns to train on
    train_columns = [
        'agg.raw_CCCMA3.1_R1_hist', 'agg.raw_CSIRO.MK3.0_R1_hist',
        'agg.raw_ECHAM5_R1_hist', 'agg.raw_MIROC3.2_R1_hist',
        'agg.raw_CCCMA3.1_R2_hist', 'agg.raw_CSIRO.MK3.0_R2_hist',
        'agg.raw_ECHAM5_R2_hist', 'agg.raw_MIROC3.2_R2_hist',
        'agg.raw_CCCMA3.1_R3_hist', 'agg.raw_CSIRO.MK3.0_R3_hist',
        'agg.raw_ECHAM5_R3_hist', 'agg.raw_MIROC3.2_R3_hist'
    ]

    gt_column = 'agg.AWAP'

    results = {}

    for model in models:
        results[model['name']] = {}
        for train_column in train_columns:
            print(f"Running {model['name']} on {train_column}")
            
            max_attempts = 10
            attempt = 0
            model_accepted = False
            
            while attempt < max_attempts and not model_accepted:
                if model['type'] == 'LSTM':
                    model_instance, valid_input, valid_target = set_model_on_data(
                        train_df, validation_df, train_column, gt_column,
                        model['losses'], model['weights'], use_QM=False
                    )
                    print(f"Training on {model_instance.device}")
                    model_instance.train(epochs=3000)
                else:  # QM
                    model_instance, valid_input, valid_target = set_model_on_data(
                        train_df, validation_df, train_column, gt_column,
                        losses=None, weights=None, use_QM=True
                    )
                # Check if there are any monthly loss functions and update them with validation months
                if model['type'] == 'LSTM':
                    for loss in model['losses']:
                        if isinstance(loss, (MonthlyAverageLoss, MonthlyMaxLoss)):
                            validation_months = validation_df['months'].values
                            loss.set_monthly_data(validation_months)
                            
                experiment_name = f"{train_column}_{model['name']}"
                model_accepted, metrics = evaluate_model(model_instance, valid_input, valid_target, train_column, experiment_name)
                
                attempt += 1
            
            if model_accepted:
                print(f"Model accepted after {attempt} attempts")
                results[model['name']][train_column] = metrics
            else:
                print(f"Warning: Model not accepted after {max_attempts} attempts.")
                results[model['name']][train_column] = {'failed': True}

    # Print summary of results
    print("\nSummary of Results:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for column, metrics in model_results.items():
            if 'failed' in metrics:
                print(f"  {column}: Failed to meet criteria")
            else:
                print(f"  {column}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
