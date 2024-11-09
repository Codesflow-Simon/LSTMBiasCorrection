import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import os
import json
import torch.nn as nn
from src.models.lstm import LSTM_BiasCorrector
from src.models.base import QuantileMapping
from src.models.metrics import (
    DryDayLoss, 
    MaximumPrecipLoss, 
    AverageRainfallLoss, 
    RainfallVariance, 
    MonthlyMaxLoss, 
    MonthlyAverageLoss
)

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
    
    return model, train_input, train_target, valid_input, valid_target

def save_results(model, train_column, experiment_name, train_input, train_target, valid_input, valid_target):
    """Save both training and validation data"""
    # Get predictions
    train_pred = model.predict(train_input)
    valid_pred = model.predict(valid_input)
    
    # Create training dataframe
    train_data = pd.DataFrame({
        'input': train_input.flatten(),
        'target': train_target.flatten(),
        'pred': train_pred.flatten(),
        'set': 'train'
    })
    train_data['date'] = pd.date_range(start='1/1/1990', periods=len(train_data), freq='D')
    
    # Create validation dataframe
    valid_data = pd.DataFrame({
        'input': valid_input.flatten(),
        'target': valid_target.flatten(),
        'pred': valid_pred.flatten(),
        'set': 'validation'
    })
    valid_data['date'] = pd.date_range(start=train_data.date.max() + pd.Timedelta(days=1), 
                                      periods=len(valid_data), freq='D')
    
    # Combine datasets
    all_data = pd.concat([train_data, valid_data])
    all_data.set_index('date', inplace=True)
    
    # Save training data if available (for LSTM models)
    if hasattr(model, 'loss_data'):
        training_data = {
            'loss_values': model.loss_data,
            'loss_weights': model.loss_coeff,
            'loss_names': [loss.__class__.__name__ for loss in model.losses]
        }
        with open(f'data/processed/{experiment_name}_training.json', 'w') as f:
            json.dump(training_data, f, indent=4)
    
    # Save essential data
    os.makedirs('data/processed', exist_ok=True)
    all_data.to_csv(f'data/processed/{experiment_name}_data.csv')
    
    # Calculate metrics for both sets
    metrics = {}
    for dataset in ['train', 'validation']:
        data_subset = all_data[all_data['set'] == dataset]
        metrics[dataset] = {
            'correlation': float(data_subset['pred'].corr(data_subset['input'])),
            'zero_pred_ratio': float(np.mean(data_subset['pred'] == 0)),
            'rmse': float(np.sqrt(np.mean((data_subset['pred'] - data_subset['target'])**2)))
        }
        
        # Add CDF-based metrics
        target_cdf_x, target_cdf_y = calculate_cdf(data_subset['target'])
        pred_cdf_x, pred_cdf_y = calculate_cdf(data_subset['pred'])
        common_x = np.unique(np.concatenate((target_cdf_x, pred_cdf_x)))
        target_interp = np.interp(common_x, target_cdf_x, target_cdf_y)
        pred_interp = np.interp(common_x, pred_cdf_x, pred_cdf_y)
        
        metrics[dataset].update({
            'max_cdf_distance': float(np.max(np.abs(target_interp - pred_interp))),
            'area_between_cdfs': float(np.trapz(np.abs(target_interp - pred_interp), common_x))
        })
    
    with open(f'data/processed/{experiment_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return all_data, metrics

def evaluate_model(model_instance, train_input, train_target, valid_input, valid_target, train_column, experiment_name):
    """Evaluate model and save results"""
    # Save results data
    data, metrics = save_results(
        model_instance, 
        train_column, 
        experiment_name,
        train_input, train_target,  # Add training data
        valid_input, valid_target
    )
    
    # Model rejection criteria using validation metrics
    pred = model_instance.predict(valid_input)
    zero_pred_ratio = np.mean(pred == 0)
    constant_pred = np.allclose(pred, pred[0], atol=1e-5)
    correlation = metrics['validation']['correlation']
    max_dist = metrics['validation']['max_cdf_distance']
    
    model_rejected = (
        zero_pred_ratio > 0.99 or
        constant_pred or
        max_dist > 0.9
    )
    
    if model_rejected:
        print(f"Model rejected. Metrics: max_dist={max_dist:.4f}, correlation={correlation:.4f}, zero_pred_ratio={zero_pred_ratio:.4f}")
    
    return not model_rejected, metrics

def main():
    # Load data
    df = pd.read_csv('data/raw/221212.csv')
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
                    model_instance, train_input, train_target, valid_input, valid_target = set_model_on_data(
                        train_df, validation_df, train_column, gt_column,
                        model['losses'], model['weights'], use_QM=False
                    )
                    print(f"Training on {model_instance.device}")
                    model_instance.train(epochs=3000)
                else:  # QM
                    model_instance, train_input, train_target, valid_input, valid_target = set_model_on_data(
                        train_df, validation_df, train_column, gt_column,
                        losses=None, weights=None, use_QM=True
                    )
                
                if model['type'] == 'LSTM':
                    for loss in model['losses']:
                        if isinstance(loss, (MonthlyAverageLoss, MonthlyMaxLoss)):
                            validation_months = validation_df['months'].values
                            loss.set_monthly_data(validation_months)
                            
                experiment_name = f"{train_column}_{model['name']}"
                model_accepted, metrics = evaluate_model(
                    model_instance,
                    train_input,
                    train_target,
                    valid_input,
                    valid_target,
                    train_column,
                    experiment_name
                )
                
                attempt += 1
            
            if model_accepted:
                print(f"Model accepted after {attempt} attempts")
                results[model['name']][train_column] = metrics
            else:
                print(f"Warning: Model not accepted after {max_attempts} attempts.")
                results[model['name']][train_column] = {'failed': True}

    # Save overall results
    with open('data/processed/overall_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary
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
