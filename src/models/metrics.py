import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DryDayLoss(nn.Module):
    def __init__(self, scaled_rain=None):
        super(DryDayLoss, self).__init__()
        self.rain_level = scaled_rain
        self.scale = 50.0

    def set_scaled_rain(self, scaled_rain):
        self.rain_level = scaled_rain

    def forward(self, output, target):
        output_thresh = torch.sigmoid(self.scale*(output-self.rain_level)).mean()
        
        target_thresh = torch.sigmoid(self.scale*(target-self.rain_level)).mean()
        return torch.abs(output_thresh - target_thresh)
    
class MaximumPrecipLoss(nn.Module):
    def __init__(self, scaled_rain=None):
        super(MaximumPrecipLoss, self).__init__()
        self.rain_level = scaled_rain

    def set_scaled_rain(self, scaled_rain):
        self.rain_level = scaled_rain

    def forward(self, output, target):
        if (output > self.rain_level).sum() < 5 or (target > self.rain_level).sum() < 5:
            output_mean = output.mean()
            output_std = output.std()
            output = output_mean + 2 * output_std

            target_mean = target.mean()
            target_std = target.std()
            target = target_mean + 2 * target_std
            return torch.abs(output - target)
        
        output_mean = output[output > self.rain_level].mean()
        output_std = output[output > self.rain_level].std()
        output = output_mean + 1 * output_std

        target_mean = target[output > self.rain_level].mean()
        target_std = target[output > self.rain_level].std()
        target = target_mean + 1 * target_std
        return torch.abs(output - target)
    
class AverageRainfallLoss(nn.Module):
    def __init__(self, scaled_rain=None):
        self.rain_level = scaled_rain
        super(AverageRainfallLoss, self).__init__()

    def set_scaled_rain(self, scaled_rain):
        self.rain_level = scaled_rain

    def forward(self, output, target):
        if (output > self.rain_level).sum()/float(len(output)) < 0.05 or (target > self.rain_level).sum()/float(len(target))/float(len(output)) < 0.05:
            output = output.mean()
            target = target.mean()
            return torch.abs(output - target)
        output = output[output > self.rain_level].mean()
        target = target[target > self.rain_level].mean()
        return torch.abs(output - target)
    
class RainfallVariance(nn.Module):
    def __init__(self, scaled_rain=None):
        self.rain_level = scaled_rain
        super(RainfallVariance, self).__init__()

    def set_scaled_rain(self, scaled_rain):
        self.rain_level = scaled_rain

    def forward(self, output, target):
        if (output > self.rain_level).sum()/float(len(output)) < 0.05 or (target > self.rain_level).sum()/float(len(output)) < 0.05:
            output = output.var()
            target = target.var()
            return torch.abs(output - target)

        output = output[output > self.rain_level].var()
        target = target[target > self.rain_level].var()
        return torch.abs(output - target)

class MonthlyAverageLoss(nn.Module):
    def __init__(self, months, scaled_rain=None):
        super(MonthlyAverageLoss, self).__init__()
        self.rain_level = scaled_rain
        self.months = months

    def set_scaled_rain(self, scaled_rain):
        self.rain_level = scaled_rain

    def set_monthly_data(self, months):
        self.months = months

    def month_data(self, input_data):
        # Ensure months array matches input data length
        if len(self.months) != len(input_data):
            # Take only the first len(input_data) elements from months
            months_subset = self.months[:len(input_data)]
        else:
            months_subset = self.months
            
        month_indices = torch.from_numpy(months_subset).to(device).to(torch.int64).view(-1, 1)
        mean = scatter_mean(input_data.squeeze(), month_indices.squeeze().to(input_data.device))
        return mean 

    def forward(self, output, target):
        monthly_output = self.month_data(output)
        monthly_target = self.month_data(target)
        diff = monthly_output - monthly_target
        monthly_loss = torch.mean(diff**2)
        return monthly_loss

class MonthlyMaxLoss(nn.Module):
    def __init__(self, months, scaled_rain=None):
        super(MonthlyMaxLoss, self).__init__()
        self.rain_level = scaled_rain
        self.months = months

    def set_scaled_rain(self, scaled_rain):
        self.rain_level = scaled_rain

    def set_monthly_data(self, months):
        self.months = months

    def month_data(self, input_data):
        # Ensure months array matches input data length
        if len(self.months) != len(input_data):
            # Take only the first len(input_data) elements from months
            months_subset = self.months[:len(input_data)]
        else:
            months_subset = self.months
            
        month_indices = torch.from_numpy(months_subset).to(device).to(torch.int64).view(-1, 1)
        mean, _ = scatter_max(input_data.squeeze(), month_indices.squeeze().to(input_data.device))
        return mean

    def forward(self, output, target):
        monthly_output = self.month_data(output)
        monthly_target = self.month_data(target)
        monthly_loss = torch.mean((monthly_output - monthly_target)**2)
        return monthly_loss
    
class MonthlySTDLoss(nn.Module):
    def __init__(self, months, scaled_rain=None):
        super(MonthlySTDLoss, self).__init__()
        self.rain_level = scaled_rain
        self.months = months

    def month_data(self, input_data):
        device = input_data.device
        month_indices = torch.from_numpy(self.months).to(device).long().view(-1, 1)

        # Step 1: Accumulate sum and count for each month
        scatter_sum = torch.zeros(np.max(self.months)+1, dtype=torch.float32, device=device)
        scatter_count = torch.zeros(np.max(self.months)+1, dtype=torch.float32, device=device)


        raining_indicies = torch.ones_like(input_data.squeeze())
        raining_indicies[input_data.squeeze() < self.rain_level] = 0
        scatter_sum.index_add_(0, month_indices.squeeze(), input_data.squeeze() * raining_indicies)
        scatter_count.index_add_(0, month_indices.squeeze(), torch.ones_like(input_data.squeeze()))

        # Step 2: Compute mean per month (handling empty counts gracefully)
        mean_result = torch.where(scatter_count > 0, scatter_sum / scatter_count, torch.zeros_like(scatter_sum))

        # Step 3: Compute squared deviations from the mean
        expanded_mean = mean_result[month_indices.squeeze()]
        squared_deviation = (input_data.squeeze() - expanded_mean) ** 2

        # Step 4: Sum the squared deviations for each month
        scatter_squared_dev = torch.zeros(np.max(self.months)+1, dtype=torch.float32, device=device)
        scatter_squared_dev.index_add_(0, month_indices.squeeze(), squared_deviation*raining_indicies)

        # Step 5: Compute variance per month (handle empty counts)
        variance_result = torch.where(
            scatter_count > 0, 
            scatter_squared_dev / scatter_count, 
            torch.zeros_like(scatter_squared_dev)
        )
        return torch.sqrt(variance_result)

    def forward(self, output, target):
        # Compute ECDF for output and target
        monthly_output = self.month_data(output)
        monthly_target = self.month_data(target)

        # Compute the difference
        monthly_loss = torch.mean((monthly_output - monthly_target)**2)

        # Return the maximum difference between ECDFs
        return monthly_loss