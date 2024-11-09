import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base import BiasCorrector

class LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM and fully connected layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)

        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = torch.relu(out + 0.1) # Ensure positive output
        return out

class LSTM_BiasCorrector(BiasCorrector):
    def __init__(self, train_input, ground_truth, losses, loss_coef, 
                 input_size=1, hidden_size=50, output_size=1, num_layers=1, lr=0.001):
        super().__init__(train_input, ground_truth)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = LSTM_Model(input_size, hidden_size, output_size, num_layers).to(self.device)
        self.input_size = input_size
        
        # Loss and optimizer setup
        self.loss_coeff = loss_coef
        self.losses = losses
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_data = [[] for _ in range(len(losses))]

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def train(self, epochs=1000):
        self.model.train()
        
        # Convert data to tensors
        target = torch.from_numpy(self.ground_truth).view(-1, 1).float().to(self.device)
        train_input = torch.from_numpy(self.train_input).view(-1, self.input_size).float().to(self.device)

        print("Training LSTM...")
        best_loss = float('inf')
        stuck_counter = 0
        stuck_threshold = 100
        improvement_threshold = 1e-4
        prev_loss = float('inf')

        for epoch in range(epochs):
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(train_input)
            loss = self.evaluate_loss(outputs, target, save_to_array=True)
            
            # Backward pass
            loss.backward()
            target = target.view(-1, 1)
            self.optimizer.step()

            # Early stopping checks
            if abs(prev_loss - loss.item()) < improvement_threshold:
                stuck_counter += 1
            else:
                stuck_counter = 0
            prev_loss = loss.item()

            if stuck_counter >= stuck_threshold:
                print(f"Training stuck at epoch {epoch+1}. Breaking...")
                break

            # Learning rate decay
            if (epoch + 1) % 100 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.9

            # Progress logging
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                for i, individual_loss in enumerate(self.loss_data):
                    print(f'Loss {i+1}: {individual_loss[-1]:.4f}')

    def predict(self, test_data):
        test_data = torch.FloatTensor(test_data).view(-1, self.input_size).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_data).cpu().numpy()
        return predictions
    
    def evaluate_loss(self, x, y, save_to_array=False):
        total_loss = 0
        for loss, coef in zip(self.losses, self.loss_coeff):
            individual_loss = loss(x, y) * coef
            total_loss += individual_loss
            if save_to_array:
                self.loss_data[self.losses.index(loss)].append(individual_loss.item())
        return total_loss
