import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, window_size, num_variables, num_outputs):
        super(LSTMModel, self).__init__()
        self.norm_layer = nn.BatchNorm1d(num_variables) 
        self.lstm = nn.LSTM(input_size=num_variables, hidden_size=112, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(112, num_outputs)

    def forward(self, x):
        x = self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :]) 
        x = torch.tanh(self.fc(x))
        return x
    
    
class StackedLSTMModel(nn.Module):
    def __init__(self, window_size, num_variables, num_outputs):
        super(StackedLSTMModel, self).__init__()
        self.norm_layer = nn.BatchNorm1d(num_variables)
        self.conv1 = nn.Conv1d(num_variables, 96, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(input_size=96, hidden_size=64, batch_first=True, return_sequences=True)
        self.dropout1 = nn.Dropout(0.15)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=2)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        self.dropout2 = nn.Dropout(0.15)
        self.fc = nn.Linear(16, num_outputs)

    def forward(self, x):
        x = self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        x = torch.tanh(self.fc(x))
        return x


