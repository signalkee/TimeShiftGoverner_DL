import torch
import torch.nn as nn
import math

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
        self.lstm1 = nn.LSTM(input_size=96, hidden_size=64, batch_first=True)
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


import torch.nn.functional as F
class TransformerModel(nn.Module):
    def __init__(self, window_size, num_variables, num_outputs, d_model=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.15):
        super(TransformerModel, self).__init__()
        self.num_variables = num_variables
        self.window_size = window_size

        self.embedding = nn.Linear(num_variables, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        transformer_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, num_layers)
        self.fc_out = nn.Linear(d_model, num_outputs)
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiheadAttentionModel(nn.Module):
    def __init__(self, window_size, num_variables, num_outputs, hidden_size=112, nhead=8, dropout=0.25):
        super(MultiheadAttentionModel, self).__init__()
        self.norm_layer = nn.BatchNorm1d(num_variables)

        # Embedding layer for multihead attention
        self.embedding = nn.Linear(num_variables, hidden_size)

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nhead)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        x = self.embedding(x)

        # Apply MultiheadAttention
        attn_output, _ = self.attention(x, x, x)

        # Apply LSTM
        lstm_output, _ = self.lstm(attn_output)
        lstm_output = self.dropout(lstm_output[:, -1, :])

        # Final fully connected layer
        out = torch.tanh(self.fc(lstm_output))
        return out