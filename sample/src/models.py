import torch
import torch.nn as nn


# Binary classifier with MLP
class BinaryClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_layers=1,
                 non_linear='relu'):
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size)] * hidden_layers)
        self.output_layer = nn.Linear(hidden_size, 1)
        if non_linear == 'relu':
            self.non_linear = torch.relu
        elif non_linear == 'tanh':
            self.non_linear = torch.tanh

    def forward(self, x):
        x = self.non_linear(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.non_linear(hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
