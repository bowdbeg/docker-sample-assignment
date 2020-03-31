import torch
import torch.nn as nn


# Binary classifier with MLP
class BinaryClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_layer=1,
                 non_linear='relu',
                 threshold=0.5,
                 dropout_in=0.0,
                 dropout_out=0.0):
        super(BinaryClassifier, self).__init__()
        self.threshold = threshold
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.dropout_in = nn.Dropout(dropout_in)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size)] * hidden_layer)
        self.dropout_out = nn.Dropout(dropout_out)
        self.output_layer = nn.Linear(hidden_size, 1)
        if non_linear == 'relu':
            self.non_linear = torch.relu
        elif non_linear == 'tanh':
            self.non_linear = torch.tanh

    def forward(self, x):
        x = self.non_linear(self.input_layer(x))
        x = self.dropout_in(x)
        for hidden_layer in self.hidden_layers:
            x = self.non_linear(hidden_layer(x))
        x = self.dropout_out(x)
        x = torch.sigmoid(self.output_layer(x)).squeeze(-1)
        return x

    def predict(self, x):
        x = self.forward(x)
        return self.predict_from_output(x)

    def predict_from_output(self, output):
        return (output > self.threshold).to(torch.float)
