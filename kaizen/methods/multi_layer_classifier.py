import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, layer_units=[]):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layer_units = layer_units

        in_dim = self.input_dim
        self.all_layers = []
        for i, num_units in enumerate(layer_units):
            layer_name = f"fc_{i}"
            layer = nn.Linear(in_dim, num_units)
            in_dim = num_units
            setattr(self, layer_name, layer)
            self.all_layers.append(layer)
        self.fc_output = nn.Linear(in_dim, self.num_classes)

    def forward(self, x):
        for l in self.all_layers:
            x = F.relu(l(x))
        return self.fc_output(x)


