import torch
import torch.nn as nn


class base_NN(nn.Module):
    """Simple three-layer convolutional neural network for MNIST (28x28 grayscale)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 28x28 -> 13x13
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 13x13 -> 5x5
        )
        self.classifier = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def get_weights(self)-> dict:
        weights = self.state_dict()
        weights_dict = {}
        for layer in weights.keys():
            weights_dict[layer] = weights[layer].detach().tolist()
        return weights_dict

    def load_weights(self, weights_dict: dict):
        weights = {}
        for layer in weights_dict.keys():
            weights[layer] = torch.Tensor(weights_dict[layer])
        self.load_state_dict(weights)
