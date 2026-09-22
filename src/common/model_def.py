"""
This module defines the PyTorch model architecture once. Training,
evaluation, and serving all import it from here, so there is exactly one
definition to keep in sync.
"""

import torch.nn as nn


class HeartDiseaseNN(nn.Module):
    """
    This is a deep neural network for heart disease prediction.

    The network takes 19 input features and passes them through three
    hidden layers of sizes 128, 64, and 32. Each hidden layer applies
    batch normalization and dropout. The final layer outputs 2 classes
    for binary classification.
    """

    def __init__(self, input_size=19, dropout_rate=0.3):
        super(HeartDiseaseNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x
