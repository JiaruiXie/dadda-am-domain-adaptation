import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):

    def __init__(self):

        super(DomainClassifier, self).__init__()

        code_size = 20

        self.fc4 = nn.Linear(
            code_size,
            48,
        )

        self.fc5 = nn.Linear(
            48,
            32,
        )

        self.fc6 = nn.Linear(
            32,
            32,
        )

        self.fc7 = nn.Linear(
            32,
            1,
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc4(x)

        x = self.relu(x)

        x = self.fc5(x)

        x = self.relu(x)

        x = self.fc6(x)

        x = self.relu(x)

        x = self.fc7(x)

        x = torch.sigmoid(x)

        return x