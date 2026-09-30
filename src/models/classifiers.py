import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskClassifier(nn.Module):

    def __init__(self):

        super(TaskClassifier, self).__init__()

        code_size = 20

        self.fc1 = nn.Linear(
            code_size,
            32,
        )

        self.fc2 = nn.Linear(
            32,
            32,
        )

        self.fc3 = nn.Linear(
            32,
            3,
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.bn2 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(
            self.bn1(x)
        )

        x = self.fc2(x)

        intermediate_features = self.relu(
            self.bn2(x)
        )

        logits = self.fc3(
            intermediate_features
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        return probabilities, intermediate_features


class TaskClassifier1(nn.Module):

    def __init__(self):

        super(TaskClassifier1, self).__init__()

        code_size = 20

        self.fc1 = nn.Linear(
            code_size,
            32,
        )

        self.fc2 = nn.Linear(
            32,
            32,
        )

        self.fc3 = nn.Linear(
            32,
            3,
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.bn2 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(
            self.bn1(x)
        )

        x = self.fc2(x)

        intermediate_features = self.relu(
            self.bn2(x)
        )

        logits = self.fc3(
            intermediate_features
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        return probabilities, intermediate_features