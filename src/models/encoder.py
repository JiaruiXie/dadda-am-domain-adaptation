import torch
import torch.nn as nn


class ConvolutionalEncoder(nn.Module):

    def __init__(self):

        super(ConvolutionalEncoder, self).__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.embedding_layer = nn.Sequential(

            nn.Linear(
                16 * 15 * 15,
                20,
            ),

            nn.BatchNorm1d(20),

            nn.ReLU(),
        )

    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.flatten(x)

        embedding = self.embedding_layer(x)

        return embedding