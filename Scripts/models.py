import torch
import torch.nn as nn


class RasterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 16 * 16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StrokeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StrokeRNN(nn.Module):
    def __init__(self, num_classes, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input from dataset is (B, 3, L) for consistency with Conv1d.
        # LSTM wants (B, L, 3), so transpose.
        x = x.transpose(1, 2)  # (B, L, 3)

        out, (h_n, c_n) = self.rnn(x)

        # Use final hidden state from last layer
        last_hidden = h_n[-1]  # (B, hidden_size)
        logits = self.classifier(last_hidden)
        return logits
