import torch.nn as nn

# Predicting std captures the underlying noise in the data (aleatoric)
# Monte Carlo dropout sampling samples the model weight uncertainty (epistemic uncertainty)

# Define a CNN that predicts both mean and log-variance (which can be exponentiated to get std)
class ThicknessPredictor(nn.Module):
    def __init__(self):
        super(ThicknessPredictor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 14, 14)
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 64, 7, 7)
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.output_mean = nn.Linear(128, 1)
        self.output_logvar = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mean = self.output_mean(x)
        logvar = self.output_logvar(x)
        return mean, logvar