"""
cnn_model.py
------------
Hybrid CNN meta-learning architecture for algorithm + hyperparameter prediction.

Architecture:
  Input: (1, 20, 20)
  Conv2d(1 → 32)  + BatchNorm + ReLU + MaxPool(2)   →  (32, 10, 10)
  Conv2d(32 → 64) + BatchNorm + ReLU + MaxPool(2)   →  (64, 5, 5)
  Conv2d(64 → 128) + BatchNorm + ReLU               →  (128, 5, 5)
  AdaptiveAvgPool(3, 3)                              →  (128, 3, 3)
  Flatten                                            →  (1152)
  Dense(1152, 256) + ReLU + Dropout(0.4)
  Dense(256, 128)  + ReLU + Dropout(0.3)
  Dense(128, num_configs)                            →  (36)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import CNN_DROPOUT, NUM_CONFIGS, MATRIX_SIZE


class MetaLearnerCNN(nn.Module):
    """
    Hybrid CNN for meta-learning.
    Predicts the best (algorithm, hyperparameter) configuration index.
    """

    def __init__(self, num_configs=NUM_CONFIGS, input_size=MATRIX_SIZE, dropout=CNN_DROPOUT):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling → fixed size regardless of input
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        # Dense layers
        self.flatten_size = 128 * 3 * 3  # = 1152
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_configs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.75)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → (32, 10, 10)
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → (64, 5, 5)
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))              # → (128, 5, 5)
        # Adaptive pool
        x = self.adaptive_pool(x)                         # → (128, 3, 3)

        # Flatten + Dense
        x = x.view(x.size(0), -1)                        # → (1152)
        x = self.dropout1(F.relu(self.fc1(x)))            # → (256)
        x = self.dropout2(F.relu(self.fc2(x)))            # → (128)
        x = self.fc3(x)                                   # → (num_configs)
        return x

    @torch.no_grad()
    def predict(self, x):
        """
        Predict config index and probabilities.

        Returns
        -------
        predictions   : torch.Tensor of shape (batch,)
        probabilities : torch.Tensor of shape (batch, num_configs)
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = MetaLearnerCNN()
    x = torch.randn(2, 1, MATRIX_SIZE, MATRIX_SIZE)
    out = model(x)
    print(f"Input:      {x.shape}")
    print(f"Output:     {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    preds, probs = model.predict(x)
    print(f"Predictions: {preds}")
    print(f"Prob sums:   {probs.sum(dim=1)}")
