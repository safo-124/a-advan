
import torch
import torch.nn as nn


class SSLModel(nn.Module):
    """CNN model for binaural sound source localization (SSL).

    Architecture:
        - 2 convolutional blocks, each with:
            * Conv2d(in, 128, 3x3, padding=1)
            * MaxPool2d with stride 4 on the frequency axis
            * BatchNorm2d(128)
            * ReLU
        - Global average pooling over the time axis
        - Flatten
        - MLP: Linear -> ReLU -> Dropout -> Linear (4 classes, no softmax)

    Input shape:  (batch, 4, T, K)   -- C=4 channels, T time frames, K freq bins
    Output shape: (batch, 4)         -- 4-class logits
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 4,
                 n_freq_bins: int = 1025, dropout_rate: float = 0.5):
        """
        :param in_channels: Number of input channels (default 4: log-mag L, log-mag R, sin IPD, cos IPD).
        :param num_classes: Number of output classes (default 4: front, left, back, right).
        :param n_freq_bins: Number of frequency bins in the input (n_fft//2 + 1, default 1025).
        :param dropout_rate: Dropout probability in the MLP.
        """
        super().__init__()

        # Convolutional block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Convolutional block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Compute the frequency dimension after two max-pool operations
        freq = n_freq_bins
        freq = (freq - 4) // 4 + 1   # after 1st pool: 1025 -> 256
        freq = (freq - 4) // 4 + 1   # after 2nd pool:  256 ->  64
        mlp_input_size = 128 * freq   # 128 * 64 = 8192

        # MLP head
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch, 4, T, K).
        :return: Logits of shape (batch, num_classes).
        """
        # Convolutional blocks
        x = self.conv_block1(x)   # (B, 128, T, K/4)
        x = self.conv_block2(x)   # (B, 128, T, K/16)

        # Average pooling over the time axis (dim=2)
        x = x.mean(dim=2)         # (B, 128, K/16)

        # Flatten and classify
        x = x.flatten(1)          # (B, 128 * K/16)
        x = self.classifier(x)    # (B, num_classes)

        return x
