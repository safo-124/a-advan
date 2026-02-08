#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple

import torch
from torch import nn, Tensor, rand


def make_cnn_block(in_channels: int,
                   out_channels: int,
                   kernel_size: Union[Tuple[int], int],
                   stride: Union[Tuple[int], int],
                   padding: Union[Tuple[int], int],
                   dropout: float) -> nn.Sequential:
    """ create a 2d convolution block: Conv2d, BatchNorm2d, ReLU, Dropout """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Dropout(p=dropout)
    )


class MyCNNSystem(nn.Module):

    def __init__(self,
                 cnn_channels_1: int,
                 cnn_kernel_1: Union[Tuple[int], int],
                 cnn_stride_1: Union[Tuple[int], int],
                 cnn_padding_1: Union[Tuple[int], int],
                 pooling_kernel_1: Union[Tuple[int], int],
                 pooling_stride_1: Union[Tuple[int], int],
                 cnn_channels_2: int,
                 cnn_kernel_2: Union[Tuple[int], int],
                 cnn_stride_2: Union[Tuple[int], int],
                 cnn_padding_2: Union[Tuple[int], int],
                 pooling_kernel_2: Union[Tuple[int], int],
                 pooling_stride_2: Union[Tuple[int], int],
                 classifier_input_features: int,
                 output_classes: int,
                 dropout: float) -> None:
        """MyCNNSystem, using two CNN layers, followed by a ReLU, a batch norm,\
        and a max-pooling process.

        :param cnn_channels_1: Output channels of first CNN.
        :type cnn_channels_1: int
        :param cnn_kernel_1: Kernel shape of first CNN.
        :type cnn_kernel_1: int|Tuple[int, int]
        :param cnn_stride_1: Strides of first CNN.
        :type cnn_stride_1: int|Tuple[int, int]
        :param cnn_padding_1: Padding of first CNN.
        :type cnn_padding_1: int|Tuple[int, int]
        :param pooling_kernel_1: Kernel shape of first pooling.
        :type pooling_kernel_1: int|Tuple[int, int]
        :param cnn_channels_out_2: Output channels of second CNN.
        :type cnn_channels_out_2: int
        :param cnn_kernel_2: Kernel shape of second CNN.
        :type cnn_kernel_2: int|Tuple[int, int]
        :param cnn_stride_2: Strides of second CNN.
        :type cnn_stride_2: int|Tuple[int, int]
        :param cnn_padding_2: Padding of second CNN.
        :type cnn_padding_2: int|Tuple[int, int]
        :param pooling_kernel_2: Kernel shape of second pooling.
        :type pooling_kernel_2: int|Tuple[int, int]
        :param pooling_stride_2: Stride of second pooling.
        :type pooling_stride_2: int|Tuple[int, int]
        :param classifier_input_features: Input features to the\
                                          classifier.
        :type classifier_input_features: int
        :param dropout: Dropout to use.
        :type dropout: float
        :param output_classes: Output classes.
        :type output_classes: int
        """
        super().__init__()
        
        # Block 1: Conv2d -> BatchNorm -> ReLU -> Dropout
        self.block_1 = make_cnn_block(
            in_channels=1,  # Input is mono audio spectrogram
            out_channels=cnn_channels_1,
            kernel_size=cnn_kernel_1,
            stride=cnn_stride_1,
            padding=cnn_padding_1,
            dropout=dropout
        )
        
        # Pooling after block 1
        self.pooling_1 = nn.MaxPool2d(kernel_size=pooling_kernel_1, stride=pooling_stride_1)
        
        # Block 2: Conv2d -> BatchNorm -> ReLU -> Dropout
        self.block_2 = make_cnn_block(
            in_channels=cnn_channels_1,
            out_channels=cnn_channels_2,
            kernel_size=cnn_kernel_2,
            stride=cnn_stride_2,
            padding=cnn_padding_2,
            dropout=dropout
        )
        
        # Global pooling to pool out mel-bins and time frames
        self.pooling_2 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier layer
        self.classifier = nn.Linear(in_features=classifier_input_features, out_features=output_classes)


    def forward(self,
                x: Tensor)\
            -> Tensor:
        """Forward pass.

        :param x: Input features\
                  (shape either `batch x time x features` or\
                  `batch x channels x time x features`).
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        h = x if x.dim() == 4 else x.unsqueeze(1)
        
        # apply block_1 to h
        h = self.block_1(h)
        h = self.pooling_1(h)
        
        # apply block_2 to h
        h = self.block_2(h)
        
        # apply global pooling and squeeze dimensions
        h = self.pooling_2(h)
        h = h.squeeze(-1).squeeze(-1)  # Remove spatial dimensions

        return self.classifier(h)


def main():
    
    # Check if CUDA is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate our CNN
    # Define the CNN model and give it the model hyperparameters
    cnn_model = MyCNNSystem(
        cnn_channels_1=32,
        cnn_kernel_1=(3, 3),
        cnn_stride_1=(1, 1),
        cnn_padding_1=(1, 1),
        pooling_kernel_1=(2, 2),
        pooling_stride_1=(2, 2),
        cnn_channels_2=64,
        cnn_kernel_2=(3, 3),
        cnn_stride_2=(1, 1),
        cnn_padding_2=(1, 1),
        pooling_kernel_2=(2, 2),
        pooling_stride_2=(2, 2),
        classifier_input_features=64,
        output_classes=10,
        dropout=0.25)
    
    # Pass DNN to the available device.
    cnn_model = cnn_model.to(device)
    
    batch_size = 8
    d_time = 350
    d_feature = 64
    x = rand(batch_size, d_time, d_feature)
    y = rand(batch_size, 1)
    
    # Give them to the appropriate device.
    x = x.to(device)
    y = y.to(device)

    # Get the predictions.
    y_hat = cnn_model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_hat.shape}")
    print("CNN model test passed!")
    
    
if __name__ == '__main__':
    main()

# EOF
