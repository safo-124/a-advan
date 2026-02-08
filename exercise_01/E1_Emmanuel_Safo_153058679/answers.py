#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, List, Union, MutableMapping

import numpy as np
import torch
import librosa
import pickle
from torch.utils.data import DataLoader


__docformat__ = 'reStructuredText'
__all__ = [
    'get_audio_file_data',
    'extract_mel_band_energies',
    'serialize_features_and_classes',
    'dataset_iteration'
]


def get_audio_file_data(audio_file: str) -> np.ndarray:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    # Load audio file using librosa
    # sr=None preserves the original sampling rate
    audio_data, sr = librosa.load(str(audio_file), sr=None)
    # Print the length in seconds
    length_in_seconds = audio_data.shape[-1] / sr
    print(f"File: {audio_file}, Length: {length_in_seconds:.2f} seconds")
    return audio_data


def extract_mel_band_energies(audio_file: str) -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    # Load audio file
    audio_data, sr = librosa.load(str(audio_file), sr=None)
    
    # Calculate parameters:
    # Window length = 0.02 seconds
    # Hop size = 50% of window length
    win_length_sec = 0.02
    win_length_samples = int(win_length_sec * sr)
    hop_length_samples = win_length_samples // 2  # 50% hop size
    n_fft = win_length_samples
    n_mels = 40  # 40 mel bands
    
    # Extract mel-band energies using librosa.feature.melspectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_samples,
        win_length=win_length_samples,
        window='hann',  # Hanning window
        n_mels=n_mels
    )
    
    return mel_spectrogram


def serialize_features_and_classes(features_and_classes: MutableMapping[str, Union[np.ndarray, int]], 
                                    output_file: str) -> None:
    """Serializes the features and classes.

    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    :param output_file: Path to the output pickle file.
    :type output_file: str
    """
    with open(output_file, 'wb') as f:
        pickle.dump(features_and_classes, f)


def dataset_iteration(dataset: torch.utils.data.Dataset,
                      drop_last: bool = True,
                      shuffle: bool = True,
                      batch_size: int = 3) -> None:
    """Iterates over the dataset using the DataLoader of PyTorch.

    :param dataset: Dataset to iterate over.
    :type dataset: torch.utils.data.Dataset
    :param drop_last: Whether to drop the last incomplete batch.
    :type drop_last: bool
    :param shuffle: Whether to shuffle the data.
    :type shuffle: bool
    :param batch_size: Batch size.
    :type batch_size: int
    """
    # Initialize the DataLoader with the specified parameters
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    
    # Iterate over the dataset and print information
    for batch_idx, (features, classes) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Features type: {type(features)}, shape: {features.shape}")
        print(f"  Classes type: {type(classes)}, values: {classes}")
        print()

# EOF
