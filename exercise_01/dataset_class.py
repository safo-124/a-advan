#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

from torch.utils.data import Dataset
import numpy
import pickle

from file_io import get_files_from_dir_with_pathlib


__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):

    def __init__(self, data_dir: str) -> None:
        """Initialize the dataset by loading serialized dictionaries from the data directory.
        
        :param data_dir: Path to the directory containing serialized pickle files.
        :type data_dir: str
        """
        super().__init__()
        # Get all files from the data directory
        file_paths = get_files_from_dir_with_pathlib(data_dir)
        
        # Load all dictionaries from pickle files
        self.data = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                self.data.append(pickle.load(f))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[numpy.ndarray, int]:
        """Return a tuple of (features, class) for the given index.
        
        Features are the temporal average of the mel-band energies (average across time frames).
        
        :param item: Index of the sample.
        :type item: int
        :return: Tuple of (features, class).
        :rtype: Tuple[numpy.ndarray, int]
        """
        sample = self.data[item]
        features = sample['features']
        class_label = sample['class']
        
        # Compute temporal average of mel-band energies
        # mel spectrogram shape is (n_mels, n_frames), average across time (axis=1)
        avg_features = numpy.mean(features, axis=1)
        
        return avg_features, class_label

# EOF

