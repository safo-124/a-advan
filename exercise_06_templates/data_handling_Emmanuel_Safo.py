
import pathlib
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset


class BinauralDataset(Dataset):
    """Dataset class for binaural sound source localization.

    Loads serialized features (.pkl) and converts the continuous DOA angle
    into one of 4 class labels:
        0 = Front  : DOA in [ pi/4,  3*pi/4)
        1 = Left   : DOA in [3*pi/4,  pi] or [-pi, -3*pi/4)
        2 = Back   : DOA in [-3*pi/4, -pi/4)
        3 = Right  : DOA in [-pi/4,   pi/4)
    """

    def __init__(self, features_dir: str):
        """Initializes the dataset.

        :param features_dir: Path to the directory containing the serialized .pkl feature files.
        :type features_dir: str
        """
        self.features_dir = pathlib.Path(features_dir)
        self.files = sorted(list(self.features_dir.glob('*.pkl')))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        """Returns the features and class label for a given index.

        :param idx: Index of the sample.
        :type idx: int
        :return: Tuple of (features, label) where features is a float32 tensor
                 of shape (4, T, K) and label is an integer class index.
        :rtype: tuple[torch.Tensor, int]
        """
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)

        features = torch.tensor(data['features'], dtype=torch.float32)
        doa = data['doa']
        label = self.doa_to_class(doa)

        return features, label

    @staticmethod
    def doa_to_class(doa: float) -> int:
        """Converts a DOA angle (in radians, range [-pi, pi]) to a class label.

        The continuous DOA space is split into 4 equal angular regions of 90 degrees:
            Front (0) : y > 0 side,  DOA in [ pi/4,  3*pi/4)
            Left  (1) : -x side,    DOA in [3*pi/4,  pi] or [-pi, -3*pi/4)
            Back  (2) : -y side,    DOA in [-3*pi/4, -pi/4)
            Right (3) : +x side,    DOA in [-pi/4,   pi/4)

        :param doa: Direction of arrival in radians.
        :type doa: float
        :return: Class label (0-3).
        :rtype: int
        """
        if -np.pi / 4 <= doa < np.pi / 4:
            return 3  # Right
        elif np.pi / 4 <= doa < 3 * np.pi / 4:
            return 0  # Front
        elif -3 * np.pi / 4 <= doa < -np.pi / 4:
            return 2  # Back
        else:
            return 1  # Left
