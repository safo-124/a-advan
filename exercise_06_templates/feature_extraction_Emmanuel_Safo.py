
from typing import List, Optional

import pathlib
import numpy as np
import librosa
import csv

from utils import get_files_from_dir_with_pathlib, get_audio_file_data, serialize_features_and_metadata


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    """Normalizes the feature to the interval [0, 1].

    :param feature: Feature to be normalized.
    :type feature: numpy.ndarray
    :return: Normalized feature.
    :rtype: numpy.ndarray
    """
    feat_min = feature.min()
    feat_max = feature.max()
    denom = feat_max - feat_min
    if denom == 0:
        return np.zeros_like(feature)
    return (feature - feat_min) / denom


def extract_binaural_features(audio_signal: np.ndarray,
                                n_fft: Optional[int] = 2048,
                                hop_length: Optional[int] = 1024,
                                window: Optional[str] = 'hamm') \
            -> np.ndarray:
        """Extracts and returns the binaural features from the `audio_signal` signal.

        :param audio_signal: Audio signal with shape (2, n_samples).
        :type audio_signal: numpy.ndarray
        :param n_fft: STFT window length (in samples), defaults to 2048.
        :type n_fft: Optional[int]
        :param hop_length: Hop length (in samples), defaults to 1024.
        :type hop_length: Optional[int]
        :param window: Window type, defaults 'hamm'.
        :type window: Optional[str]
        :return: Numpy array with shape (channels, n_frames, n_bins). The first channel
                    contains the normalized log-magnitude spectrogram of the left channel,
                    the second channel contains the normalized log-magnitude spectrogram
                    of the right channel, and the third and fourth channels contain the
                    sin & cos of the interaural phase differences (IPDs).
        :rtype: numpy.ndarray
        """

        # Compute STFTs for left and right channels.
        # librosa.stft returns complex array of shape (n_fft//2+1, n_frames) = (K, T)
        spectrogram_l = librosa.core.stft(audio_signal[0], n_fft=n_fft, hop_length=hop_length, window=window)
        spectrogram_r = librosa.core.stft(audio_signal[1], n_fft=n_fft, hop_length=hop_length, window=window)

        # Normalized log-magnitude spectrograms (shape: K, T -> transposed to T, K)
        mag_spect_l = normalize_feature(np.log1p(np.abs(spectrogram_l))).T
        mag_spect_r = normalize_feature(np.log1p(np.abs(spectrogram_r))).T

        # Interchannel phase difference: IPD = angle(X_l) - angle(X_r)
        ipd = np.angle(spectrogram_l) - np.angle(spectrogram_r)
        ipd_sin = np.sin(ipd).T
        ipd_cos = np.cos(ipd).T

        # Stack into (4, T, K) tensor
        features = np.stack([mag_spect_l, mag_spect_r, ipd_sin, ipd_cos])
        return features


def main(dataset_root_path: pathlib.Path, splits: List[str]):
    """Extracts the binaural features from the audio files in the dataset paths.

    :param dataset_root_path: Root path of the dataset.
    :type dataset_root_path: list[pathlib.Path]
    :param splits: Splits of the dataset.
    :type splits: list[str]
    """

    for split in splits:
        # Get the audio files.
        split_path = dataset_root_path / split
        audio_files = sorted(get_files_from_dir_with_pathlib(split_path))

        # Output directory.
        output_dir = pathlib.Path(str(split_path) + '_features')
        output_dir.mkdir(exist_ok=True)

        # Read metadata file.
        metadata_file_path = dataset_root_path / (split + "_metadata.csv")
        doas, distances, rt60s = [], [], []
        with open(metadata_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            headers = next(csv_reader)
            for row in csv_reader:
                doas.append(float(row[2]))
                distances.append(float(row[3]))
                rt60s.append(float(row[4]))

        # Extract the features from every audio file.
        for audio_file, doa, distance, rt60 in zip(audio_files, doas, distances, rt60s):
            print("Processing audio file {}.".format(audio_file))

            # Get the audio data of the file `audio_file`.
            audio_data, sampling_rate = get_audio_file_data(audio_file)

            # Extract the features from the audio data.
            binaural_features = extract_binaural_features(audio_data)

            # Serialize the features and metadata.
            features_and_metadata = {'features': binaural_features,
                                     'doa': doa,
                                     'distance': distance,
                                     'rt60': rt60}
            serialize_features_and_metadata(str(output_dir / (audio_file.stem + '.pkl')), features_and_metadata)


if __name__ == '__main__':
    # Paths of the datasets.
    dataset_root_path = pathlib.Path("dataset/binaurals")
    splits = ['training', 'validation', 'testing']
    main(dataset_root_path, splits)
