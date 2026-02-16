#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union, Tuple, MutableMapping
from pathlib import Path
import pathlib
import os
import numpy as np
import librosa
import pickle


__docformat__ = 'reStructuredText'
__all__ = ['get_files_from_dir_with_pathlib',
           'get_audio_file_data',
           'serialize_features_and_metadata',
           ]


def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())


def get_audio_file_data(audio_file: Union[str, pathlib.Path]) \
        -> Tuple[np.ndarray, float]:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: Tuple[numpy.ndarray, float]
    """
    return librosa.core.load(path=audio_file, sr=None, mono=False)


def serialize_features_and_metadata(file: str, features_and_classes: MutableMapping[str, Union[np.ndarray, int]])\
        -> None:
    """Serializes the features and classes.

    :param file: File to dump the serialized features
    :type file: str
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    with open(file, 'wb') as pkl_file:
        pickle.dump(features_and_classes, pkl_file)

# EOF
