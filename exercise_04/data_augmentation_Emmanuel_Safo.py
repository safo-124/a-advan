#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Augmentation Functions for Audio Classification
Exercise 04 - Advanced Audio Processing

This module implements various data augmentation techniques for audio signals:
- add_noise(): Adds white noise to the signal (Section 2.1)
- add_pitch_shift(): Shifts pitch up/down (Section 2.1)
- add_impulse_response(): Applies room impulse response for reverberation (Section 2.2)
- freq_masking(): Applies frequency masking on spectrograms (Section 2.3 - SpecAugment)
"""

import numpy as np
import librosa
from scipy.signal import fftconvolve
from typing import Optional


def add_noise(y: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add white noise to the audio signal at a specified SNR level.
    
    The noise is generated as Gaussian white noise and scaled to achieve
    the desired signal-to-noise ratio (SNR) in decibels.
    
    SNR_dB = 10 * log10(P_signal / P_noise)
    
    :param y: Input audio signal.
    :type y: np.ndarray
    :param snr_db: Desired signal-to-noise ratio in decibels (default 20.0 dB).
    :type snr_db: float
    :return: Audio signal with added noise.
    :rtype: np.ndarray
    """
    # Calculate signal power
    signal_power = np.mean(y ** 2)
    
    # Calculate noise power needed for desired SNR
    # SNR_dB = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / (10^(SNR_dB/10))
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate white noise with the calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    
    # Add noise to signal
    y_noised = y + noise
    
    return y_noised


def add_pitch_shift(y: np.ndarray, n_steps: float, sr: int = 44100) -> np.ndarray:
    """
    Shift the pitch of an audio signal by n_steps semitones.
    
    Uses librosa's pitch_shift function which shifts frequency components
    up or down by a constant factor. Positive n_steps shifts pitch up,
    negative n_steps shifts pitch down.
    
    :param y: Input audio signal.
    :type y: np.ndarray
    :param n_steps: Number of semitones to shift (can be fractional).
    :type n_steps: float
    :param sr: Sampling rate of the audio signal (default 44100 Hz).
    :type sr: int
    :return: Pitch-shifted audio signal.
    :rtype: np.ndarray
    """
    y_shift = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    
    return y_shift


def add_impulse_response(y: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    """
    Apply room impulse response (RIR) to simulate sound in different environments.
    
    Convolves the input signal with a room impulse response to create
    a reverberant signal that sounds like it was recorded in the room
    where the impulse response was captured.
    
    :param y: Input audio signal.
    :type y: np.ndarray
    :param impulse_response: Room impulse response signal.
    :type impulse_response: np.ndarray
    :return: Reverberant audio signal.
    :rtype: np.ndarray
    """
    # Convolve signal with impulse response using FFT-based convolution for efficiency
    y_impulse = fftconvolve(y, impulse_response, mode='full')
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(y_impulse))
    if max_val > 0:
        y_impulse = y_impulse * (np.max(np.abs(y)) / max_val)
    
    return y_impulse


def freq_masking(y: np.ndarray, max_bands: Optional[int] = None) -> np.ndarray:
    """
    Apply frequency masking (SpecAugment) to a mel spectrogram.
    
    Randomly selects a contiguous range of frequency channels and masks
    them (sets to zero). This is part of the SpecAugment technique [2]
    which helps prevent overfitting by simulating missing frequency information.
    
    The range [f0, f0+f] is masked, where:
    - f is sampled uniformly from [0, max_bands]
    - f0 is sampled uniformly from [0, mF - f] where mF is the number of mel channels
    
    :param y: Input mel spectrogram (shape: n_mels x time_frames).
    :type y: np.ndarray
    :param max_bands: Maximum number of consecutive frequency bands to mask (default 40).
    :type max_bands: Optional[int]
    :return: Mel spectrogram with frequency masking applied.
    :rtype: np.ndarray
    """
    y_masking = y.copy()
    n_mels = y.shape[0]  # Number of mel frequency channels
    
    # Set default max_bands if not provided
    if max_bands is None:
        max_bands = min(40, n_mels)
    
    # Ensure max_bands doesn't exceed the number of mel bands
    max_bands = min(max_bands, n_mels)
    
    # Sample f uniformly from [0, max_bands]
    f = np.random.randint(0, max_bands + 1)
    
    # Sample f0 uniformly from [0, n_mels - f]
    if n_mels - f > 0:
        f0 = np.random.randint(0, n_mels - f)
    else:
        f0 = 0
    
    # Apply mask: zero out frequency channels [f0, f0+f)
    y_masking[f0:f0 + f, :] = 0
    
    return y_masking
