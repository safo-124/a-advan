#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy


__docformat__ = 'reStructuredText'
__all__ = [
    'plot_mel_band_energies',
    'plot_audio_signal'
]


def plot_mel_band_energies(mel_band_energies: numpy.ndarray) -> None:
    """Plots the mel-band energies to the screen.

    :param mel_band_energies: Mel-band energies.
    :type mel_band_energies: numpy.ndarray
    """
    plt.imshow(mel_band_energies.T, aspect='auto', origin='lower',
               norm=colors.LogNorm(vmin=mel_band_energies.min(),
                                   vmax=mel_band_energies.max()))


def plot_audio_signal(audio_data: numpy.ndarray) -> None:
    """Plots raw audio data to screen.

    :param audio_data: Audio data.
    :type audio_data: numy.ndarray
    """
    plt.plot(audio_data)
    plt.xlim((0, len(audio_data)))
    plt.xlabel('Audio')
    plt.show()


# EOF

