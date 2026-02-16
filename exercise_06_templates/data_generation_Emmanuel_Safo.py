
from typing import List, Tuple

import random
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import csv
from pathlib import Path

from utils import get_files_from_dir_with_pathlib, get_audio_file_data


def trim_audio_signal(audio_signal: np.ndarray, sr: float, duration: float) -> np.ndarray:
    """Trims a random segment from `audio_signal` with the `duration` duration.

    :param audio_signal: Audio signal.
    :type audio_signal: numpy.ndarray
    :param sr: Sampling rate of the `audio_signal`.
    :type sr: float
    :param duration: Duration of the output audio signal [s].
    :type duration: float
    :return: Trimmed audio signal.
    :rtype: numpy.ndarray
    """
    target_samples = int(sr * duration)
    if len(audio_signal) >= target_samples:
        start = random.randint(0, len(audio_signal) - target_samples)
        return audio_signal[start:start + target_samples]
    else:
        # Pad with zeros if the signal is shorter than the desired duration
        padded = np.zeros(target_samples)
        padded[:len(audio_signal)] = audio_signal
        return padded


def random_float_in_interval(min_value: float, max_value: float) -> float:
    """ Returns a random float in the interval [min_value, max_value].

    :param min_value: Minimum value of the interval.
    :type min_value: float
    :param max_value: Maximum value of the interval.
    :type max_value: float
    :return: Random float in the interval [min_value, max_value].
    :rtype: float
    """
    return random.uniform(min_value, max_value)


def simulate_random_room(source_audio_signal: np.ndarray) \
        -> Tuple[np.ndarray, float, float, float]:
    """ Returns a random room impulse response.

    :param source_audio_signal: Audio signal of the source.
    :type source_audio_signal: numpy.ndarray
    :return: Binaural simulation of the signal in a random room.
    :rtype: numpy.ndarray
    :return: Direction of arrival of the source [radians].
    :rtype: float
    :return: Distance of the source to the microphone [meters].
    :rtype: float
    :return: Reverberation time (T60) of the room [seconds].
    :rtype: float
    """
    # Random room properties.
    room_size = [random_float_in_interval(3, 7),
                 random_float_in_interval(3, 7),
                 random_float_in_interval(2, 4)]
    rt60 = random_float_in_interval(0.0, 1.0)

    # Compute the walls absorption and the ISM order for the room.
    e_absorption, max_order = pra.inverse_sabine(rt60, room_size)

    # Create the room.
    room = pra.ShoeBox(
        room_size, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
    )

    # Random microphone position.
    mic_center = [random_float_in_interval(0.5, room_size[0] - 0.5),
                  random_float_in_interval(0.5, room_size[1] - 0.5),
                  random_float_in_interval(0.5, room_size[2] - 0.5)]
    mic_positions = np.array([[mic_center[0] - 0.08, mic_center[1], mic_center[2]],
                              [mic_center[0] + 0.08, mic_center[1], mic_center[2]]]).T
    room.add_microphone_array(mic_positions)

    # Random source position.
    source_position = [random_float_in_interval(0.5, room_size[0] - 0.5),
                       random_float_in_interval(0.5, room_size[1] - 0.5),
                       random_float_in_interval(0.5, room_size[2] - 0.5)]
    room.add_source(source_position, signal=source_audio_signal)

    # Compute the RIR.
    room.simulate()
    binaural_audio = room.mic_array.signals.T
    binaural_audio = binaural_audio[:len(source_audio_signal), :]

    # Compute the direction of arrival of the source (angle from mic center to source in the horizontal plane).
    # x-axis = right, y-axis = front (as shown in the exercise diagram).
    direction_of_arrival = np.arctan2(
        source_position[1] - mic_center[1],
        source_position[0] - mic_center[0]
    )

    # Compute the distance of the source to the microphone center.
    distance = np.sqrt(
        (source_position[0] - mic_center[0]) ** 2 +
        (source_position[1] - mic_center[1]) ** 2 +
        (source_position[2] - mic_center[2]) ** 2
    )

    return binaural_audio, direction_of_arrival, distance, rt60


def main(utterances_root_path: Path, splits: List, output_root_path: Path):
    """Simulates a random room for every utterance in the `utterances_root_path` directory.

    :param utterances_root_path: Path of the directory with the utterances.
    :type utterances_root_path: Path
    :param splits: Splits of the utterances dataset.
    :type splits: List[str]
    :param output_root_path: Path of the output directory.
    :type output_root_path: Path
    """
    for split in splits:
        # Get all the files in the split.
        utterances_path = utterances_root_path / split
        utterances_files = sorted(get_files_from_dir_with_pathlib(utterances_path))

        # Output directory and metadata lists.
        output_dir = output_root_path / split
        output_dir.mkdir(exist_ok=True)
        file_names = []
        source_utterances = []
        doas = []
        distances = []
        rt60s = []

        # Simulate a random position for every utterance.
        for utt_idx, utterance_file in enumerate(utterances_files):
            print("Processing utterance {}/{} of the {} split.".format(utt_idx, len(utterances_files), split))

            # Get the audio data of the file `utterance_file`.
            audio_data, sampling_rate = get_audio_file_data(utterance_file)
            audio_data = trim_audio_signal(audio_data, sampling_rate, 5.0)

            # Simulate a random room.
            while True:
                # Some room sizes and reverberation times are not compatible and raise an exception,
                # so we try again until we get a valid room.
                try:
                    binaural_audio, direction_of_arrival, distance, rt60 = simulate_random_room(audio_data)
                    break
                except ValueError:
                    pass

            # Save the simulated audio.
            output_file = output_dir / "{}_{:03}.wav".format(split, utt_idx)
            sf.write(str(output_file), binaural_audio, 16000)

            # Append the metadata.
            file_names.append(output_file.name)
            source_utterances.append(utterance_file.name)
            doas.append(direction_of_arrival)
            distances.append(distance)
            rt60s.append(rt60)
            print("DOA: {:.2f}º, distance: {:.2f}m, rt60: {:.2f}s".format(np.rad2deg(direction_of_arrival),
                                                                           distance, rt60), flush=True)

        # Save the metadata.
        metadata_file = output_dir.parent / (split + "_metadata.csv")
        with open(metadata_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['file_name', 'source_utterance', 'doa', 'distance', 'rt60'])
            for file_name, source_utterance, doa, distance, rt60 in zip(file_names, source_utterances, doas, distances,
                                                                        rt60s):
                csv_writer.writerow([file_name, source_utterance, doa, distance, rt60])


if __name__ == '__main__':
    utterances_root_path = Path("dataset/utterances")
    splits = ['training', 'validation', 'testing']
    output_root_path = Path("dataset/binaurals")
    output_root_path.mkdir(exist_ok=True)
    main(utterances_root_path, splits, output_root_path)
