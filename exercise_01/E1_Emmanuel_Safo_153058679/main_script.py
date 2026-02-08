# This is a script to perform the execution of the functions provided 
# in exercise no. 1 and also the ones implemented by the student. Please, place
# the execution of each task as separated by the template below. The whole code should 
# run automatically, provided the correct directories. 
# The TA will not debug your code.

import os
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from file_io import get_files_from_dir_with_os, get_files_from_dir_with_pathlib
from answers import get_audio_file_data, extract_mel_band_energies, serialize_features_and_classes, dataset_iteration
from plotting import plot_mel_band_energies, plot_audio_signal
from dataset_class import MyDataset


#### TASK 2.1
#### Place the execution of task 2.1 below

audio_dir = 'audio'

# Using os package
files_os = get_files_from_dir_with_os(audio_dir)
print("Files using os package:")
print(files_os)

# Using pathlib package
files_pathlib = get_files_from_dir_with_pathlib(audio_dir)
print("\nFiles using pathlib package:")
print(files_pathlib)

# Sort the paths
sorted_os = sorted(files_os)
sorted_pathlib = sorted(files_pathlib)
print("\nSorted files (os):", sorted_os)
print("Sorted files (pathlib):", sorted_pathlib)

# Answers to questions 2.1:
# a) What is the difference of the paths that are returned by each function?
#    - get_files_from_dir_with_os() returns just the filenames as strings (e.g., "1.wav")
#    - get_files_from_dir_with_pathlib() returns full Path objects with the directory included (e.g., "audio/1.wav")
#
# b) What is the difference in the values returned by each function?
#    - os.listdir() returns a list of strings containing only filenames
#    - pathlib.Path.iterdir() returns pathlib.Path objects which include the full path
#
# c) Using the sorted() function in Python, sort the paths. What do you observe in the ordering of the paths?
#    - The files are sorted lexicographically (alphabetically), not numerically
#    - This means "10.wav" comes before "2.wav" because "1" < "2" in string comparison
#    - The order would be: 1.wav, 10.wav, 11.wav, 2.wav, 20.wav, 21.wav, 3.wav, 30.wav, 31.wav
#
# d) Without changing the conceptual name of the files, how can you fix what you observed from the ordering?
#    - Use zero-padding in filenames: 01.wav, 02.wav, ..., 10.wav, 11.wav
#    - Or use a custom sort key: sorted(files, key=lambda x: int(x.stem) if using pathlib or sorted(files, key=lambda x: int(x.split('.')[0]))


#### TASK 2.2
#### Place the execution of task 2.2 below

print("\n\n=== TASK 2.2: Reading audio files ===")
# Get all audio file paths
audio_files = get_files_from_dir_with_pathlib(audio_dir)

# Load each audio file and print its length
for audio_file in audio_files:
    audio_data = get_audio_file_data(str(audio_file))


#### TASK 3.1
#### Place the execution of task 3.1 below

print("\n\n=== TASK 3.1: Extraction of mel-band energies ===")
# Load ex1.wav (which is 1.wav in the audio directory)
ex1_file = os.path.join(audio_dir, '1.wav')

# Extract mel-band energies
mel_energies = extract_mel_band_energies(ex1_file)
print(f"Mel-band energies shape: {mel_energies.shape}")

# Load audio for plotting
import librosa
audio_data, sr = librosa.load(ex1_file, sr=None)

# Plot the audio signal
plt.figure(figsize=(12, 4))
plt.plot(audio_data)
plt.xlim((0, len(audio_data)))
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Audio Signal - 1.wav')
plt.savefig('audio_signal.png', dpi=150, bbox_inches='tight')
plt.close()
print("Audio signal plot saved to audio_signal.png")

# Plot the mel-band energies
plt.figure(figsize=(12, 4))
plot_mel_band_energies(mel_energies)
plt.colorbar(label='Energy')
plt.xlabel('Time Frames')
plt.ylabel('Mel Bands')
plt.title('Mel-Band Energies')
plt.savefig('mel_energies.png', dpi=150, bbox_inches='tight')
plt.close()
print("Mel-band energies plot saved to mel_energies.png")

# Answers to questions 3.1:
# a) Can you recognise which is the dimension of the array of the features that holds the different mel-bands?
#    - The first dimension (axis 0) holds the mel-bands. With n_mels=40, this dimension has size 40.
#    - Shape is (n_mels, n_frames) = (40, n_frames)
#
# b) What would be the other dimension?
#    - The second dimension (axis 1) represents the time frames.
#    - The number of frames depends on the audio length, hop size, and window length.
#
# c) Is there any connection between the length of the corresponding audio file and the extracted features?
#    - Yes, there is a connection. The number of time frames in the mel spectrogram is related to the audio length.
#    - n_frames ≈ (audio_length_samples - win_length) / hop_length + 1
#    - Longer audio files will have more time frames in the mel spectrogram.
#
# d) Qualitatively explain the figure of the mel-band energies, in relation to the figure of the audio.
#    - The mel spectrogram shows energy distribution across frequency bands (vertical axis) over time (horizontal axis).
#    - High energy regions in the mel spectrogram correspond to loud/intense parts of the audio waveform.
#    - Quiet sections of the audio correspond to lower energy values (darker regions) in the mel spectrogram.
#    - The mel scale emphasizes lower frequencies, which is more aligned with human perception.


#### TASK 3.2
#### Place the execution of task 3.2 below

print("\n\n=== TASK 3.2: Serializing the extracted features ===")
# Create directory for serialized features if it doesn't exist
serialized_dir = 'serialized_features'
os.makedirs(serialized_dir, exist_ok=True)

# Loop over all audio files, extract mel-band energies, and serialize with random class
for i, audio_file in enumerate(audio_files):
    # Extract mel-band energies
    mel_features = extract_mel_band_energies(str(audio_file))
    
    # Randomly assign class 0 or 1
    random_class = random.randint(0, 1)
    
    # Create dictionary with features and class
    features_and_classes = {
        'features': mel_features,
        'class': random_class
    }
    
    # Serialize to pickle file
    output_file = os.path.join(serialized_dir, f'sample_{i}.pkl')
    serialize_features_and_classes(features_and_classes, output_file)
    print(f"Serialized {audio_file} with class {random_class} to {output_file}")


#### TASK 4.1
#### Place the execution of task 4.1 below

print("\n\n=== TASK 4.1: Creation of dataset class ===")
# Create dataset from serialized features
dataset = MyDataset(serialized_dir)
print(f"Dataset length: {len(dataset)}")

# Test __getitem__
sample_features, sample_class = dataset[0]
print(f"Sample features shape: {sample_features.shape}")
print(f"Sample class: {sample_class}")


#### TASK 4.2
#### Place the execution of task 4.2 below

print("\n\n=== TASK 4.2: Iterating over the dataset ===")
# Iterate over dataset with default parameters (drop_last=True, shuffle=True, batch_size=3)
print("With drop_last=True, shuffle=True, batch_size=3:")
dataset_iteration(dataset, drop_last=True, shuffle=True, batch_size=3)

# Experiment with different parameters
print("\nWith drop_last=False, shuffle=False, batch_size=2:")
dataset_iteration(dataset, drop_last=False, shuffle=False, batch_size=2)

