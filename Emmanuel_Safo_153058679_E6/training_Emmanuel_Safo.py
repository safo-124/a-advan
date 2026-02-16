#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

import torch
from torch import cuda, no_grad
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

from models_Emmanuel_Safo import SSLModel
from data_handling_Emmanuel_Safo import BinauralDataset


__docformat__ = 'reStructuredText'


def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    epochs = 20

    # Instantiate our DNN
    model = SSLModel(in_channels=4, num_classes=4, n_freq_bins=1025, dropout_rate=0.5)

    # Pass DNN to the available device.
    model = model.to(device)

    # Define the optimizer and give the parameters of the CNN model to an optimizer.
    optimizer = Adam(params=model.parameters(), lr=1e-3)

    # Instantiate the loss function as a class.
    loss_function = nn.CrossEntropyLoss()

    # Init training, validation, and testing dataset.
    batch_size = 32

    train_dataset = BinauralDataset('dataset/binaurals/training_features')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    valid_dataset = BinauralDataset('dataset/binaurals/validation_features')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    test_dataset = BinauralDataset('dataset/binaurals/testing_features')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 5
    patience_counter = 0

    best_model = None

    # Start training.
    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        model.train()

        # For each batch of our dataset.
        for batch in train_loader:
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y = batch

            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device)

            # Get the predictions.
            y_hat = model(x)

            # Calculate the loss.
            loss = loss_function(y_hat, y)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Append the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode
        model.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():

            # For every batch of our validation data.
            for batch in valid_loader:
                # Get the batch
                x_val, y_val = batch

                # Pass the data to the appropriate device.
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                # Get the predictions of the model.
                y_hat = model(x_val)

                # Calculate the loss.
                loss = loss_function(y_hat, y_val)

                # Append the validation loss.
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if patience_counter >= patience or epoch == epochs - 1:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Save and load the best model.
                torch.save(best_model, 'best_model_ssl.pt')
                model.load_state_dict(best_model)

                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                labels_test = []
                predictions_test = []
                model.eval()
                with no_grad():
                    for batch in test_loader:
                        x_test, y_test = batch

                        # Pass the data to the appropriate device.
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)

                        # Make the prediction
                        y_hat = model(x_test)

                        # Calculate the loss.
                        loss = loss_function(y_hat, y_test)
                        testing_loss.append(loss.item())

                        # Save the predictions and labels for later analysis.
                        predictions_test.append(y_hat.argmax(dim=-1).cpu().numpy())
                        labels_test.append(y_test.cpu().numpy())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')

                # Calculate the accuracy.
                predictions_test = np.concatenate(predictions_test)
                labels_test = np.concatenate(labels_test)
                accuracy = np.mean(predictions_test == labels_test)
                print(f'Testing accuracy: {accuracy:7.4f}')

                # Calculate the confusion matrix.
                confusion_matrix_test = confusion_matrix(labels_test, predictions_test)
                print(f'Confusion matrix:\n{confusion_matrix_test}')

                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()

# Training log:
"""
 Could not find platform independent libraries <prefix>
Process on cpu

Epoch: 000 | Mean training loss:  1.4440 | Mean validation loss  1.0781
Epoch: 001 | Mean training loss:  0.9861 | Mean validation loss  0.9160
Epoch: 002 | Mean training loss:  0.8460 | Mean validation loss  0.8238
Epoch: 003 | Mean training loss:  0.8000 | Mean validation loss  0.8394
Epoch: 004 | Mean training loss:  0.7385 | Mean validation loss  0.8059
Epoch: 005 | Mean training loss:  0.6831 | Mean validation loss  0.7903
Epoch: 006 | Mean training loss:  0.6595 | Mean validation loss  0.8283
Epoch: 007 | Mean training loss:  0.6462 | Mean validation loss  0.8604
Epoch: 008 | Mean training loss:  0.6460 | Mean validation loss  0.9706
Epoch: 009 | Mean training loss:  0.5682 | Mean validation loss  0.8554

Exiting due to early stopping

Best epoch 5 with loss 0.7903252243995667
"""

# Evaluation results:
"""
Starting testing | Testing loss:  0.7450
Testing accuracy:  0.6050
Confusion matrix:
[[ 8  2 24  2]
 [ 1 54  9  0]
 [22  6 26  1]
 [ 6  0  6 33]]
"""


# Analysis:
# Yes, there is a clear problem visible in the confusion matrix: the model heavily
# confuses the Front and Back classes. 24 out of 36 Front samples are misclassified
# as Back, and 22 out of 55 Back samples are misclassified as Front. Meanwhile,
# Left (54/64 = 84% recall) and Right (33/45 = 73% recall) perform much better.
#
# This front-back confusion is a well-known problem in binaural audio processing.
# It arises because the two microphones are placed along the x-axis (left-right),
# so sources in front of and behind the listener produce very similar interaural
# phase differences (IPD) and interaural level differences (ILD). The microphone
# arrangement is symmetric with respect to the y-axis, making front and back
# sources nearly indistinguishable from the features alone.
#
# Possible fixes:
# 1. Use Head-Related Transfer Functions (HRTFs) in the simulation instead of
#    omnidirectional point microphones. HRTFs introduce spectral modifications
#    (especially from the pinna) that provide monaural cues to resolve front-back
#    ambiguity.
# 2. Add more microphones or use directional microphones to break the front-back
#    symmetry.
# 3. Use data augmentation or a larger/more diverse dataset to help the model
#    learn subtle reverberation-based cues that differ between front and back.
# 4. If the application allows it, merge Front and Back into a single class,
#    since they are physically ambiguous with this specific microphone configuration.

# EOF
