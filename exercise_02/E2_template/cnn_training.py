#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
import torch
from torch import cuda, no_grad

import torch.nn as nn
from torch.optim import Adam

from cnn_system import MyCNNSystem
from pathlib import Path
from data_loader import load_data

__docformat__ = 'reStructuredText'



def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    epochs = 100

    num_classes = 10  # 10 music genres

    # Instantiate our DNN
    # Define the CNN model and give it the model hyperparameters
    cnn_model = MyCNNSystem(
        cnn_channels_1=32,
        cnn_kernel_1=(3, 3),
        cnn_stride_1=(1, 1),
        cnn_padding_1=(1, 1),
        pooling_kernel_1=(2, 2),
        pooling_stride_1=(2, 2),
        cnn_channels_2=64,
        cnn_kernel_2=(3, 3),
        cnn_stride_2=(1, 1),
        cnn_padding_2=(1, 1),
        pooling_kernel_2=(2, 2),
        pooling_stride_2=(2, 2),
        classifier_input_features=64,
        output_classes=num_classes,
        dropout=0.25)

    # Pass DNN to the available device.
    cnn_model = cnn_model.to(device)

    # Define the optimizer and give the parameters of the CNN model to an optimizer.
    optimizer = Adam(cnn_model.parameters(), lr=0.001)

    # Instantiate the loss function as a class.
    loss_function = nn.CrossEntropyLoss()

    # Init training, validation, and testing dataset.
    data_path = Path(__file__).parent.parent / 'genres_dataset'
    batch_size = 4
    
    # Load training data
    split = "training"
    train_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)
    
    # Load validation data
    split = 'validation'
    valid_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)
    
    # Load testing data (no shuffle to keep track of predictions)
    split = 'testing'
    test_loader = load_data(data_path, split, batch_size, shuffle=False, drop_last=False, num_workers=1)
   

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 30
    patience_counter = 0

    best_model = None

    # Start training.
    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        cnn_model.train()

        # For each batch of our dataset.
        for batch in train_loader:
            
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y = batch
                   
            # Give them to the appropriate device.
            x = x.float().to(device)
            y = y.long().to(device)
            
            # Get the predictions.
            y_hat = cnn_model(x)

            # Calculate the loss.
            loss = loss_function(y_hat, y)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Append the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode
        cnn_model.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():

            # For every batch of our validation data.
            for batch in valid_loader:
                # Get the batch
                x_val, y_val = batch
                
                # Pass the data to the appropriate device.
                x_val = x_val.float().to(device)
                y_val = y_val.long().to(device)

                # Get the predictions of the model.
                y_hat = cnn_model(x_val)

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
            best_model = deepcopy(cnn_model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if (patience_counter >= patience) or (epoch==epochs-1):
            
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                # Load best model 
                cnn_model.load_state_dict(best_model)
                cnn_model.eval()

                all_predictions = []
                all_targets = []
                
                confusion_matrix = torch.zeros(num_classes, num_classes)

                with no_grad():
                    for batch in test_loader:
                        x_test, y_test = batch

                        # Pass the data to the appropriate device.
                        x_test = x_test.float().to(device)
                        y_test = y_test.long().to(device)

                        # make the prediction
                        y_hat = cnn_model(x_test)
                        
                        # Calculate the loss.
                        loss = loss_function(y_hat, y_test)

                        testing_loss.append(loss.item())
                        
                        # collect data for confusion matrix
                        y_hat = torch.argmax(y_hat, dim=-1)

                # store confusion matrix for plotting/later inspection or print it out
                torch.save(confusion_matrix, 'confusion_matrix.pt')

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')
                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()

# EOF
