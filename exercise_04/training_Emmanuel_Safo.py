# -*- coding: utf-8 -*-
"""
Training Script for Audio Classification with Data Augmentation
Exercise 04 - Advanced Audio Processing

This script trains a CNN model for audio classification (ESC-50 subset)
with optional data augmentation techniques including:
- Noise addition
- Pitch shifting  
- Reverberation (impulse response)
- Frequency masking (SpecAugment)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import utils
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
from my_cnn_system import MyCNNSystem
from dataset_class_template import MyDataset
from copy import deepcopy
from sklearn.metrics import confusion_matrix


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    batch_size = 4
    epochs = 30
    learning_rate = 1e-4

    cnn_1_channels = 16
    cnn_1_kernel = (5, 5)
    cnn_1_stride = (2, 2)
    cnn_1_padding = (2, 2)

    pooling_1_kernel = (3, 3)
    pooling_1_stride = (1, 1)

    cnn_2_channels = 32
    cnn_2_kernel = (5, 5)
    cnn_2_stride = (2, 2)
    cnn_2_padding = (2, 2)

    pooling_2_kernel = (3, 3)
    pooling_2_stride = (2, 2)

    classifier_input_features = 64  # n_mels
    classifier_output = 4  # 4 classes: rain, sea_waves, chainsaw, helicopter
    
    # Labels for the 4 classes
    labels = ['rain', 'sea_waves', 'chainsaw', 'helicopter']
    
    # Instantiate our DNN
    model = MyCNNSystem(
        cnn_channels_out_1=cnn_1_channels,
        cnn_kernel_1=cnn_1_kernel,
        cnn_stride_1=cnn_1_stride,
        cnn_padding_1=cnn_1_padding,
        pooling_kernel_1=pooling_1_kernel,
        pooling_stride_1=pooling_1_stride,
        cnn_channels_out_2=cnn_2_channels,
        cnn_kernel_2=cnn_2_kernel,
        cnn_stride_2=cnn_2_stride,
        cnn_padding_2=cnn_2_padding,
        pooling_kernel_2=pooling_2_kernel,
        pooling_stride_2=pooling_2_stride,
        classifier_input_features=classifier_input_features,
        output_classes=classifier_output
    )
    model = model.to(device)

    # Load data
    data_path = Path(__file__).parent
    
    # Training dataset with augmentation enabled
    ds_train = MyDataset(
        audio_dir_path=data_path / 'training',
        meta_csv_file_path=data_path / 'train_meta.csv',
        labels=labels,
        sr=44100,
        n_fft=1024,
        hop_length=512,
        n_mels=classifier_input_features,
        # Enable augmentations for training
        pitch_shift_augmentation=True,
        pitch_shift_max_steps=2.0,
        reverb_augmentation=True,
        impulses_dir_path=data_path / 'impulse_responses',
        noise_augmentation=True,
        snr_db=15.0,
        freqmask_augmentation=True,
        max_bands=20
    )
    
    # Validation dataset without augmentation
    ds_val = MyDataset(
        audio_dir_path=data_path / 'validation',
        meta_csv_file_path=data_path / 'val_meta.csv',
        labels=labels,
        sr=44100,
        n_fft=1024,
        hop_length=512,
        n_mels=classifier_input_features
    )
    
    # Test dataset without augmentation
    ds_test = MyDataset(
        audio_dir_path=data_path / 'testing',
        meta_csv_file_path=data_path / 'test_meta.csv',
        labels=labels,
        sr=44100,
        n_fft=1024,
        hop_length=512,
        n_mels=classifier_input_features
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
            
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 10
    patience_counter = 0

    val_losses, test_losses, test_accs = [], [], []
    best_model = None
    
    for epoch in range(epochs):
        epoch_loss_training = []
        epoch_loss_validation = []
        epoch_acc_training = 0
        epoch_acc_validation = 0
        train_total = 0
        val_total = 0
        
        model.train()
        for i, batch in enumerate(train_loader):
            features, targets = batch
            features = features.float().to(device)
            targets = targets.float().to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss
            loss = loss_function(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Record loss
            epoch_loss_training.append(loss.item())
            
            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(targets, dim=1)
            epoch_acc_training += (predicted == true_labels).sum().item()
            train_total += targets.size(0)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                features, targets = batch
                features = features.float().to(device)
                targets = targets.float().to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Calculate loss
                loss = loss_function(outputs, targets)
                epoch_loss_validation.append(loss.item())
                
                # Calculate accuracy
                predicted = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(targets, dim=1)
                epoch_acc_validation += (predicted == true_labels).sum().item()
                val_total += targets.size(0)

        # Calculate mean losses
        epoch_loss_training = np.mean(epoch_loss_training)
        epoch_loss_validation = np.mean(epoch_loss_validation)
        epoch_acc_training = 100 * epoch_acc_training / train_total
        epoch_acc_validation = 100 * epoch_acc_validation / val_total
        
        print(f"Epoch {epoch}")
        print(f"Training loss {epoch_loss_training:.4f} acc {epoch_acc_training:.1f}%")
        print(f"Validation loss {epoch_loss_validation:.4f} acc {epoch_acc_validation:.1f}%")

        # Check early stopping conditions
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            best_validation_epoch = epoch
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            val_losses.append(epoch_loss_validation)
        else:
            patience_counter += 1

        # If we have to stop, do the testing
        if (patience_counter >= patience) or (epoch == epochs - 1):
            print('\nExiting training', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss:.4f}', end='\n\n')
            
            if best_model is None:
                print('No best model.')
            else:
                print('Starting testing', end=' | ')
                testing_loss = []
                testing_acc = 0
                test_total = 0
                pred = []
                gt = []

                # Load best model
                model.load_state_dict(best_model)
                model.eval()
                
                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        features, targets = batch
                        features = features.float().to(device)
                        targets = targets.float().to(device)
                        
                        # Forward pass
                        outputs = model(features)
                        
                        # Calculate loss
                        loss = loss_function(outputs, targets)
                        testing_loss.append(loss.item())
                        
                        # Calculate accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        true_labels = torch.argmax(targets, dim=1)
                        testing_acc += (predicted == true_labels).sum().item()
                        test_total += targets.size(0)
                        
                        # Store predictions and ground truth for confusion matrix
                        pred.extend(predicted.cpu().numpy())
                        gt.extend(true_labels.cpu().numpy())
                
                testing_loss = np.mean(testing_loss)
                test_losses.append(testing_loss)
                test_accs.append(100 * testing_acc / test_total)
                
                print(f'Testing loss: {testing_loss:.4f} acc: {100 * testing_acc / test_total:.1f}%')
                
                # Calculate and plot confusion matrix
                cm = confusion_matrix(gt, pred)
                print('\nConfusion Matrix:')
                plt.figure(figsize=(8, 6))
                utils.plot_confusion_matrix(cm, labels, normalize=True, 
                                           title='Confusion Matrix (Normalized)')
                plt.savefig(data_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
                print(f'Confusion matrix saved to {data_path / "confusion_matrix.png"}')
                # plt.show()  # Commented out for non-interactive environments
                
            break

    print(f'\nValidation losses: {val_losses}')
    print(f'Testing losses: {test_losses}')
    print(f'Testing accuracies: {test_accs}')

    if len(val_losses) > 0:
        print(f'\nValidation losses mean: {np.mean(val_losses):.4f}')
    if len(test_losses) > 0:
        print(f'Testing losses mean: {np.mean(test_losses):.4f}')
    if len(test_accs) > 0:
        print(f'Testing accuracies mean: {np.mean(test_accs):.4f}%', end='\n\n')


if __name__ == '__main__':
    main()
