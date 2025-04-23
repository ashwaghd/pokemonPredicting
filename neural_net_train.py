#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import argparse
import json
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network for depression prediction')
    
    # Data parameters
    parser.add_argument('--input-file', type=str, default='depression-preprocessed-train.csv',
                        help='Input preprocessed training data file')
    parser.add_argument('--label', type=str, default='Depression',
                        help='Target column name')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of data to use for training vs validation')
    
    # Model architecture
    parser.add_argument('--layers', type=int, nargs='+', default=[100],
                        help='List of hidden layer sizes')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function for hidden layers')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate to apply after each hidden layer (0 = no dropout)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'rmsprop', 'adagrad'],
                        help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Output parameters
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    
    # Loss function parameters
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                        choices=['binary_crossentropy', 'hinge', 'squared_hinge'],
                        help='Loss function to use')
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"exp_{timestamp}"
    
    return args

def get_optimizer(name, learning_rate):
    """Return the specified optimizer with the given learning rate."""
    if name == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif name == 'adagrad':
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def build_model(input_shape, layer_sizes, activation, dropout_rate):
    """Build a model with the specified architecture."""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    for units in layer_sizes:
        model.add(keras.layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    
    # Output layer is always sigmoid for binary classification
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    return model

def lr_scheduler(epoch, learning_rate, decay_factor=0.005):
    """Schedule the learning rate with exponential decay after 10 epochs."""
    if epoch >= 10:
        return learning_rate * float(tf.exp(-decay_factor))
    return learning_rate

def save_experiment_results(args, history, epochs, train_metrics, val_metrics):
    """Save experiment configuration and results."""
    # Create experiments directory if it doesn't exist
    os.makedirs('experiments', exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'experiment_name': args.exp_name,
        'configuration': vars(args),
        'epochs_trained': epochs,
        'final_metrics': {
            'train_loss': train_metrics[0],
            'train_auc': train_metrics[1],
            'val_loss': val_metrics[0],
            'val_auc': val_metrics[1]
        }
    }
    
    # Save to JSON file
    with open(f'experiments/{args.exp_name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Experiment results saved to experiments/{args.exp_name}_results.json")
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Output files based on experiment name
    model_filename = f"{args.exp_name}_model.keras"
    learning_curve_filename = f"{args.exp_name}_learning_curve.png"
    
    print(f"Starting experiment: {args.exp_name}")
    print(f"Configuration: {vars(args)}")
    
    # Load the training dataframe, separate into X/y
    dataframe = pd.read_csv(args.input_file, index_col=0)
    X = dataframe.drop(args.label, axis=1)
    y = dataframe[args.label]
    
    # Prepare a tensorflow dataset from the dataframe
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Find the shape of the inputs
    for features, labels in dataset.take(1):
        input_shape = features.shape
        output_shape = labels.shape
    
    # Split the dataset into train and validation sets
    dataset_size = dataset.cardinality().numpy()
    train_size = int(args.train_ratio * dataset_size)
    validate_size = dataset_size - train_size
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)
    
    # Shuffle and batch datasets
    train_dataset = train_dataset.shuffle(buffer_size=train_size)
    validation_dataset = validation_dataset.shuffle(buffer_size=validate_size)
    train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build the model
    tf.random.set_seed(42)
    model = build_model(input_shape, args.layers, args.activation, args.dropout)
    print(model.summary())
    
    # Compile the model
    loss = args.loss
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=["AUC"])
    
    # Callbacks
    learning_rate_callback = keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: lr_scheduler(epoch, lr))
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        x=train_dataset,
        epochs=args.epochs,
        validation_data=validation_dataset,
        callbacks=[learning_rate_callback, early_stop_callback]
    )
    epochs = len(history.epoch)
    
    # Evaluate the model
    train_metrics = model.evaluate(train_dataset)
    val_metrics = model.evaluate(validation_dataset)
    print(f"Training metrics - Loss: {train_metrics[0]:.4f}, AUC: {train_metrics[1]:.4f}")
    print(f"Validation metrics - Loss: {val_metrics[0]:.4f}, AUC: {val_metrics[1]:.4f}")
    
    # Display the learning curves
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(learning_curve_filename)
    print(f"Learning curves saved to {learning_curve_filename}")
    
    # Save the model
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
    
    # Save experiment results
    save_experiment_results(args, history, epochs, train_metrics, val_metrics)

if __name__ == "__main__":
    main()
