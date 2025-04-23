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
    parser = argparse.ArgumentParser(description='Train a neural network for Pokemon battle prediction')
    
    # Data parameters
    parser.add_argument('--train-features', type=str, default='pokemon_X_train.csv',
                        help='Input preprocessed training features file')
    parser.add_argument('--train-labels', type=str, default='pokemon_y_train.csv',
                        help='Input preprocessed training labels file')
    parser.add_argument('--val-features', type=str, default='pokemon_X_val.csv',
                        help='Input preprocessed validation features file')
    parser.add_argument('--val-labels', type=str, default='pokemon_y_val.csv',
                        help='Input preprocessed validation labels file')
    
    # Model architecture
    parser.add_argument('--layers', type=int, nargs='+', default=[100, 50],
                        help='List of hidden layer sizes')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function for hidden layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate to apply after each hidden layer (0 = no dropout)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'rmsprop', 'adagrad'],
                        help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
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
        args.exp_name = f"pokemon_exp_{timestamp}"
    
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

def create_tf_dataset(X, y, batch_size, shuffle=True):
    """Create a TensorFlow dataset from pandas DataFrames."""
    dataset = tf.data.Dataset.from_tensor_slices((X.values.astype('float32'), y.values.astype('float32')))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    # Parse arguments
    args = parse_args()
    
    # Output files based on experiment name
    model_filename = f"{args.exp_name}_model.keras"
    learning_curve_filename = f"{args.exp_name}_learning_curve.png"
    
    print(f"Starting experiment: {args.exp_name}")
    print(f"Configuration: {vars(args)}")
    
    # Load the training and validation data
    X_train = pd.read_csv(args.train_features)
    y_train = pd.read_csv(args.train_labels)
    X_val = pd.read_csv(args.val_features)
    y_val = pd.read_csv(args.val_labels)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(X_train, y_train, args.batch_size)
    val_dataset = create_tf_dataset(X_val, y_val, args.batch_size, shuffle=False)
    
    # Get input shape from the data
    input_shape = (X_train.shape[1],)
    
    # Build the model
    tf.random.set_seed(42)
    model = build_model(input_shape, args.layers, args.activation, args.dropout)
    print(model.summary())
    
    # Compile the model
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    model.compile(
        loss=args.loss,
        optimizer=optimizer,
        metrics=[
            "accuracy",
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # Evaluate the model
    train_metrics = model.evaluate(train_dataset)
    val_metrics = model.evaluate(val_dataset)
    
    metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    
    print("\nTraining metrics:")
    for name, value in zip(metric_names, train_metrics):
        print(f"  {name}: {value:.4f}")
    
    print("\nValidation metrics:")
    for name, value in zip(metric_names, val_metrics):
        print(f"  {name}: {value:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # Plot Precision/Recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Precision/Recall Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(learning_curve_filename)
    print(f"Learning curves saved to {learning_curve_filename}")
    
    # Save the model
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
    
    # Save experiment results
    os.makedirs('experiments', exist_ok=True)
    results = {
        'experiment_name': args.exp_name,
        'configuration': vars(args),
        'epochs_trained': len(history.history['loss']),
        'final_metrics': {
            'train': dict(zip(metric_names, [float(m) for m in train_metrics])),
            'validation': dict(zip(metric_names, [float(m) for m in val_metrics]))
        }
    }
    
    with open(f'experiments/{args.exp_name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Experiment results saved to experiments/{args.exp_name}_results.json")

if __name__ == "__main__":
    main() 