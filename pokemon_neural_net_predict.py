#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import keras
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions from a trained Pokemon battle prediction model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--test-features', type=str, default='pokemon_X_test.csv',
                        help='Input preprocessed test features file')
    parser.add_argument('--test-labels', type=str, default='pokemon_y_test.csv',
                        help='Input preprocessed test labels file (optional for evaluation)')
    parser.add_argument('--train-features', type=str, default='pokemon_X_train.csv',
                        help='Input preprocessed training features file (for column checking)')
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output files (default: derived from model name)')
    
    args = parser.parse_args()
    
    # Generate output prefix if not provided
    if args.output_prefix is None:
        args.output_prefix = args.model.split('_model.keras')[0]
    
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Output filenames
    predictions_filename = f"{args.output_prefix}_predictions.csv"
    
    # Load the test data
    X_test = pd.read_csv(args.test_features)
    
    # Ensure all values are numeric and handle NaN values
    X_test = X_test.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, force non-numeric to NaN
    X_test = X_test.fillna(0)  # Replace NaN with 0 (or other appropriate value)

    # Print shape to ensure it matches the model's expected input
    print(f"X_test shape: {X_test.shape}")
    
    # Check if test labels exist for evaluation
    try:
        y_test = pd.read_csv(args.test_labels)
        have_labels = True
    except:
        have_labels = False
        print("No test labels found. Will only generate predictions.")
    
    # Load the trained model
    model = keras.saving.load_model(args.model)
    print(model.summary())
    
    # After loading X_test, add these lines:
    print("X_test info:")
    print(X_test.info())
    print("X_test head:")
    print(X_test.head())
    print("Any NaN values:", X_test.isna().any().any())
    
    # Load a few rows of training data to check feature consistency
    X_train_sample = pd.read_csv(args.train_features, nrows=1)
    print("Training data columns:", X_train_sample.columns.tolist())
    print("Test data columns:", X_test.columns.tolist())

    # Check if columns match
    if not all(c1 == c2 for c1, c2 in zip(X_train_sample.columns, X_test.columns)):
        print("WARNING: Column mismatch between training and test data!")
        # Reorder test columns to match training data
        X_test = X_test[X_train_sample.columns]
    
    # Generate predictions
    X_test_array = X_test.values.astype('float32')
    y_pred_proba = model.predict(X_test_array)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Predicted_Probability': y_pred_proba.flatten(),
        'Predicted_Winner': y_pred.flatten()
    })
    
    # If we have actual labels, evaluate the model
    if have_labels:
        predictions_df['Actual_Winner'] = y_test.values
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\nTest set evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        # Add metrics to the filename
        predictions_filename = f"{args.output_prefix}_predictions_acc{accuracy:.4f}.csv"
    
    # Save predictions to CSV
    predictions_df.to_csv(predictions_filename, index=False)
    print(f"Predictions saved to {predictions_filename}")

if __name__ == "__main__":
    main() 