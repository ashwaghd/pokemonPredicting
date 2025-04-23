#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import keras
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions from a trained neural network')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--input-file', type=str, default='depression-preprocessed-test.csv',
                        help='Input preprocessed test data file')
    parser.add_argument('--label', type=str, default='Depression',
                        help='Target column name')
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
    proba_filename = f"{args.output_prefix}_predictions_proba.csv"
    predictions_filename = f"{args.output_prefix}_predictions.csv"
    
    # Load the test data, removing the label column, if it exists
    dataframe = pd.read_csv(args.input_file, index_col=0)
    if args.label in dataframe.columns:
        X = dataframe.drop(args.label, axis=1)
    else:
        X = dataframe
    
    # Load the trained model
    model = keras.saving.load_model(args.model)
    print(model.summary())
    
    # Predict the labels
    y_hat = model.predict(X)
    y_hat_binary = (y_hat > 0.5).astype(int)
    
    # Save probabilistic predictions
    merged = dataframe.index.to_frame()
    merged[args.label] = y_hat[:,0]
    merged.to_csv(proba_filename, index=False)
    print(f"Probabilistic predictions saved to {proba_filename}")
    
    # Save binary predictions
    merged = dataframe.index.to_frame()
    merged[args.label] = y_hat_binary[:,0]
    merged.to_csv(predictions_filename, index=False)
    print(f"Binary predictions saved to {predictions_filename}")

if __name__ == "__main__":
    main()

