#!/usr/bin/env python3

import pandas as pd
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Compare incorrect predictions between two models')
    
    parser.add_argument('--predictions1', type=str, required=True,
                        help='First prediction file')
    parser.add_argument('--predictions2', type=str, required=True,
                        help='Second prediction file')
    parser.add_argument('--test-labels', type=str, default='pokemon_y_test.csv',
                        help='Actual test labels file')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load prediction files
    print(f"Loading predictions from {args.predictions1}...")
    preds1 = pd.read_csv(args.predictions1)
    
    print(f"Loading predictions from {args.predictions2}...")
    preds2 = pd.read_csv(args.predictions2)
    
    # Load actual labels
    print(f"Loading actual labels from {args.test_labels}...")
    y_test = pd.read_csv(args.test_labels)
    
    # Check if 'Actual_Winner' is already in the prediction files
    if 'Actual_Winner' in preds1.columns:
        actuals = preds1['Actual_Winner'].values
    elif 'Actual_Winner' in preds2.columns:
        actuals = preds2['Actual_Winner'].values
    else:
        actuals = y_test.values.flatten()
    
    # Get predicted winners
    pred_winners1 = preds1['Predicted_Winner'].values
    pred_winners2 = preds2['Predicted_Winner'].values
    
    # Calculate overall accuracy
    accuracy1 = np.mean(pred_winners1 == actuals)
    accuracy2 = np.mean(pred_winners2 == actuals)
    
    print(f"\nModel 1 Accuracy: {accuracy1:.4f}")
    print(f"Model 2 Accuracy: {accuracy2:.4f}")
    
    # Find incorrect predictions
    incorrect1 = pred_winners1 != actuals
    incorrect2 = pred_winners2 != actuals
    
    num_incorrect1 = np.sum(incorrect1)
    num_incorrect2 = np.sum(incorrect2)
    
    print(f"\nNumber of incorrect predictions:")
    print(f"Model 1: {num_incorrect1} ({num_incorrect1/len(actuals):.2%})")
    print(f"Model 2: {num_incorrect2} ({num_incorrect2/len(actuals):.2%})")
    
    # Find shared incorrect predictions
    shared_incorrect = np.logical_and(incorrect1, incorrect2)
    num_shared_incorrect = np.sum(shared_incorrect)
    
    print(f"\nShared incorrect predictions: {num_shared_incorrect}")
    print(f"Percentage of Model 1 incorrect also missed by Model 2: {num_shared_incorrect/num_incorrect1:.2%}")
    print(f"Percentage of Model 2 incorrect also missed by Model 1: {num_shared_incorrect/num_incorrect2:.2%}")
    
    # Find unique incorrect predictions
    only_incorrect1 = np.logical_and(incorrect1, ~incorrect2)
    only_incorrect2 = np.logical_and(~incorrect1, incorrect2)
    
    print(f"\nUnique incorrect predictions:")
    print(f"Only Model 1 missed: {np.sum(only_incorrect1)}")
    print(f"Only Model 2 missed: {np.sum(only_incorrect2)}")
    
    # Prepare detailed comparison DataFrame
    if len(preds1) <= 100:  # Only show detailed comparison for small datasets
        comparison = pd.DataFrame({
            'Actual': actuals,
            'Model1_Predicted': pred_winners1,
            'Model2_Predicted': pred_winners2,
            'Model1_Correct': pred_winners1 == actuals,
            'Model2_Correct': pred_winners2 == actuals,
            'Both_Incorrect': shared_incorrect
        })
        
        print("\nSample of battles both models predicted incorrectly:")
        print(comparison[shared_incorrect].head(10))
    
    print("\nConclusion:")
    if num_shared_incorrect/num_incorrect1 > 0.9:
        print("The models are making very similar mistakes (>90% overlap in errors).")
    elif num_shared_incorrect/num_incorrect1 > 0.7:
        print("The models have substantial overlap in their mistakes (70-90%).")
    else:
        print("The models are making different types of mistakes (<70% overlap).")

if __name__ == "__main__":
    main() 