#!/usr/bin/env python3
# filepath: /home/bcarolle/multilayer-perception/src/analyze_data.py

"""
This script analyzes dataset files to count the number of malignant (M) and benign (B) samples.
It can analyze both raw data and processed data files with one-hot encoded labels.
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to allow imports from mlp_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_tools.process_data_train import ProcessData


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Analyze dataset distribution')
    parser.add_argument('--file-path', type=str, required=True,
                        help='Path to the data file to analyze')
    parser.add_argument('--type', type=str, choices=['raw', 'processed', 'predictions'],
                        default='processed',
                        help='Type of file to analyze (raw, processed, or predictions)')
    return parser.parse_args()


def analyze_raw_data(file_path):
    """
    Analyze a raw dataset file where labels are directly in a column.
    
    Args:
        file_path: Path to the raw data file
        
    Returns:
        counts: Dictionary with counts of benign and malignant samples
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if 'diagnosis' column exists (typical format for breast cancer datasets)
        if 'diagnosis' in df.columns:
            label_column = 'diagnosis'
        else:
            # Try to find a column that might contain 'B' and 'M' values
            potential_label_columns = []
            for col in df.columns:
                unique_vals = df[col].unique()
                if set(['B', 'M']).issubset(set(unique_vals)) and len(unique_vals) <= 5:
                    potential_label_columns.append(col)
            
            if not potential_label_columns:
                print(f"Error: Could not find a label column in {file_path}")
                print(f"Available columns: {df.columns.tolist()}")
                return None
            
            label_column = potential_label_columns[0]
            print(f"Using column '{label_column}' as the label column")
        
        # Count benign and malignant samples
        counts = df[label_column].value_counts()
        
        print(f"\nAnalysis of raw data file: {file_path}")
        print(f"Total samples: {len(df)}")
        print(f"Benign (B): {counts.get('B', 0)} ({counts.get('B', 0)/len(df)*100:.1f}%)")
        print(f"Malignant (M): {counts.get('M', 0)} ({counts.get('M', 0)/len(df)*100:.1f}%)")
        
        return {
            'total': len(df),
            'benign': counts.get('B', 0),
            'malignant': counts.get('M', 0)
        }
    
    except Exception as e:
        print(f"Error analyzing raw data: {e}")
        return None


def analyze_processed_data(file_path):
    """
    Analyze a processed dataset file where labels are one-hot encoded.
    
    Args:
        file_path: Path to the processed data file
        
    Returns:
        counts: Dictionary with counts of benign and malignant samples
    """
    try:
        # Use ProcessData to load the data
        _, y_data, _, _ = ProcessData.get_data(file_path)
        
        # Count benign and malignant samples
        # Assuming label format is one-hot encoded [1,0] for benign, [0,1] for malignant
        y_classes = np.argmax(y_data, axis=1)
        benign_count = np.sum(y_classes == 0)
        malignant_count = np.sum(y_classes == 1)
        total_count = len(y_data)
        
        print(f"\nAnalysis of processed data file: {file_path}")
        print(f"Total samples: {total_count}")
        print(f"Benign (B): {benign_count} ({benign_count/total_count*100:.1f}%)")
        print(f"Malignant (M): {malignant_count} ({malignant_count/total_count*100:.1f}%)")
        
        return {
            'total': total_count,
            'benign': benign_count,
            'malignant': malignant_count
        }
    
    except Exception as e:
        print(f"Error analyzing processed data: {e}")
        return None


def analyze_predictions_data(file_path):
    """
    Analyze a predictions file where there are predicted labels and possibly true labels.
    
    Args:
        file_path: Path to the predictions file
        
    Returns:
        counts: Dictionary with counts and analysis of predictions
    """
    try:
        df = pd.read_csv(file_path)
        
        print(f"\nAnalysis of predictions file: {file_path}")
        print(f"Total samples: {len(df)}")
        
        # Count predicted labels
        if 'Prediction' in df.columns:
            pred_counts = df['Prediction'].value_counts()
            print(f"Predicted Benign (B): {pred_counts.get('B', 0)} ({pred_counts.get('B', 0)/len(df)*100:.1f}%)")
            print(f"Predicted Malignant (M): {pred_counts.get('M', 0)} ({pred_counts.get('M', 0)/len(df)*100:.1f}%)")
        
        # If true labels are available, calculate accuracy and confusion matrix
        if 'True_Label' in df.columns and 'Prediction' in df.columns:
            accuracy = (df['True_Label'] == df['Prediction']).mean()
            print(f"Accuracy: {accuracy:.4f}")
            
            # Confusion matrix
            true_b = (df['True_Label'] == 'B').sum()
            true_m = (df['True_Label'] == 'M').sum()
            
            true_b_pred_b = ((df['True_Label'] == 'B') & (df['Prediction'] == 'B')).sum()
            true_b_pred_m = ((df['True_Label'] == 'B') & (df['Prediction'] == 'M')).sum()
            true_m_pred_b = ((df['True_Label'] == 'M') & (df['Prediction'] == 'B')).sum()
            true_m_pred_m = ((df['True_Label'] == 'M') & (df['Prediction'] == 'M')).sum()
            
            print("\nConfusion Matrix:")
            print(f"                Predicted B  Predicted M")
            print(f"True Benign (B)    {true_b_pred_b:4d}         {true_b_pred_m:4d}")
            print(f"True Malignant (M) {true_m_pred_b:4d}         {true_m_pred_m:4d}")
            
            # Calculate metrics
            specificity = true_b_pred_b / true_b if true_b > 0 else 0
            sensitivity = true_m_pred_m / true_m if true_m > 0 else 0
            
            print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
            print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
            
            return {
                'total': len(df),
                'predicted_benign': pred_counts.get('B', 0),
                'predicted_malignant': pred_counts.get('M', 0),
                'true_benign': true_b,
                'true_malignant': true_m,
                'accuracy': accuracy,
                'specificity': specificity,
                'sensitivity': sensitivity
            }
        
        return {
            'total': len(df),
            'predicted_benign': pred_counts.get('B', 0) if 'Prediction' in df.columns else 0,
            'predicted_malignant': pred_counts.get('M', 0) if 'Prediction' in df.columns else 0
        }
    
    except Exception as e:
        print(f"Error analyzing predictions data: {e}")
        return None


def main():
    args = parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} does not exist")
        return
    
    if args.type == 'raw':
        results = analyze_raw_data(args.file_path)
    elif args.type == 'processed':
        results = analyze_processed_data(args.file_path)
    elif args.type == 'predictions':
        results = analyze_predictions_data(args.file_path)
    
    if results:
        print("\nAnalysis complete!")
    else:
        print("\nAnalysis failed.")


if __name__ == "__main__":
    main()
