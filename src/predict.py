from mlp_core import network
import numpy as np


def argparse():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict using a trained model.")
    parser.add_argument(
        "--model-path", type=str, required=True, nargs='+', help="Path to the trained model.", default="models/model.pkl"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the input data.", default="data/processed/validation.csv"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed output including raw predictions."
    )
    return parser.parse_args()


def predict(model: network.Network, x_data):
    """
    Make predictions using the trained model.

    Args:
        model: Trained model.
        x_data: Input feature data.

    Returns:
        results: Dictionary containing model predictions and processed results (classes and probabilities).
    """
    predictions = model.predict(x_data)
    
    class_indices = np.argmax(predictions, axis=1)
    class_labels = np.array(['B', 'M'])[class_indices]
    
    confidence = np.max(predictions, axis=1)
    
    results = {
        'raw_predictions': predictions,
        'class_indices': class_indices,
        'class_labels': class_labels,
        'confidence': confidence
    }
    return results


def binary_cross_entropy(y_true, y_pred, verbose=False):
    """
    Calculate binary cross-entropy error function.
    
    E = -(1/N) * Σ[y*log(p) + (1-y)*log(1-p)]
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_pred: Predicted probabilities
        verbose: Whether to print debug information
        
    Returns:
        error: Binary cross-entropy error
    """    
    loss = lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-15) + 
                                                   (1 - y_true) * np.log(1 - y_pred + 1e-15))
    
    return loss(y_true, y_pred)


def main():
    # try:
        from data_tools import process_data_train
        import pandas as pd
        import os

        args = argparse()

        for model_path in args.model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = network.Network.load(model_path)
            x_val, y_val, _, _ = process_data_train.ProcessData.get_data(args.data_path)

            results = predict(model, x_val)
            
            bce_error = binary_cross_entropy(y_val, results['raw_predictions'], verbose=args.verbose)
            
            y_true_classes = np.argmax(y_val, axis=1)
            y_true_labels = np.array(['B', 'M'])[y_true_classes]

            df_results = pd.DataFrame({
                'True_Label': y_true_labels,
                'Prediction': results['class_labels'],
                'Correct': y_true_labels == results['class_labels'],
                'Confidence': results['confidence'],
                'Benign_Prob': results['raw_predictions'][:, 0],
                'Malignant_Prob': results['raw_predictions'][:, 1]
            })
            
            # Print results summary
            print("\n===== Prediction Results =====")
            print(f"Total samples: {len(df_results)}")
            print(f"Predicted Benign (B): {sum(results['class_labels'] == 'B')}")
            print(f"Predicted Malignant (M): {sum(results['class_labels'] == 'M')}")
            print(f"Correct predictions: {sum(df_results['Correct'])} ({sum(df_results['Correct'])/len(df_results):.4f})")
            print(f"Average confidence: {results['confidence'].mean():.4f}")
            print(f"Binary Cross-Entropy Error: {bce_error:.6f}")
            
            accuracy = np.mean(results['class_indices'] == y_true_classes)
            print(f"Accuracy: {accuracy:.4f}")
            
            print("\n===== Sample Predictions (first 10) =====")
            print(df_results.head(10))

            print("\n===== False Predictions (first 10) =====")
            print(df_results[~df_results['Correct']].head(10))
            
            output_path = args.data_path.replace(".csv", "_predictions.csv")
            output_path = output_path.replace("processed", "predictions")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_results.to_csv(output_path, index=False)
            print(f"\nSaved predictions to: {output_path}")
    
    # except Exception as e:
    #     print(f"Error during prediction: {e}")
    #     return None


if __name__ == "__main__":
    main()
