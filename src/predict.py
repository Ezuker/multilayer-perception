from mlp_core import network


def argparse():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict using a trained model.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model.", default="models/model.pkl"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input data.", default="data/processed/validation.csv"
    )
    return parser.parse_args()


def predict(model: network.Network, x_data):
    """
    Make predictions using the trained model.

    Args:
        model: Trained model.
        data: Input data for prediction (tuple from ProcessData.get_data).

    Returns:
        predictions: Model predictions and processed results (classes and probabilities).
    """
    import numpy as np
    predictions = model.forward(x_data)
    
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


def main():
    try:
        from data_tools import process_data_train
        import pandas as pd
        import os

        args = argparse()

        model = network.Network.load(args.model_path)
        x_val, _, _, _ = process_data_train.ProcessData.get_data(args.data_path)

        results = predict(model, x_val)
        
        df_results = pd.DataFrame({
            'Prediction': results['class_labels'],
            'Confidence': results['confidence'],
            'Benign_Prob': results['raw_predictions'][:, 0],
            'Malignant_Prob': results['raw_predictions'][:, 1]
        })
        
        print("\n===== Prediction Results =====")
        print(f"Total samples: {len(df_results)}")
        print(f"Predicted Benign (B): {sum(results['class_labels'] == 'B')}")
        print(f"Predicted Malignant (M): {sum(results['class_labels'] == 'M')}")
        print(f"Average confidence: {results['confidence'].mean():.4f}")
        
        print("\n===== Sample Predictions (first 10) =====")
        print(df_results.head(10))
        
        output_path = args.data_path.replace(".csv", "_predictions.csv")
        output_path = output_path.replace("processed", "predictions")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"\nSaved predictions to: {output_path}")
        return df_results
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


if __name__ == "__main__":
    main()