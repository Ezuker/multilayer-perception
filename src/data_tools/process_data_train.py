import numpy as np
class ProcessData:
    def __init__(self):
        pass

    @staticmethod
    def load_data(train_path: str, validation_path: str) -> tuple:
        """
        Load training and validation data from CSV files.

        Args:
            train_path (str): Path to the training data CSV file.
            validation_path (str): Path to the validation data CSV file.

        Returns:
            tuple: A tuple containing two DataFrames: (train_data, validation_data).
        """
        import pandas as pd
        train_data = pd.read_csv(train_path)
        validation_data = pd.read_csv(validation_path)

        return train_data, validation_data
    
    @staticmethod
    def get_data(train_path: str, validation_path: str) -> tuple:
        """
        Get training and validation data.
        
        Args:
            train_path (str): Path to the training data CSV file.
            validation_path (str): Path to the validation data CSV file.
            
        Returns:
            tuple: (x_train, y_train, x_val, y_val, x_mean, x_std)
        """
        train_data, val_data = ProcessData.load_data(train_path, validation_path)
        x_train = train_data.iloc[:, 2:].values
        y_train_raw = train_data.iloc[:, 1].values
        
        x_val = val_data.iloc[:, 2:].values
        y_val_raw = val_data.iloc[:, 1].values
        
        # Normalize the data
        x_mean = np.mean(x_train, axis=0)
        x_std = np.std(x_train, axis=0) + 1e-15
        x_train = (x_train - x_mean) / x_std
        x_val = (x_val - x_mean) / x_std
        
        label_mapping = {'B': 0, 'M': 1}
        
        y_train = np.array([label_mapping[label] for label in y_train_raw])
        y_val = np.array([label_mapping[label] for label in y_val_raw])
        
        y_train_onehot = np.eye(2)[y_train]
        y_val_onehot = np.eye(2)[y_val]
        
        print(f"X train shape: {x_train.shape}, Y train shape: {y_train_onehot.shape}")
        print(f"X validation shape: {x_val.shape}, Y validation shape: {y_val_onehot.shape}")
        
        return x_train, y_train_onehot, x_val, y_val_onehot