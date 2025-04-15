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
            tuple: A tuple containing two DataFrames: (train_data, validation_data).
        """
        train_data, val_data = ProcessData.load_data(train_path, validation_path)
        x_train = train_data.iloc[:, 2:].values
        y_train = train_data.iloc[:, 1].values

        x_val = val_data.iloc[:, 2:].values
        y_val = val_data.iloc[:, 1].values

        # Normalize the data
        x_mean = np.mean(x_train, axis=0)
        x_std = np.std(x_train, axis=0)
        x_train = (x_train - x_mean) / x_std
        x_val = (x_val - x_mean) / x_std
        # Convert labels to one-hot encoding
        for i, label in enumerate(np.unique(y_train)):
            y_train[y_train == label] = i
        for i, label in enumerate(np.unique(y_val)):    
            y_val[y_val == label] = i
        return x_train, y_train, x_val, y_val, x_mean, x_std