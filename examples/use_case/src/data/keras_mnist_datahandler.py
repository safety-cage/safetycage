from typing import Optional, Dict, Tuple
import joblib
import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(search_from=Path(__file__), indicator=[".project-root"], pythonpath=True)
from safetycage_testing.ABC.base_datahandler import BaseDataHandler

class KerasMNISTDataHandler(BaseDataHandler):
    """Handler for MNIST data from Keras"""
    
    def __init__(
        self,
        data_dir = None,
        from_cache: bool = True,
        batch_size: int = 128,
        val_split: float = 0.2,
        use_onehot_encoder:bool = False,
        device:str = "cpu"
        ):
        super().__init__(data_dir, from_cache, batch_size, device)

        self.val_split = val_split
        self.use_onehot_encoder = use_onehot_encoder
        
        # setup the data handler
        self.setup()
        

    @property
    def classes(self) -> dict:
        """Get the class names."""
        return [i for i in range(0, 10)]
        # return {i: str(i) for i in range(10)}
    
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.classes)

    @property
    def dataset_name(self) -> str:
        """Get the name of the dataset."""
        return "mnist"

    def setup(self)->None:
        """Setup the data handler by loading and transforming the dataset."""
        # Load data if saved, else download
        
        path_data = (self.data_dir / self.dataset_name).with_suffix(".npz")
        
        (x_train, y_train), (x_test, y_test) = self._load_data(path_data)

        x_train, y_train = self._transform(x_train, y_train)
        x_test, y_test = self._transform(x_test, y_test)
        
        # Split data and stratify w.r.t. label y
        (x_train, y_train), (x_val, y_val) = self._split(
            x = x_train,
            y = y_train,
            split = self.val_split
            )

        # Store the base data (without predictions)
        self.data_train = (x_train, y_train)
        self.data_val= (x_val, y_val)
        self.data_test = (x_test, y_test)

        
    def _load_data(self, filepath):
        """Load the MNIST dataset. If the dataset is not found at the specified path, it will be downloaded from Keras.

        Args:
            filepath (Path): The path to the dataset file.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The training and testing data.
        """

        if self.from_cache:
            try:
                data = np.load(filepath)
                return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])
            
            except (FileNotFoundError, IOError):
                print(f"Data not found at {filepath}.\nDownloading from Keras...")
                return self._download_data(filepath)
            
        else:
            print(f"Downloading MNIST data from Keras...")
            return self._download_data(filepath)


    def _download_data(self, path)-> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Download the MNIST dataset from Keras.

        Args:
            path (str): The path to save the dataset.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The training and testing data.
        """
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download data from keras
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path)

        return (x_train, y_train), (x_test, y_test)
    
    
    def _transform(self, x, y) -> None:
        """Transform the data for use in MLP.

        Args:
            x (np.ndarray): The input features.
            y (np.ndarray): The input labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The transformed features and labels.
        """
        
        # Reshape to (samples, 28*28) for MLP and normalize
        x = x.reshape((-1, 28*28))
        x = x.astype('float32') / 255.0
        
        # Apply one-hot encoding if true
        if self.use_onehot_encoder:
            y = keras.utils.to_categorical(y, num_classes=self.num_classes)

        return x, y


    def _split(self, x, y, split) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Split the dataset into training and testing sets.

        Args:
            x (np.ndarray): The input features.
            y (np.ndarray): The input labels.
            split (float): The proportion of the dataset to include in the test split.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The training and testing data.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, stratify=y, random_state=42, test_size=split
        )
        
        return (x_train, y_train), (x_test, y_test)


    def train_dataset(self) -> tf.data.Dataset:
        """Get the training dataset.

        Returns:
            tf.data.Dataset: The training dataset.
        """
        return tf.data.Dataset.from_tensor_slices(self.data_train).batch(self.batch_size)

    
    def val_dataset(self) -> tf.data.Dataset:
        """Get the validation dataset.

        Returns:
            tf.data.Dataset: The validation dataset.
        """
        return tf.data.Dataset.from_tensor_slices(self.data_val).batch(self.batch_size)

    
    def test_dataset(self) -> tf.data.Dataset:
        """Get the test dataset.

        Returns:
            tf.data.Dataset: The test dataset.
        """
        return tf.data.Dataset.from_tensor_slices(self.data_test).batch(self.batch_size)


    def set_predictions(self, predictions: Dict[str, np.ndarray]) -> None:
        """Set model predictions for all datasets.

        Args:
            predictions (Dict[str, np.ndarray]): Dictionary with keys 'train', 'val', and 'test'
                                                  containing prediction arrays for each dataset.
        """
        
        # Extract predictions
        y_train_pred = predictions.get('y_pred_train')
        y_val_pred = predictions.get('y_pred_val')
        y_test_pred = predictions.get('y_pred_test')

        # Validate predictions
        if y_train_pred is None or y_val_pred is None or y_test_pred is None:
            raise ValueError("Predictions dictionary must contain 'y_pred_train', 'y_pred_val', and 'y_pred_test' keys")

        # Update data tuples with predictions
        self.data_train = (self.data_train[0], self.data_train[1], y_train_pred)
        self.data_val = (self.data_val[0], self.data_val[1], y_val_pred)
        self.data_test = (self.data_test[0], self.data_test[1], y_test_pred)
        
    def to_joblib(self, path: Optional[str] = None):
        """Save the data handler to a joblib file.

        Args:
            path (Optional[str], optional): The path to save the joblib file. Defaults to None.
        """
        
        if path is None:
            path = (self.data_dir/f"{self.dataset_name}_data_handler").with_suffix(".joblib")
        
        # Create parent directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, path)
            
            
    def from_joblib(self, path: Optional[str] = None):
        """Load the data handler from a joblib file."""
        
        if path is None:
            path = (self.data_dir/f"{self.dataset_name}_data_handler").with_suffix(".joblib")

        return joblib.load(path)
    
    
    def plot_samples(self, n_samples_per_class: int = 5, fig_scale:int = .5, cmap:str = "gray_r") -> None:
        """Plot sample images from the dataset.

        Args:
            n_samples_per_class  (int, optional): Number of samples per class to plot. Defaults to 5.
        """
        import matplotlib.pyplot as plt

        x_train, y_train = self.data_train[:2]
        
        # Handle one-hot encoded labels or (N, 1) shape
        if self.use_onehot_encoder:
            y_labels = np.argmax(y_train, axis=1)
        else:
            y_labels = y_train.flatten()

        fig, axes = plt.subplots(n_samples_per_class, len(self.classes), figsize=(len(self.classes) * fig_scale, n_samples_per_class * fig_scale), squeeze=True)
        
        for class_id, class_name in self.classes.items():
            # Find indices for this class
            idx = np.where(y_labels == class_id)[0]
            selected_idx = idx[:n_samples_per_class]
            
            for i in range(n_samples_per_class):
                ax = axes[i, class_id]
                if i < len(selected_idx):
                    img_idx = selected_idx[i]
                    img = x_train[img_idx].reshape(28,28)
                    ax.imshow(img, cmap=cmap)
                
                    if i == 0:
                        ax.set_title(class_name)
                ax.axis("off")
        
        plt.tight_layout()
        plt.show()
        