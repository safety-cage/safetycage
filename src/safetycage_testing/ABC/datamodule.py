from abc import ABC, abstractmethod
from typing import List, Any
from pathlib import Path

class DataModule(ABC):
    """Abstract base class for handling batched data regardless of source"""
    
    def __init__(
        self,
        data_dir:str=None,
        from_cache:bool = False,
        batch_size: int = 32,
        device:str="cpu"
        ) -> None:

        # Data parameters
        self.from_cache = from_cache
        self.batch_size = batch_size
        
        # Device configuration
        self.device = device
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)


    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset"""
        raise NotImplementedError("Subclasses should implement this method to return the number of classes.")


    @property
    @abstractmethod
    def classes(self) -> List[Any]:
        """Returns the class names in the dataset"""
        raise NotImplementedError("Subclasses should implement this method to return the class names.")


    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Get the name of the dataset."""
        raise NotImplementedError("Subclasses should implement this method to return the dataset name.")


    @abstractmethod
    def setup(self) -> None:
        """Setup the data module, should be implemented by subclasses"""
        raise NotImplementedError("Subclasses should implement this method to setup the data module.")


    @abstractmethod
    def _load_data(self, filepath:str) -> None:
        """Load data from the source or cache"""
        raise NotImplementedError("Subclasses should implement this method to load data.")


    @abstractmethod
    def _transform(self,x,y):
        """Transform the data, should be implemented by subclasses"""
        raise NotImplementedError("Subclasses should implement this method to transform the data.")


    @abstractmethod
    def _split(self,x,y,split):
        """Split the data into training and validation sets."""
        raise NotImplementedError("Subclasses should implement this method to split the data.")
    
    @abstractmethod
    def to_joblib(self, path: str) -> None:
        """Save the data module to a joblib file."""
        raise NotImplementedError("Subclasses should implement this method to save the data module to a joblib file.")
    
    @abstractmethod
    def from_joblib(self, path: str) -> None:
        """Load the data module from a joblib file."""
        raise NotImplementedError("Subclasses should implement this method to load the data module from a joblib file.")