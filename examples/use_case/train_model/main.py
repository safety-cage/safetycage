import pyrootutils
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root_use_case = root / "use_case"
data_dir = root_use_case / "data/cifar10"

from use_case.src.data.keras_cifar10_datahandler import KerasCIFAR10DataHandler
from use_case.src.model.cnn import CNN_cifar10
from trainer import Trainer

def main():

    data_handler_args = {
        "data_dir": data_dir,
        "from_cache": True,
        "batch_size": 128,
        "val_split": 0.2,
        "rgb2grey": False,
        "use_onehot_encoder": True,
        "device": "cpu"
    }
    
    # Initialize data handler
    data_handler = KerasCIFAR10DataHandler(**data_handler_args)
    
    # Initialize model
    model = CNN_cifar10
    
    # Initialize trainer
    trainer = Trainer(data_handler, model)
    
    # Train model
    trainer.fit(epochs=100)
    
    # Test model
    trainer.test()
    
    # Save model
    trainer.save_model(root_use_case / "model/cnn")
    
    # Save data handler to joblib
    data_handler.to_joblib()

if __name__ == "__main__":
    main()

