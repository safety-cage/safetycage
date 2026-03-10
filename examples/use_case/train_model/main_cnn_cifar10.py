import pyrootutils
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root_use_case = root / "use_case"
data_dir = root_use_case / "data/cifar10"

# may have to edit file path depending on where you are running from and where your venv/env is.
from use_case.src.data.keras_cifar10_datamodule import CIFAR10DataModule
from use_case.src.model.cnn import CNN_cifar10
from trainer import Trainer

def main():

    data_module_args = {
        "data_dir": data_dir,
        "from_cache": True,
        "batch_size": 128,
        "val_split": 0.2,
        "rgb2grey": False,
        "use_onehot_encoder": True,
        "device": "cpu"
    }
    
    # Initialize data module
    data_module = CIFAR10DataModule(**data_module_args)
    
    # Initialize model
    model = CNN_cifar10
    
    # Initialize trainer
    trainer = Trainer(data_module, model)
    
    # Train model
    trainer.fit(epochs=100)
    
    # Test model
    trainer.test()
    
    # Save model
    trainer.save_model(root_use_case / "model/cnn")
    
    # Save data module to joblib
    data_module.to_joblib()

if __name__ == "__main__":
    main()

