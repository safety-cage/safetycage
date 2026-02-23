import pyrootutils
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root_use_case = root / "use_case"
data_dir = root_use_case / "data/mnist"
model_dir = root_use_case / "model/mlp"
from use_case.src.data.keras_mnist_datahandler import KerasMNISTDataHandler
from use_case.src.model.mlp import MLP
from trainer import Trainer

def main():

    data_handler_args = {
        "data_dir": data_dir,
        "from_cache": True,
        "batch_size": 128,
        "val_split": 0.2,
        "use_onehot_encoder": True,
        "device": "cpu"
    }
    
    # Initialize data handler
    data_handler = KerasMNISTDataHandler(**data_handler_args)
    
    # Initialize model
    model = MLP
    
    # Initialize trainer
    trainer = Trainer(data_handler, model)
    
    # Train model
    trainer.fit(epochs=10)
    
    # Test model
    trainer.test()
    
    # Save model
    trainer.save_model(model_dir)
    
    # Save data handler to joblib
    data_handler.to_joblib()

if __name__ == "__main__":
    main()

