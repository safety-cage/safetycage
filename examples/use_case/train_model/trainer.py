
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


class Trainer:
    def __init__(
        self,
        data_handler, 
        model,
        model_path = None
        ):
        self.data_handler = data_handler
        self.model = model
        
    def fit(self, epochs=100):
        # Compile model
        # Use categorical_crossentropy since we're using one-hot encoded labels
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        
        # Train model
        history = self.model.fit(
            self.data_handler.train_dataset(),
            validation_data=self.data_handler.val_dataset(),
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
        )
        
        return history

    def test(self, dataset=None):
        if dataset is None:
            dataset = self.data_handler.test_dataset()

        # Get predictions and true labels
        y_test_pred = []
        y_test_true = []
        
        for batch in dataset:

            x_batch, y_batch = batch

            # Get predictions for batch
            batch_preds = self.model.predict(x_batch, verbose=0)
            batch_pred_classes = np.argmax(batch_preds, axis=1)
            batch_true_classes = np.argmax(y_batch, axis=1)
            
            y_test_pred.extend(batch_pred_classes)
            y_test_true.extend(batch_true_classes)
        
        # Convert to numpy arrays
        y_test_pred = np.array(y_test_pred)
        y_test_true = np.array(y_test_true)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_true, y_test_pred)
        cm = confusion_matrix(y_test_true, y_test_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        return y_test_pred

    def predict(self, dataset):
        
        # Get predictions 
        y_pred = []
        
        # Iterate through the dataset to get predictions and true labels
        for x_batch, _ in dataset:
            batch_preds = self.model.predict(x_batch)
            batch_pred_classes = np.argmax(batch_preds, axis=1)
            
            y_pred.extend(batch_pred_classes)

        # Convert to numpy arrays for metrics
        y_pred = np.array(y_pred)

        return y_pred

    def predict_all(self):
        y_pred_train = self.predict(
            dataset = self.data_handler.train_dataset()
        )
        
        # Convert predictions to one-hot encoding
        y_pred_train = tf.keras.utils.to_categorical(y_pred_train, num_classes=10)
        
        y_pred_val = self.predict(
            dataset = self.data_handler.val_dataset()
        )
        y_pred_val = tf.keras.utils.to_categorical(y_pred_val, num_classes=10)

        y_pred_test = self.predict(
            dataset = self.data_handler.test_dataset()
        )
        y_pred_test = tf.keras.utils.to_categorical(y_pred_test, num_classes=10)
        
        return {
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "y_pred_test": y_pred_test
        }
        
    def save_model(self, model_path):
        self.model.save(model_path)
        
    def run(self):
        
        # Train model
        print("Training model...")
        self.train()        
        
        # Test model
        print("Testing model...")
        self.test()

