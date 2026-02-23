from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

MLP = Sequential(
    [
    Dense(256, input_shape=(28*28*1,), activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
    ],
    name="mlp"
)

