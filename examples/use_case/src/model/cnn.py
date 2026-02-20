from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D

CNN_mnist = Sequential(
    [
    # Reshape the input to be compatible with Conv2D (assuming MNIST data)
    Reshape((28,28,1), input_shape=(28*28*1,)),
    
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # Second convolutional layer
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Thrid convolutional layer
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    
    # Flatten the output for the dense layers
    Flatten(),
    

    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
    ],
    name="cnn"
)

CNN_cifar10 = Sequential(
    [
    # Reshape the input to be compatible with Conv2D (assuming CIFAR-10 data)
    Reshape((32,32,3), input_shape=(32*32*3,)),
    
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    # Second convolutional layer
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Thrid convolutional layer
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    
    # Flatten the output for the dense layers
    Flatten(),
    

    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
    ],
    name="cnn"
)