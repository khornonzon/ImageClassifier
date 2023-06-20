from DataLoader import DataLoader
from keras import layers, models
import tensorflow as tf

class NeuralNetwork:
    def __init__(self):
        self.data = DataLoader()
        self.model = models.Sequential()
    def configure(self):
        self.model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation='relu', input_shape=(32, 32, 3), kernel_initializer = 'he_uniform'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same' ))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same' ))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = 'same' ))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(7, activation = 'softmax'))
        self.model.summary()

    def train(self):
        self.configure()
        self.model.compile(optimizer='adam',
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

        self.model.fit(self.data.x_train, self.data.y_train, epochs=50)
