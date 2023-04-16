from tensorflow import keras


class MyModel(keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = keras.layers.Dense()