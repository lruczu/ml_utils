from typing import List
import keras

# keras=2.0.8, tensorflow=1.4.0


def create_model(n_features: int, n_classes: int, hidden_dims: List[int]):
    i = keras.layers.Input(shape=(n_features,), sparse=True, dtype='float32')
    ds = None
    for h in hidden_dims:
        if ds is not None:
            ds = keras.layers.Dense(h, activation='relu')(ds)
        else:
            ds = keras.layers.Dense(h, activation='relu')(i)

    o = keras.layers.Dense(n_classes, activation='softmax')(ds)

    return keras.models.Model(i, o)
