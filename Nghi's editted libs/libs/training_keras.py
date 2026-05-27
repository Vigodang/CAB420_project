from keras import optimizers, metrics, losses
from libs import datasets_keras
from libs.config import USE_ELEVATION
from libs.util_keras import FBeta
import numpy as np


def train_model(dataset, model, use_elevation=USE_ELEVATION):
    epochs = 1
    lr     = 1e-4
    bs     = 12 # reduce this if you are running out of GPU memory

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['Accuracy']
    )

    train_data, valid_data = datasets_keras.load_dataset(dataset, bs, use_elevation=use_elevation)

    #for batch in train_data:
    #    print(batch)

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs
    )
