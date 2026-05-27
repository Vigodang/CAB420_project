from keras import optimizers, metrics
from libs import datasets_keras
from libs.config import USE_ELEVATION
from libs.util_keras import FBeta
import numpy as np


def train_model(dataset, model, use_elevation=USE_ELEVATION):
    epochs = 15
    lr     = 1e-4
    bs     = 8 # reduce this if you are running out of GPU memory

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=[
            metrics.Precision(top_k=1, name='precision'),
            metrics.Recall(top_k=1, name='recall'),
            FBeta(name='f_beta')
        ]
    )

    train_data, valid_data = datasets_keras.load_dataset(dataset, bs, use_elevation=use_elevation)
    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs
    )
