from keras import optimizers, metrics, losses, callbacks, saving
from tensorflow import timestamp
import tensorflow as tf
from libs import datasets_keras
from libs.config import USE_ELEVATION
from libs.util_keras import FBeta
import time
import numpy as np

class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs=None):
        # Initialize an empty list to store execution times
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        # Record the exact start time of the epoch
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Calculate duration and append to the list
        duration = time.time() - self.epoch_start_time
        self.times.append(duration)

def get_early_stopping_callback():
    return callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True, mode = 'min')

class train_for_time(callbacks.Callback):
    """callback to terminate training after a time limit is reached

    Can be used to control how long training runs for, and will terminate
    training once a specified time limit is reached.
    """
    def __init__(
        self,
        train_time_mins=15,
    ):
        super().__init__()

        self.train_time_mins = train_time_mins
        self.epochs = 0
        self.train_time = 0
        self.end_early = False

    def on_train_begin(self, logs=None):
        # save the start time
        self.start_time = timestamp()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1
        current_time = timestamp()
        training_time = (current_time - self.start_time)
        if (training_time / 60) > self.train_time_mins:
            self.train_time = current_time - self.start_time
            self.model.stop_training = True
            self.end_early = True

@saving.register_keras_serializable(package='losses')
class FocalLoss(losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.exp(-ce)
        return self.alpha * tf.pow(1 - p_t, self.gamma) * ce

    def get_config(self):
        return {**super().get_config(), 'gamma': self.gamma, 'alpha': self.alpha}

def train_model(dataset, model, epochs=15, lr=1e-4, bs=12, use_elevation=USE_ELEVATION, time_history=None):

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=FocalLoss(),
        metrics=['Accuracy']
    )

    train_data, valid_data = datasets_keras.load_dataset(dataset, bs, use_elevation=use_elevation)

    #for batch in train_data:
    #    print(batch)

    if time_history is None:
        history = model.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs, 
            callbacks=[train_for_time(60), get_early_stopping_callback()]
        )
    else:
        history = model.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs, 
            callbacks=[train_for_time(60), get_early_stopping_callback(), time_history]
        )

    return history
