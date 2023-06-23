import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from sklearn.preprocessing import RobustScaler

class prime():
    def __init__(self, saved_model = None, saved_insc = None, saved_tarsc = None) -> None:
        if saved_model is None:
            self.model = ks.models.load_model('prime_v0.1.0')
        else:
            self.model = ks.models.load_model(saved_model)
        if saved_insc is None:
            self.inscaler = RobustScaler()
            self.inscaler = self.inscaler.fit(np.load('primeinsc_v0.1.0.npy'))
        else:
            self.inscaler = RobustScaler()
            self.inscaler = self.inscaler.fit(np.load(saved_insc))
        if saved_tarsc is None:
            self.tarscaler = RobustScaler()
            self.tarscaler = self.tarscaler.fit(np.load('primetarsc_v0.1.0.npy'))
        else:
            self.tarscaler = RobustScaler()
            self.tarscaler = self.tarscaler.fit(np.load(saved_tarsc))
    def predict_func(self, X):
        X = self.inscaler.transform(X)
        return self.tarscaler.inverse_transform(self.model.predict(X))
    #Functions to mimic keras model
    def get_weights(self):
        return self.model.get_weights()
    def set_weights(self, weights):
        self.model.set_weights(weights)
    def get_config(self):
        return self.model.get_config()
    def save(self, path):
        self.model.save(path)
    def save_weights(self, path):
        self.model.save_weights(path)
    def summary(self):
        self.model.summary()
    def get_layer(self, name):
        return self.model.get_layer(name)
    def get_layer_weights(self, name):
        return self.model.get_layer(name).get_weights()
    def set_layer_weights(self, name, weights):
        self.model.get_layer(name).set_weights(weights)
    def get_layer_config(self, name):
        return self.model.get_layer(name).get_config()
    def get_layer_output(self, name, X):
        return self.model.get_layer(name).output(X)
    def get_layer_input(self, name, X):
        return self.model.get_layer(name).input(X)
    def get_layer_input_shape(self, name):
        return self.model.get_layer(name).input_shape
    def get_layer_output_shape(self, name):
        return self.model.get_layer(name).output_shape
    def get_layer_weights_shape(self, name):
        return self.model.get_layer(name).get_weights().shape
    #Functions to do predictions for given time range
    def predict(self, t0, t1, loc='bow shock'):
        'uh'