from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class EmotionModelV1:
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear",
                     "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, weights_path):
        self.model = self._build_model()
        self.model.load_weights(weights_path)

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.EMOTIONS_LIST), activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict_emotion(self, img):
        preds = self.model.predict(img)
        return EmotionModelV1.EMOTIONS_LIST[np.argmax(preds)]


class EmotionModelV2:
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear",
                     "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, weights_path):
        self.model = self._build_model()
        self.model.load_weights(weights_path)

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))  
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def predict_emotion(self, img):
        preds = self.model.predict(img)
        return EmotionModelV2.EMOTIONS_LIST[preds.argmax()]
 
